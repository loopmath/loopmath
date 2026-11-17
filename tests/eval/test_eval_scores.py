"""`eval scores import | check | add | export`: the study folder, its checks and its way out to OCP.

Every refusal message is pinned. All data is synthetic.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import scores as S
from loopmath.eval.adapters import ocp as A

STUDY = Path(__file__).resolve().parent / "data" / "study-small"
RUBRIC = str(STUDY / "rubric.json")
JUDGES = str(STUDY / "judges.json")
AT = "2026-02-03T10:00:00-08:00"
TABLES = (C.RUNS_FILE, C.SCORES_FILE, C.PAIRS_FILE, C.ITEMS_FILE)


def scores(capsys, *argv):
    code = cli.main(["eval", "scores", *(str(arg) for arg in argv)])
    out, err = capsys.readouterr()
    return code, out, err


def scores_json(capsys, *argv):
    code, out, err = scores(capsys, *argv, "--json")
    return code, json.loads(out)


def copy_study(tmp_path: Path, name: str = "study") -> Path:
    return Path(shutil.copytree(STUDY, tmp_path / name))


def snapshot(folder: Path) -> dict:
    return {name: (folder / name).read_bytes() for name in TABLES if (folder / name).exists()}


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


def write_documents(folder: Path, study: C.Study, persons=("reviewer",)) -> list[Path]:
    entries, _ = A.export(study.runs, study.scores, study.pairs, persons=persons)
    rows = {row["run"]: row for row in study.runs}
    folder.mkdir()
    paths = []
    for entry in entries:
        paths.append(folder / f"{entry['run']}.ocp.json")
        paths[-1].write_text(json.dumps(A.merge(document(rows[entry["run"]]), entry)), encoding="utf-8")
    return paths


def write_csv(path: Path, header: str, *lines: str) -> Path:
    path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
    return path


def small_study(tmp_path: Path, runs=("r1:case-01:baseline", "r2:case-01:strong", "r3:case-02:baseline")) -> Path:
    folder = tmp_path / "small"
    rows = []
    for spec in runs:
        name, input_id, arm = spec.split(":")
        rows.append({"run": name, "input": input_id, "arm": arm,
                     "steps": [{"step": "final", "model": {"id": f"model-{arm}", "family": arm}}]})
    C.write_lines(folder / C.RUNS_FILE, C.SCHEMA_RUNS, rows)
    return folder


def by_id(rows: list[dict]) -> dict:
    return {row["id"]: row for row in rows}


# ---------------------------------------------------------------------------------------------------
# import: OCP documents (acceptance check 2, through the commands)

def test_import_of_exported_documents_gives_the_study_back_and_a_second_import_changes_nothing(tmp_path, capsys):
    study = C.read_study(STUDY)
    paths = write_documents(tmp_path / "ocp", study)
    out = tmp_path / "study"
    code, report = scores_json(capsys, "import", "--ocp", *paths, "--out", out)
    assert code == 0 and report["ok"] is True and report["schema"] == "loopmath.eval-scores-import/1"
    assert report["runs"] == {"new": 16, "unchanged": 0, "refused": 0}
    assert report["readings"] == {"new": 305, "unchanged": 0, "refused": 0}
    assert report["pairs"] == {"new": 18, "unchanged": 0, "refused": 0}
    assert report["items"] == {"new": 0, "unchanged": 0, "refused": 0}
    assert report["refused"] == [] and report["not_used"] == []
    assert report["written"] == [C.RUNS_FILE, C.SCORES_FILE, C.PAIRS_FILE]
    back = C.read_study(out)
    assert by_id(back.scores) == by_id(study.scores) and by_id(back.pairs) == by_id(study.pairs)
    assert [(r["run"], r["input"], r["arm"], r["steps"]) for r in back.runs] == [
        (r["run"], r["input"], r["arm"], r["steps"]) for r in study.runs]

    before = snapshot(out)
    code, text, err = scores(capsys, "import", "--ocp", *paths, "--out", out)
    assert code == 0 and err == ""
    assert text.splitlines()[1:] == [
        "runs: 0 new, 16 unchanged, 0 refused", "readings: 0 new, 305 unchanged, 0 refused",
        "pairs: 0 new, 18 unchanged, 0 refused", "items: 0 new, 0 unchanged, 0 refused", "nothing new: no file changed"]
    assert snapshot(out) == before

    # into the study the documents were written from: the runs there say more (cost, refs) and stay as they are
    home = copy_study(tmp_path, "home")
    before = snapshot(home)
    code, report = scores_json(capsys, "import", "--ocp", *paths, "--out", home)
    assert code == 0 and report["runs"]["unchanged"] == 16 and report["readings"]["unchanged"] == 305
    assert report["written"] == [] and snapshot(home) == before

    # and out again: the same signals and preferences as from the first study
    shutil.copy(JUDGES, out / "judges.json")
    first = [json.loads(line) for line in scores(capsys, "export", STUDY)[1].splitlines()]
    second = [json.loads(line) for line in scores(capsys, "export", out)[1].splitlines()]
    assert len(first) == 16 and [entry["run"] for entry in first] == [entry["run"] for entry in second]
    for one, two in zip(first, second):
        assert by_id(one["signals"]) == by_id(two["signals"])
        assert by_id(one["preferences"]) == by_id(two["preferences"]) and one.get("slate") == two.get("slate")
    assert scores(capsys, "check", out, "--rubric", RUBRIC, "--judges", JUDGES)[0] == 0


def test_import_refuses_an_id_seen_before_with_other_content_and_writes_nothing(tmp_path, capsys):
    home = copy_study(tmp_path)
    study = C.read_study(home)
    changed = dict(study.scores[0], value=1)
    new = dict(study.scores[1], id="s9001", call="c9001")
    entries, _ = A.export(study.runs, [changed, new], [])
    path = tmp_path / "baseline-01.ocp.json"
    path.write_text(json.dumps(A.merge(document(study.runs[0]), entries[0])), encoding="utf-8")
    before = snapshot(home)
    code, out, err = scores(capsys, "import", "--ocp", path, "--out", home)
    assert code == 1
    assert err == ('error: baseline-01.ocp.json: the reading id "s0001" is already in the study with other content; '
                   "give a changed reading a new id, and name the one it corrects in replaces\n")
    assert "readings: 1 new, 0 unchanged, 1 refused" in out
    assert out.splitlines()[-1] == "nothing written: a refused row refuses the import"
    assert snapshot(home) == before
    code, report = scores_json(capsys, "import", "--ocp", path, "--out", home)
    assert code == 1 and report["ok"] is False and report["written"] == []
    assert [(entry["table"], entry["id"], entry["source"]) for entry in report["refused"]] == [
        ("readings", "s0001", "baseline-01.ocp.json")]
    assert snapshot(home) == before

    other = dict(study.runs[0], arm="strong")
    path.write_text(json.dumps(document(other)), encoding="utf-8")
    code, out, err = scores(capsys, "import", "--ocp", path, "--out", home)
    assert code == 1 and err == ('error: baseline-01.ocp.json: the run id "baseline-01" is already in the study with '
                                 "other content; give a changed run a new id\n")


def test_import_refuses_a_row_that_breaks_its_contract_and_names_the_source(tmp_path, capsys):
    study = C.read_study(STUDY)
    doc = document(study.runs[0], signals=[{
        "id": "g1", "kind": "score", "name": "depth", "value": 4, "observed_at": AT, "tier": "reported",
        "source": {"kind": "judge", "ref": "judge strong"}, "ext": {A.NS: {"rubric": "brief@1"}}}])
    path = tmp_path / "r.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    code, out, err = scores(capsys, "import", "--ocp", path, "--out", tmp_path / "study")
    assert code == 1 and not (tmp_path / "study").exists()
    assert err.startswith('error: r.json: reading "g1": judge: ') and "readings: 0 new, 0 unchanged, 1 refused" in out


def test_a_simulated_run_needs_allow_simulated(tmp_path, capsys):
    study = C.read_study(STUDY)
    doc = document(study.runs[0])
    doc["run"]["ext"] = {"dev.loopmath.simulate": {"seed": 1}}
    path = tmp_path / "sim.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "study"
    code, _, err = scores(capsys, "import", "--ocp", path, "--out", out)
    assert code == 1 and not out.exists()
    assert err == 'error: sim.json: the run "baseline-01" is simulated; --allow-simulated takes simulated runs\n'
    code, report = scores_json(capsys, "import", "--ocp", path, "--out", out, "--allow-simulated")
    assert code == 0 and report["runs"]["new"] == 1 and C.read_study(out).runs[0]["simulated"] is True
    real = tmp_path / "real.json"
    real.write_text(json.dumps(document(study.runs[1])), encoding="utf-8")
    code, _, err = scores(capsys, "import", "--ocp", real, "--out", out)
    assert code == 1 and err == ('error: real.json: the study holds simulated runs and the run "strong-01" is not '
                                 "simulated; --allow-simulated mixes the two\n")


def test_import_file_problems_are_refusals_that_write_nothing(tmp_path, capsys):
    out = tmp_path / "study"
    assert scores(capsys, "import", "--out", out) == (
        1, "", "error: eval scores import: no source given; give --ocp, --rows, --csv, --pairs-csv or --items\n")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(C.EXAMPLES[C.SCHEMA_LABELS]["valid"][0]), encoding="utf-8")
    items = tmp_path / "items.jsonl"
    items.write_text("", encoding="utf-8")
    assert scores(capsys, "import", "--labels", plan, "--items", items, "--out", out) == (
        1, "", "error: eval scores import: --labels: a labels plan is joined to a filled form; give the form with "
               "--csv or --pairs-csv\n")
    assert scores(capsys, "import", "--csv", tmp_path / "none.csv", "--rubric", RUBRIC, "--out", out) == (
        1, "", f"error: none.csv: no such file; check the path {tmp_path / 'none.csv'}\n")
    bad = write_csv(tmp_path / "bad.csv", "run,component,judge", "r1,depth,judge-a")
    assert scores(capsys, "import", "--csv", bad, "--rubric", RUBRIC, "--out", out) == (
        1, "", 'error: bad.csv: line 1: no column "value"; give the columns run, step, component, judge, call, '
               "value, item, of, observed_at\n")
    fine = write_csv(tmp_path / "fine.csv", "run,component,judge,value", "r1,depth,judge-a,4")
    assert scores(capsys, "import", "--csv", fine, "--out", out) == (
        1, "", "error: fine.csv: the rows name no rubric; give --rubric, or a rubric column with id@version\n")
    assert scores(capsys, "import", "--csv", fine, "--rubric", RUBRIC, "--pair-component", "speed", "--out", out) == (
        1, "", 'error: eval scores import: --pair-component: unknown component "speed"; use one of numbers_right, '
               "clarity, depth, overall\n")
    twice = write_csv(tmp_path / "twice.csv", "run,component,value,value", "r1,depth,4,5")
    assert scores(capsys, "import", "--csv", twice, "--rubric", RUBRIC, "--out", out) == (
        1, "", 'error: twice.csv: line 1: the column "value" appears twice; give every column once\n')
    assert scores(capsys, "import", "--ocp", tmp_path / "none.json", "--out", out) == (
        1, "", f"error: none.json: no such file; check the path {tmp_path / 'none.json'}\n")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    code, text, err = scores(capsys, "import", "--ocp", broken, "--out", out)
    assert (code, text) == (1, "") and err.startswith("error: broken.json: not valid JSON")
    assert not out.exists()
    code, payload = scores_json(capsys, "import", "--out", out)
    assert code == 1 and payload["schema"] == "loopmath.error/1"


# ---------------------------------------------------------------------------------------------------
# import: CSV files, forms and items

def test_import_csv_reads_values_on_the_rubric_scale_and_is_the_same_the_second_time(tmp_path, capsys):
    folder = small_study(tmp_path)
    table = write_csv(
        tmp_path / "readings.csv", "run,step,component,judge,call,value,item,of,observed_at,notes",
        f"r1,final,clarity,reviewer,k1,good,,,{AT},fine",
        f"r1,final,depth,reviewer,k1,7.5,,,{AT},",
        f"r1,final,numbers_right,reviewer,k1,1,,,{AT},",
        "r2,,depth,reviewer,,6,,,,",
        "r2,,depth,reviewer,,6,,,,",
        "r2,,clarity,reviewer,,,,,,",
        "")
    code, report = scores_json(capsys, "import", "--csv", table, "--rubric", RUBRIC, "--out", folder)
    assert code == 0 and report["readings"] == {"new": 5, "unchanged": 0, "refused": 0}
    assert report["ignored_columns"] == {"readings.csv": ["notes"]}
    assert report["not_used"] == [{"source": "readings.csv line 7", "kind": "row", "id": None, "reason": "no value"}]
    rows = C.read_study(folder).scores
    assert [(row["component"], row["value"], row["call"]) for row in rows[:3]] == [
        ("clarity", "good", "k1"), ("depth", 7.5, "k1"), ("numbers_right", 1, "k1")]
    assert all(row["rubric"] == "brief@1" and row["id"].startswith("r-") for row in rows)
    # two equal lines are two readings, each its own call (a repeat), with the file's time
    assert rows[3]["id"] != rows[4]["id"] and rows[3]["call"] == rows[3]["id"] and rows[4]["call"] == rows[4]["id"]
    assert rows[3]["step"] == "final" and rows[3]["observed_at"] == rows[4]["observed_at"] != AT
    before = snapshot(folder)
    code, report = scores_json(capsys, "import", "--csv", table, "--rubric", RUBRIC, "--out", folder)
    assert code == 0 and report["readings"] == {"new": 0, "unchanged": 5, "refused": 0} and snapshot(folder) == before
    os.utime(table, (1_700_000_000, 1_700_000_000))  # a copy of the file has another time: still the same rows
    code, report = scores_json(capsys, "import", "--csv", table, "--rubric", RUBRIC, "--out", folder)
    assert code == 0 and report["readings"] == {"new": 0, "unchanged": 5, "refused": 0} and snapshot(folder) == before
    assert scores(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES)[0] == 0


def test_import_csv_with_ids_items_and_a_rubric_column(tmp_path, capsys):
    folder = small_study(tmp_path)
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps({"schema": C.SCHEMA_ITEMS, "id": name, "input": "case-01",
                                         "component": "facts"}) + "\n" for name in ("i1", "i2")), encoding="utf-8")
    table = write_csv(
        tmp_path / "readings.csv", "id,run,component,judge,value,item,of,observed_at,rubric,replaces,cost_usd",
        f"s1,r1,facts,judge-a,1,i1,,{AT},facts@2,,0.01", f"s2,r1,facts,judge-a,0,i2,,{AT},facts@2,,",
        f"s3,r2,facts,judge-a,7,,9,{AT},facts@2,,", f"s4,r2,facts,judge-a,8,,9,{AT},facts@2,s3,")
    code, report = scores_json(capsys, "import", "--csv", table, "--items", items, "--out", folder)
    assert code == 0 and report["readings"]["new"] == 4 and report["items"] == {"new": 2, "unchanged": 0, "refused": 0}
    assert report["written"] == [C.SCORES_FILE, C.ITEMS_FILE]
    rows = by_id(C.read_study(folder).scores)
    assert rows["s1"]["call"] == rows["s2"]["call"] != "s1"  # the item lines of one target share a call
    assert (rows["s1"]["item"], rows["s1"]["cost_usd"], rows["s3"]["of"]) == ("i1", 0.01, 9)
    assert rows["s4"]["replaces"] == "s3"
    assert [row["id"] for row in C.read_study(folder).items] == ["i1", "i2"]
    code, report = scores_json(capsys, "import", "--csv", table, "--items", items, "--out", folder)
    assert code == 0 and report["readings"]["unchanged"] == 4 and report["items"]["unchanged"] == 2
    items.write_text(json.dumps({"schema": C.SCHEMA_ITEMS, "id": "i1", "input": "case-02", "component": "facts"})
                     + "\n", encoding="utf-8")
    code, _, err = scores(capsys, "import", "--items", items, "--out", folder)
    assert code == 1 and err == ('error: items.jsonl: the item id "i1" is already in the study with other content; '
                                 "give a changed item a new id\n")


def test_import_pairs_csv_reads_verdicts_and_left_right_choices(tmp_path, capsys):
    folder = small_study(tmp_path)
    table = write_csv(
        tmp_path / "pairs.csv", "a,b,step,component,judge,call,shown_first,verdict,choice,observed_at",
        f"r1,r2,final,overall,reviewer,q1,a,b,,{AT}", f"r1,r2,final,overall,reviewer,q2,b,,left,{AT}",
        f"r1,r2,final,overall,reviewer,q3,b,,right,{AT}", f"r1,r2,final,,reviewer,q4,,tie,,{AT}",
        f"r1,r2,final,overall,reviewer,q5,,,,{AT}")
    code, report = scores_json(capsys, "import", "--pairs-csv", table, "--rubric", RUBRIC, "--out", folder)
    assert code == 0 and report["pairs"] == {"new": 4, "unchanged": 0, "refused": 0}
    assert [entry["reason"] for entry in report["not_used"]] == ["no verdict"]
    rows = C.read_study(folder).pairs
    assert [(row["call"], row["shown_first"], row["verdict"], row["component"]) for row in rows] == [
        ("q1", "a", "b", "overall"), ("q2", "b", "b", "overall"), ("q3", "b", "a", "overall"),
        ("q4", None, "tie", None)]
    bad = write_csv(tmp_path / "bad.csv", "a,b,judge,choice", "r1,r2,reviewer,left")
    code, _, err = scores(capsys, "import", "--pairs-csv", bad, "--rubric", RUBRIC, "--out", folder)
    assert code == 1 and err.startswith('error: bad.csv line 2: the choice "left" is not left, right or tie, or the '
                                        "row has no shown_first to read left and right by")


def test_import_joins_a_filled_form_to_its_labels_plan(tmp_path, capsys):
    folder = small_study(tmp_path, runs=("baseline-01:case-01:baseline", "strong-01:case-01:strong"))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(C.EXAMPLES[C.SCHEMA_LABELS]["valid"][0]), encoding="utf-8")
    form = write_csv(tmp_path / "form.csv", "assignment,component,value", "a1,clarity,ok", "a1,depth,6", "a1,overall,")
    pairs = write_csv(tmp_path / "pairs-form.csv", "assignment,choice", "a2,left")
    code, report = scores_json(capsys, "import", "--csv", form, "--pairs-csv", pairs, "--labels", plan, "--rubric",
                               RUBRIC, "--out", folder)
    assert code == 0 and report["readings"]["new"] == 2 and report["pairs"]["new"] == 1
    study = C.read_study(folder)
    assert [(r["run"], r["step"], r["judge"], r["assignment"], r["component"], r["value"]) for r in study.scores] == [
        ("baseline-01", "final", "reviewer", "a1", "clarity", "ok"),
        ("baseline-01", "final", "reviewer", "a1", "depth", 6)]
    pair = study.pairs[0]
    assert (pair["a"], pair["b"], pair["judge"], pair["assignment"], pair["shown_first"], pair["verdict"],
            pair["component"]) == ("baseline-01", "strong-01", "reviewer", "a2", "b", "b", "overall")
    assert scores(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES, "--require-assignment")[0] == 0

    # a form that names the target and not the assignment is joined by target, component and reviewer
    other = small_study(tmp_path / "other", runs=("baseline-01:case-01:baseline", "strong-01:case-01:strong"))
    named = write_csv(tmp_path / "named.csv", "run,component,value", "baseline-01,depth,6")
    code, report = scores_json(capsys, "import", "--csv", named, "--labels", plan, "--rubric", RUBRIC, "--out", other)
    assert code == 0 and C.read_study(other).scores[0]["assignment"] == "a1"

    wrong = write_csv(tmp_path / "wrong.csv", "assignment,component,value,run", "a9,depth,6,", "a1,overall,6,",
                      "a1,depth,6,strong-01", "a2,depth,6,")
    code, _, err = scores(capsys, "import", "--csv", wrong, "--labels", plan, "--rubric", RUBRIC, "--out", other)
    assert code == 1 and err.splitlines() == [
        'error: wrong.csv line 2: no score assignment "a9" in the plan',
        'error: wrong.csv line 3: the assignment "a1" does not ask for the component "overall"',
        'error: wrong.csv line 4: run "strong-01" does not match the assignment "a1", which has "baseline-01"',
        'error: wrong.csv line 5: no score assignment "a2" in the plan']


def test_import_rows_reads_harness_runs_rows(tmp_path, capsys):
    row = {"schema": "loopmath.harness-runs/0.1", "task_id": "case-01", "arm": "baseline", "attempt": 1,
           "harness": "cli", "model": "model-small", "effort": "low", "started_at": "2026-02-03T09:00:00-08:00",
           "ended_at": "2026-02-03T09:05:00-08:00", "cost": {"usd": 0.2},
           "outcome": {"accepted": True, "judges": [{"id": "judge-strong", "score": 6, "observed_at": AT}]}}
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, arm="strong")) + "\n", encoding="utf-8")
    out = tmp_path / "study"
    code, report = scores_json(capsys, "import", "--rows", path, "--rubric", RUBRIC, "--out", out)
    assert code == 0 and report["runs"]["new"] == 2 and report["readings"]["new"] == 2
    study = C.read_study(out)
    assert [(r["input"], r["arm"], r["cost_usd"]) for r in study.runs] == [("case-01", "baseline", 0.2),
                                                                           ("case-01", "strong", 0.2)]
    assert {(r["component"], r["judge"], r["value"], r["step"]) for r in study.scores} == {
        ("overall", "judge-strong", 6, "final")}
    code, report = scores_json(capsys, "import", "--rows", path, "--rubric", RUBRIC, "--out", out)
    assert code == 0 and report["runs"]["unchanged"] == 2 and report["readings"]["unchanged"] == 2
    path.write_text(json.dumps({"schema": "loopmath.harness-runs/0.1"}) + "\n", encoding="utf-8")
    code, _, err = scores(capsys, "import", "--rows", path, "--out", tmp_path / "none")
    assert code == 1 and err.startswith("error: rows.jsonl: line 1: ") and not (tmp_path / "none").exists()


@pytest.mark.parametrize("text, component, value", [
    ("good", "clarity", "good"), ("7", "depth", 7), ("7.5", "depth", 7.5), ("1e1", "depth", 10.0), ("-2", None, -2),
    ("high", "depth", "high"), ("nan", "depth", "nan"), ("inf", None, "inf"), ("1", "numbers_right", 1),
    ("9" * 400, None, int("9" * 400)), ("1e999", None, "1e999"),
], ids=["level", "int", "float", "exponent", "negative", "word", "nan", "inf", "binary", "long", "overflow"])
def test_parse_value(text, component, value):
    rubric = C.read_document(RUBRIC, C.SCHEMA_RUBRIC)
    comp = None if component is None else C.component(rubric, component)
    got = S.parse_value(text, comp)
    assert got == value and type(got) is type(value)


# ---------------------------------------------------------------------------------------------------
# check

def test_check_passes_the_small_study_and_prints_the_design_table(capsys):
    code, out, err = scores(capsys, "check", STUDY, "--rubric", RUBRIC, "--judges", JUDGES)
    assert code == 0 and err == ""
    lines = out.splitlines()
    assert lines[0].endswith("16 runs, 305 readings (304 read), 18 pairs (18 read), 0 items")
    assert lines[1:3] == ["rubric brief@1", "mixed_arm: none"] and lines[-1] == "ok"
    assert lines[5].split() == ["term", "judge", "dimension", "count", "met", "needs"]
    code, report = scores_json(capsys, "check", STUDY, "--rubric", RUBRIC, "--judges", JUDGES)
    assert code == 0 and report["ok"] is True and report["schema"] == "loopmath.eval-scores-check/1"
    assert report["problems"] == [] and report["warnings"] == [] and report["mixed_arm"] == []
    assert report["not_read"] == [{"table": "readings", "id": "s0021", "reason": "replaced", "by": "s0305"}]
    design = {(row["term"], row["judge"], row["dimension"]): row for row in report["design"]}
    assert design[("run_vs_input", None, None)]["count"] == 8 and design[("run_vs_input", None, None)]["met"]
    assert design[("sigma", "reviewer", "final/depth")] == {
        "term": "sigma", "judge": "reviewer", "dimension": "final/depth", "count": 2, "met": True, "targets": 10,
        "needs": "2 or more calls by the judge on the same target and component"}
    assert design[("tau", "judge-cheap", "final/clarity")]["met"] is True
    assert design[("judge_input", "judge-strong", "final/overall")]["count"] == 8
    assert design[("judge_input", "judge-strong", "final/overall")]["met"] is False  # 8 inputs, 10 needed
    anchor = design[("anchor", None, "final/numbers_right")]
    assert (anchor["count"], anchor["state"], anchor["met"]) == (10, "weak", True)
    assert design[("binary_status", None, "final/numbers_right")]["met"] is True
    assert design[("family", None, None)]["count"] == 2 and design[("family", None, None)]["met"] is True
    assert design[("position", "judge-strong", "final/overall")]["met"] is True
    assert design[("sigma_pair", "reviewer", "final/overall")]["met"] is False
    assert design[("tie_width", "judge-strong", "final/overall")]["count"] == 3
    assert not any(row["term"] in ("halo", "reference_family") for row in report["design"])
    # without a judges file the rows that need it are absent, and the study still passes
    code, report = scores_json(capsys, "check", STUDY, "--rubric", RUBRIC)
    assert code == 0 and not any(row["term"] in ("anchor", "family") for row in report["design"])


def edit_lines(folder: Path, name: str, change) -> None:
    path = folder / name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    path.write_text("".join(json.dumps(row) + "\n" for row in change(rows)), encoding="utf-8")


def reading(ident: str, **more) -> dict:
    return {"schema": C.SCHEMA_SCORES, "id": ident, "run": "baseline-01", "step": "final", "component": "depth",
            "judge": "judge-strong", "call": f"c-{ident}", "value": 5, "rubric": "brief@1", "observed_at": AT, **more}


def pair(ident: str, **more) -> dict:
    return {"schema": C.SCHEMA_PAIRS, "id": ident, "a": "baseline-01", "b": "strong-01", "step": "final",
            "component": "overall", "judge": "judge-strong", "call": f"c-{ident}", "shown_first": "a",
            "verdict": "a", "rubric": "brief@1", "observed_at": AT, **more}


CHECKS = [
    ("scores", reading("x1", component="speed"),
     'scores.jsonl: id "x1": component: unknown component "speed"; use one of numbers_right, clarity, depth, overall'),
    ("scores", reading("x1", judge="judge-new"),
     'scores.jsonl: id "x1": judge: unknown judge "judge-new"; use one of reviewer, judge-strong, judge-cheap, or '
     "add the judge to the judges file"),
    ("scores", reading("x1", step="plan"),
     'scores.jsonl: id "x1": step: unknown step "plan"; use one of draft, final'),
    ("scores", reading("x1", rubric="brief@2"),
     'scores.jsonl: id "x1": rubric: unknown rubric version "brief@2"; this check reads "brief@1": give the rubric '
     "the row was scored under"),
    ("scores", reading("x1", run="nobody"),
     'scores.jsonl: id "x1": run: no run "nobody" in runs.jsonl; import the run, or fix the reading'),
    ("scores", reading("x1", replaces="s9999"),
     'scores.jsonl: id "x1": replaces: "s9999" names no reading; name the id of the reading it corrects'),
    ("scores", reading("x1", call="c0001", component="numbers_right", value=1, judge="judge-strong"),
     'scores.jsonl: call "c0001": two readings of one call for the component "numbers_right" ("s0001" and "x1"); '
     "give a repeat its own call, or name the reading it corrects in replaces"),
    ("pairs", pair("y1", b="nobody"),
     'pairs.jsonl: id "y1": b: no run "nobody" in runs.jsonl; import the run, or fix the pair'),
    ("pairs", pair("y1", b="strong-02"),
     'pairs.jsonl: id "y1": the runs "baseline-01" and "strong-02" have different inputs ("case-01" and "case-02"); '
     "compare two runs of one input"),
    ("pairs", pair("y1", component="speed"),
     'pairs.jsonl: id "y1": component: unknown component "speed"; use one of numbers_right, clarity, depth, overall'),
    ("pairs", pair("y1", judge="judge-new"),
     'pairs.jsonl: id "y1": judge: unknown judge "judge-new"; use one of reviewer, judge-strong, judge-cheap, or '
     "add the judge to the judges file"),
    ("pairs", pair("y1", replaces="p999"),
     'pairs.jsonl: id "y1": replaces: "p999" names no pair; name the id of the pair it corrects'),
]


@pytest.mark.parametrize("table, row, message", CHECKS, ids=[
    "component", "judge", "step", "rubric", "no-run", "replaces", "call-twice", "pair-no-run", "pair-inputs",
    "pair-component", "pair-judge", "pair-replaces"])
def test_check_refuses_by_name(tmp_path, capsys, table, row, message):
    folder = copy_study(tmp_path)
    edit_lines(folder, f"{table}.jsonl", lambda rows: [*rows, row])
    code, out, err = scores(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES)
    assert code == 1 and out.splitlines()[-1] == "1 problems"
    assert err == f"error: {message}\n"
    code, report = scores_json(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES)
    assert code == 1 and report["ok"] is False and [entry["message"] for entry in report["problems"]] == [message]


def test_check_names_every_problem_at_once_and_a_replaces_loop(tmp_path, capsys):
    folder = copy_study(tmp_path)
    edit_lines(folder, C.SCORES_FILE, lambda rows: [
        *rows, reading("x1", value=11), reading("x2", component="clarity", value="superb"),
        reading("x3", value=4, of=9), reading("x4", replaces="x5"), reading("x5", replaces="x4")])
    code, report = scores_json(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES)
    assert code == 1 and report["ok"] is False
    assert [entry["message"] for entry in report["problems"]] == [
        'scores.jsonl: id "x1": value: 11 is outside the scale of "depth"; give a number from 0 to 10',
        'scores.jsonl: id "x2": value: "superb" is not a level of "clarity"; use one of poor, ok, good',
        'scores.jsonl: id "x3": of: the component "depth" has a "linear" scale; drop of, or give the component the '
        'scale kind "items"',
        'scores.jsonl: id "x4": replaces: the corrections form a loop; a correction names an earlier reading only']
    assert report["problems"][-1]["code"] == "replaces_loop"
    code, out, err = scores(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES)
    assert code == 1 and out.splitlines()[-1] == "4 problems" and len(err.splitlines()) == 4


def test_check_items_counts_and_assignments(tmp_path, capsys):
    folder = small_study(tmp_path)
    rubric = tmp_path / "rubric.json"
    rubric.write_text(json.dumps({
        "schema": C.SCHEMA_RUBRIC, "id": "facts", "version": "2", "steps": ["final"],
        "components": [{"id": "facts", "scale": {"kind": "items"}},
                       {"id": "tone", "scale": {"kind": "levels", "levels": [1, 2, 3]}}]}), encoding="utf-8")
    judges = tmp_path / "judges.json"
    judges.write_text(json.dumps({"schema": C.SCHEMA_JUDGES, "anchor": ["reviewer"], "judges": [
        {"id": "reviewer", "kind": "person", "mode": "per_component", "reference": {"shown": False}},
        {"id": "judge-a", "kind": "model", "model": {"id": "model-x", "family": "alpha"}, "mode": "panel",
         "reference": {"shown": True, "author": {"kind": "model", "model": {"id": "model-y", "family": "alpha"}}}}]}),
        encoding="utf-8")

    def row(ident, **more):
        return {"id": ident, "run": "r1", "step": "final", "component": "facts", "judge": "judge-a", "call": "k1",
                "value": 1, "rubric": "facts@2", "observed_at": AT, **more}

    C.write_lines(folder / C.ITEMS_FILE, C.SCHEMA_ITEMS, [
        {"id": "i1", "input": "case-01", "component": "facts"}, {"id": "i2", "input": "case-01", "component": "facts"},
        {"id": "i3", "input": "case-02", "component": "facts"}, {"id": "i4", "input": "case-01", "component": "tone"}])
    C.write_lines(folder / C.SCORES_FILE, C.SCHEMA_SCORES, [
        row("s1", item="i1"), row("s2", item="i2", value=0),            # item lines of one call: fine
        row("s3", item="i1", call="k2"), row("s4", item="i1", call="k2"),  # one item twice in a call
        row("s5", item="i2", call="k3"), row("s6", value=3, of=4, call="k3"),  # item lines and a count line
        row("s7", item="i9", call="k4"), row("s8", item="i3", call="k5"),  # no such item; an item of another input
        row("s9", component="tone", value=2, call="k6", judge="reviewer"),
        row("s10", run="r2", value=2, of=5, call="k7", judge="reviewer", assignment="a1")])
    code, report = scores_json(capsys, "check", folder, "--rubric", rubric, "--judges", judges, "--require-assignment")
    assert code == 1
    assert [entry["message"] for entry in report["problems"]] == [
        'scores.jsonl: id "s7": item: unknown item "i9"; add it to items.jsonl (eval scores import --items)',
        'scores.jsonl: id "s8": item: the item "i3" is written for the input "case-02" and the run is on "case-01"; '
        "name an item of the run's input",
        'scores.jsonl: id "s9": the person "reviewer" gave it with no assignment; name the labels next assignment '
        "that asked for it",
        'items.jsonl: id "i4": component: the component "tone" has a "levels" scale; name a component with the scale '
        'kind "items"',
        'scores.jsonl: call "k2": two readings of one call for the item "i1"; give a repeat its own call, or name '
        "the reading it corrects in replaces",
        'scores.jsonl: call "k3": the call has item lines and a count line for the component "facts" ("s5" and '
        '"s6"); write item lines or one count line, never both']
    design = {(row["term"], row["judge"]): row for row in report["design"]}
    assert design[("reference_family", "judge-a")]["state"] == "confounded"
    halo = design[("halo", "judge-a")]  # a panel judge alone: its repeated calls on one target are what there is
    assert (halo["count"], halo["met"], halo["modes"]) == (1, True, ["panel"])
    assert report["warnings"] == []
    code, _, err = scores(capsys, "check", folder, "--rubric", rubric, "--require-assignment")
    assert (code, err) == (1, "error: eval scores check: --require-assignment: the judges file says who is a person; "
                              "give --judges with it\n")


def test_check_reports_mixed_arms_and_a_shared_call_and_a_pair_with_no_overall(tmp_path, capsys):
    folder = copy_study(tmp_path)

    def other_model(rows):
        rows[0]["steps"][1]["model"] = {"id": "model-mid", "family": "beta"}
        return rows

    edit_lines(folder, C.RUNS_FILE, other_model)
    edit_lines(folder, C.SCORES_FILE, lambda rows: [*rows, reading("x1", call="c0001")])
    code, report = scores_json(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES)
    assert code == 0 and report["ok"] is True
    assert [(entry["arm"], entry["runs"], [s["runs"] for s in entry["settings"]]) for entry in report["mixed_arm"]] == [
        ("baseline", 8, [1, 7])]
    assert report["warnings"] == []  # one judge, one target, another component: one call
    edit_lines(folder, C.SCORES_FILE, lambda rows: [*rows, reading("x2", call="c0001", run="strong-01",
                                                            component="clarity", value="ok")])
    code, out, err = scores(capsys, "check", folder, "--rubric", RUBRIC)
    assert code == 0 and "mixed_arm: baseline: 2 settings over 8 runs" in out
    assert ('warning: scores.jsonl: call "c0001": shared by 2 pairs of judge and target; a call is one judge '
            "scoring one target") in out
    plain = tmp_path / "plain.json"
    rubric = json.loads(Path(RUBRIC).read_text(encoding="utf-8"))
    rubric["components"] = [comp for comp in rubric["components"] if comp["id"] != "overall"]
    plain.write_text(json.dumps(rubric), encoding="utf-8")
    edit_lines(folder, C.PAIRS_FILE, lambda rows: [{k: v for k, v in pair("y1").items() if k != "component"}])
    edit_lines(folder, C.SCORES_FILE, lambda rows: [row for row in rows if row["component"] != "overall"])
    code, _, err = scores(capsys, "check", folder, "--rubric", plain)
    assert code == 1 and err == ('error: pairs.jsonl: id "y1": component: the pair names no component and step '
                                 '"final" has no overall component; name the component compared\n')


def test_check_needs_a_study(tmp_path, capsys):
    code, out, err = scores(capsys, "check", tmp_path / "none", "--rubric", RUBRIC)
    assert code == 1 and out == "" and err.startswith("error: ")


# ---------------------------------------------------------------------------------------------------
# add

def test_add_appends_a_reading_with_its_defaults(tmp_path, capsys):
    folder = copy_study(tmp_path)
    code, out, err = scores(capsys, "add", folder, "--rubric", RUBRIC, "--run", "baseline-01", "--component", "clarity",
                            "--judge", "reviewer", "--value", "good")
    assert (code, out, err) == (0, "added reading s0306 to scores.jsonl\n", "")
    row = C.read_study(folder).scores[-1]
    assert (row["id"], row["call"], row["value"], row["step"], row["rubric"]) == ("s0306", "c-s0306", "good", "final",
                                                                                  "brief@1")
    assert C.validate_score(dict(row), "scores.jsonl", "") == row  # the time it wrote is a time with an offset
    code, report = scores_json(capsys, "add", folder, "--rubric", RUBRIC, "--run", "baseline-01", "--step", "final",
                               "--component", "depth", "--judge", "reviewer", "--value", "7.5", "--call", "k1",
                               "--observed-at", AT, "--replaces", "s0306", "--assignment", "a1", "--id", "mine")
    assert code == 0 and report["schema"] == "loopmath.eval-scores-add/1" and report["added"] == "reading"
    assert report["row"] == {"schema": C.SCHEMA_SCORES, "id": "mine", "run": "baseline-01", "step": "final",
                             "component": "depth", "judge": "reviewer", "call": "k1", "value": 7.5,
                             "rubric": "brief@1", "observed_at": AT, "replaces": "s0306", "assignment": "a1"}
    assert len(C.read_study(folder).scores) == 307
    assert scores(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES)[0] == 0


def test_add_appends_a_pair(tmp_path, capsys):
    folder = copy_study(tmp_path)
    code, out, _ = scores(capsys, "add", folder, "--rubric", RUBRIC, "--pair", "baseline-01", "strong-01", "--judge",
                          "reviewer", "--verdict", "b", "--shown-first", "b")
    assert (code, out) == (0, "added pair p019 to pairs.jsonl\n")
    row = C.read_study(folder).pairs[-1]
    assert (row["id"], row["a"], row["b"], row["verdict"], row["shown_first"], row["component"], row["call"]) == (
        "p019", "baseline-01", "strong-01", "b", "b", None, "c-p019")
    code, report = scores_json(capsys, "add", folder, "--rubric", RUBRIC, "--pair", "baseline-01", "strong-01",
                               "--judge", "reviewer", "--verdict", "tie", "--component", "depth", "--replaces", "p019")
    assert code == 0 and report["row"]["component"] == "depth" and report["row"]["replaces"] == "p019"
    assert "shown_first" not in report["row"]
    assert scores(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES)[0] == 0


READING = ("--run", "baseline-01", "--component", "depth", "--judge", "reviewer", "--value", "5")
PAIR = ("--pair", "baseline-01", "strong-01", "--judge", "reviewer", "--verdict", "a")
ADDS = [
    (("--judge", "reviewer", "--value", "5"),
     "eval scores add: give --run for a reading or --pair for a pair, one of the two; name the run scored, or the "
     "two runs compared"),
    ((*READING, "--pair", "baseline-01", "strong-01"),
     "eval scores add: give --run for a reading or --pair for a pair, one of the two; name the run scored, or the "
     "two runs compared"),
    ((*READING, "--verdict", "a"), "eval scores add: --verdict: not an option of a reading; remove it"),
    ((*READING, "--step", "draft"),
     'eval scores add: --step: the component "depth" is not scored at step "draft"; it applies to final'),
    ((*PAIR, "--value", "5"), "eval scores add: --value: not an option of a pair; remove it"),
    (("--run", "nobody", "--component", "depth", "--judge", "reviewer", "--value", "5"),
     'eval scores add: --run: no run "nobody" in runs.jsonl; name a run of the study, or import it first'),
    (("--run", "baseline-01", "--judge", "reviewer", "--value", "5"),
     "eval scores add: --component: missing; name the rubric component"),
    (("--run", "baseline-01", "--component", "speed", "--judge", "reviewer", "--value", "5"),
     'eval scores add: --component: unknown component "speed"; use one of numbers_right, clarity, depth, overall'),
    (("--run", "baseline-01", "--component", "depth", "--judge", "reviewer"),
     "eval scores add: --value: missing; give the raw score on the component's scale"),
    ((*READING, "--id", "s0001"),
     'eval scores add: --id: the reading id "s0001" is already in scores.jsonl; give a new id, and name the one it '
     "corrects in --replaces"),
    ((*READING, "--replaces", "s9999"),
     'eval scores add: --replaces: "s9999" names no reading of scores.jsonl; name the id of the reading it corrects'),
    (("--run", "baseline-01", "--component", "numbers_right", "--judge", "judge-strong", "--value", "1", "--call",
      "c0001"),
     'eval scores add: --call: the call "c0001" already has the reading "s0001" for the component "numbers_right"; '
     "give a repeat its own call, or name the reading it corrects in --replaces"),
    (("--pair", "baseline-01", "strong-02", "--judge", "reviewer", "--verdict", "a"),
     'eval scores add: --pair: the runs "baseline-01" and "strong-02" have different inputs; compare two runs of one '
     "input"),
    (("--pair", "baseline-01", "strong-01", "--judge", "reviewer"),
     "eval scores add: --verdict: missing; give a, b or tie"),
    ((*PAIR, "--replaces", "s0001"),
     'eval scores add: --replaces: "s0001" names no pair of pairs.jsonl; name the id of the pair it corrects'),
]


@pytest.mark.parametrize("argv, message", ADDS, ids=[
    "neither", "both", "verdict-on-reading", "step", "value-on-pair", "no-run", "no-component", "unknown-component",
    "no-value",
    "id-taken", "replaces-unknown", "call-twice", "pair-inputs", "no-verdict", "pair-replaces"])
def test_add_refusals_write_nothing(tmp_path, capsys, argv, message):
    folder = copy_study(tmp_path)
    before = snapshot(folder)
    assert scores(capsys, "add", folder, "--rubric", RUBRIC, *argv) == (1, "", f"error: {message}\n")
    assert snapshot(folder) == before


def test_add_checks_the_value_against_the_scale_and_takes_a_correction_in_the_same_call(tmp_path, capsys):
    folder = copy_study(tmp_path)
    before = snapshot(folder)
    for argv in ((*READING[:-1], "11"), ("--run", "baseline-01", "--component", "clarity", "--judge", "reviewer",
                                         "--value", "superb"), (*READING, "--of", "9"),
                 (*READING, "--observed-at", "yesterday"), (*READING, "--judge", "the reviewer")):
        code, out, err = scores(capsys, "add", folder, "--rubric", RUBRIC, *argv)
        assert code == 1 and out == "" and err.startswith("error: scores.jsonl: ")
    assert snapshot(folder) == before
    code, out, _ = scores(capsys, "add", folder, "--rubric", RUBRIC, "--run", "baseline-01", "--component",
                          "numbers_right", "--judge", "judge-strong", "--value", "1", "--call", "c0001", "--replaces",
                          "s0001")
    assert code == 0 and out == "added reading s0306 to scores.jsonl\n"
    assert scores(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES)[0] == 0


# ---------------------------------------------------------------------------------------------------
# export

def test_export_prints_one_line_per_run_and_changes_no_file(tmp_path, capsys):
    folder = copy_study(tmp_path)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    code, out, err = scores(capsys, "export", folder, "--run", "strong-01", "--run", "baseline-01")
    assert code == 0 and err == ""
    entries = [json.loads(line) for line in out.splitlines()]
    assert [entry["run"] for entry in entries] == ["strong-01", "baseline-01"]
    assert set(entries[0]) == {"run", "signals", "preferences", "slate"}
    study = C.read_study(folder)
    assert len(entries[1]["signals"]) == sum(1 for row in study.scores if row["run"] == "baseline-01")
    assert {s["id"] for s in entries[1]["signals"]} >= {"s0021", "s0305"}  # the corrected reading and its correction
    assert {s["source"]["kind"] for s in entries[1]["signals"] if s["source"]["ref"] == "reviewer"} == {"user"}
    code, report = scores_json(capsys, "export", folder)
    assert code == 0 and report["schema"] == "loopmath.eval-scores-export/1"
    assert len(report["runs"]) == 16 and report["not_used"] == []
    assert sum(len(entry["signals"]) for entry in report["runs"]) == 305
    assert len({p["id"] for entry in report["runs"] for p in entry["preferences"]}) == 18
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert scores(capsys, "export", folder, "--run", "nobody") == (
        1, "", 'error: eval scores export: --run: no run "nobody" in runs.jsonl; name a run of the study\n')


def test_export_leaves_out_runs_with_nothing_to_say_unless_asked(tmp_path, capsys):
    folder = small_study(tmp_path)
    assert scores(capsys, "export", folder) == (0, "", "")
    code, out, _ = scores(capsys, "export", folder, "--run", "r1")
    assert code == 0 and json.loads(out) == {"run": "r1", "signals": [], "preferences": []}
    edit = folder / C.SCORES_FILE
    edit.write_text(json.dumps({"schema": C.SCHEMA_SCORES, "id": "s1", "run": "gone", "step": "final",
                                "component": "depth", "judge": "judge-a", "call": "k1", "value": 5,
                                "rubric": "brief@1", "observed_at": AT}) + "\n", encoding="utf-8")
    code, out, err = scores(capsys, "export", folder)
    assert (code, out, err) == (0, "", "not used: reading 's1': no run \"gone\" in the study\n")


def test_add_item_lines_and_a_count_on_an_items_component(tmp_path, capsys):
    folder = small_study(tmp_path)
    rubric = tmp_path / "rubric.json"
    rubric.write_text(json.dumps({"schema": C.SCHEMA_RUBRIC, "id": "facts", "version": "2", "steps": ["final"],
                                  "components": [{"id": "facts", "scale": {"kind": "items"}}]}), encoding="utf-8")
    C.write_lines(folder / C.ITEMS_FILE, C.SCHEMA_ITEMS, [
        {"id": "i1", "input": "case-01", "component": "facts"}, {"id": "i2", "input": "case-01", "component": "facts"},
        {"id": "i3", "input": "case-02", "component": "facts"}])
    base = ("add", folder, "--rubric", rubric, "--component", "facts", "--judge", "judge-a")
    assert scores(capsys, *base, "--run", "r1", "--value", "1", "--item", "i1", "--call", "k1")[0] == 0
    assert scores(capsys, *base, "--run", "r1", "--value", "0", "--item", "i2", "--call", "k1")[0] == 0
    assert scores(capsys, *base, "--run", "r2", "--value", "7", "--of", "9")[0] == 0
    rows = C.read_study(folder).scores
    assert [(r["id"], r["call"], r["value"], r.get("item"), r.get("of")) for r in rows] == [
        ("s0001", "k1", 1, "i1", None), ("s0002", "k1", 0, "i2", None), ("s0003", "c-s0003", 7, None, 9)]
    assert scores(capsys, *base, "--run", "r1", "--value", "1", "--item", "i1", "--call", "k1") == (
        1, "", 'error: eval scores add: --call: the call "k1" already has the reading "s0001" for the component '
               '"facts"; give a repeat its own call, or name the reading it corrects in --replaces\n')
    assert scores(capsys, *base, "--run", "r1", "--value", "4", "--of", "5", "--call", "k1")[0] == 1
    assert scores(capsys, *base, "--run", "r1", "--value", "1", "--item", "i3") == (
        1, "", 'error: eval scores add: --item: no item "i3" of the component "facts" for the input "case-01" in '
               "items.jsonl; name an item of the run's input, or import the items first\n")
    assert len(C.read_study(folder).scores) == 3
    assert scores(capsys, "check", folder, "--rubric", rubric)[0] == 0


# ---------------------------------------------------------------------------------------------------
# One call, one verdict: pairs

def test_check_refuses_two_pairs_of_one_call(tmp_path, capsys):
    folder = copy_study(tmp_path)
    base = ("check", folder, "--rubric", RUBRIC, "--judges", JUDGES)
    # the same runs, judge, component and call, with opposite verdicts: one call cannot say both
    edit_lines(folder, C.PAIRS_FILE, lambda rows: [*rows, pair("y1", call="q1", verdict="a"),
                                                   pair("y2", call="q1", verdict="b")])
    code, report = scores_json(capsys, *base)
    assert code == 1 and report["ok"] is False
    assert [entry["message"] for entry in report["problems"]] == [
        'pairs.jsonl: call "q1": two pairs of one call for the component "overall" ("y1" and "y2"); give a repeat '
        "its own call, or name the pair it corrects in replaces"]
    assert report["problems"][0]["code"] == "call_twice"
    # a pair with no component is on the overall component: it clashes with one that names it
    edit_lines(folder, C.PAIRS_FILE, lambda rows: [
        *rows[:-1], {k: v for k, v in pair("y2", call="q1", verdict="b").items() if k != "component"}])
    code, _, err = scores(capsys, *base)
    assert code == 1 and err == ('error: pairs.jsonl: call "q1": two pairs of one call for the component "overall" '
                                 '("y1" and "y2"); give a repeat its own call, or name the pair it corrects in '
                                 "replaces\n")
    # as a correction it is one verdict: 18 pairs of the study plus this one are read, and y1 is not
    edit_lines(folder, C.PAIRS_FILE, lambda rows: [*rows[:-1], pair("y2", call="q1", verdict="b", replaces="y1")])
    code, report = scores_json(capsys, *base)
    assert code == 0 and report["ok"] is True
    assert (report["counts"]["pairs"], report["counts"]["pairs_read"]) == (20, 19)
    assert {"table": "pairs", "id": "y1", "reason": "replaced", "by": "y2"} in report["not_read"]
    # as a repeat with its own call both are read, and so is a second component compared in the same call
    edit_lines(folder, C.PAIRS_FILE, lambda rows: [*rows[:-1], pair("y2", call="q2", verdict="b"),
                                                   pair("y3", call="q1", component="depth")])
    code, report = scores_json(capsys, *base)
    assert code == 0 and (report["counts"]["pairs"], report["counts"]["pairs_read"]) == (21, 21)


def test_add_refuses_a_second_pair_in_a_call_and_takes_it_as_a_correction(tmp_path, capsys):
    folder = copy_study(tmp_path)
    first = ("add", folder, "--rubric", RUBRIC, "--pair", "baseline-01", "strong-01", "--judge", "judge-strong",
             "--call", "q1")
    assert scores(capsys, *first, "--verdict", "a")[0] == 0
    before = snapshot(folder)
    assert scores(capsys, *first, "--verdict", "b") == (
        1, "", 'error: eval scores add: --call: the call "q1" already has the pair "p019" for the component '
               '"overall"; give a repeat its own call, or name the pair it corrects in --replaces\n')
    assert scores(capsys, *first, "--verdict", "b", "--component", "overall")[0] == 1
    assert snapshot(folder) == before and len(C.read_study(folder).pairs) == 19
    assert scores(capsys, *first, "--verdict", "b", "--replaces", "p019")[0] == 0
    assert scores(capsys, *first, "--verdict", "b", "--component", "depth")[0] == 0  # another component, same call
    assert scores(capsys, "add", folder, "--rubric", RUBRIC, "--pair", "baseline-01", "strong-01", "--judge",
                  "judge-strong", "--verdict", "a")[0] == 0  # a repeat, with a call of its own
    assert len(C.read_study(folder).pairs) == 22
    assert scores(capsys, "check", folder, "--rubric", RUBRIC, "--judges", JUDGES)[0] == 0


def test_import_refuses_rows_that_share_a_call_and_writes_nothing(tmp_path, capsys):
    folder = small_study(tmp_path)
    header = "id,a,b,component,judge,call,shown_first,verdict,observed_at"
    twice = write_csv(tmp_path / "twice.csv", header, f"p1,r1,r2,overall,reviewer,q1,a,a,{AT}",
                      f"p2,r1,r2,overall,reviewer,q1,a,b,{AT}", f"p3,r1,r2,depth,reviewer,q1,a,b,{AT}")
    code, out, err = scores(capsys, "import", "--pairs-csv", twice, "--rubric", RUBRIC, "--out", folder)
    assert code == 1 and not (folder / C.PAIRS_FILE).exists()
    assert err == ('error: twice.csv line 3: the pair "p2" shares the call "q1" with the pair "p1" for the component '
                   '"overall"; give a repeat its own call, or name the pair it corrects in replaces\n')
    assert "pairs: 2 new, 0 unchanged, 1 refused" in out
    # the first two rows alone, the second as a correction of the first: both are kept, one is read
    fine = write_csv(tmp_path / "fine.csv", header + ",replaces", f"p1,r1,r2,overall,reviewer,q1,a,a,{AT},",
                     f"p2,r1,r2,overall,reviewer,q1,a,b,{AT},p1")
    code, report = scores_json(capsys, "import", "--pairs-csv", fine, "--rubric", RUBRIC, "--out", folder)
    assert code == 0 and report["pairs"] == {"new": 2, "unchanged": 0, "refused": 0}
    # a later file that reuses the call of a pair already in the study
    later = write_csv(tmp_path / "later.csv", header, f"p9,r1,r2,,reviewer,q1,b,a,{AT}")
    before = snapshot(folder)
    code, report = scores_json(capsys, "import", "--pairs-csv", later, "--rubric", RUBRIC, "--out", folder)
    assert code == 1 and report["pairs"] == {"new": 0, "unchanged": 0, "refused": 1}
    assert report["refused"] == [{"table": "pairs", "id": "p9", "source": "later.csv line 2", "reason": (
        'the pair "p9" shares the call "q1" with the pair "p2" for the component "overall"; give a repeat its own '
        "call, or name the pair it corrects in replaces")}]
    assert snapshot(folder) == before
    # readings follow the same rule on the way in
    readings = write_csv(tmp_path / "readings.csv", "id,run,component,judge,call,value,observed_at",
                         f"s1,r1,depth,reviewer,k1,4,{AT}", f"s2,r1,depth,reviewer,k1,9,{AT}",
                         f"s3,r1,clarity,reviewer,k1,ok,{AT}")
    code, _, err = scores(capsys, "import", "--csv", readings, "--rubric", RUBRIC, "--out", folder)
    assert code == 1 and err == ('error: readings.csv line 3: the reading "s2" shares the call "k1" with the reading '
                                 '"s1" for the component "depth"; give a repeat its own call, or name the reading '
                                 "it corrects in replaces\n")
    assert not (folder / C.SCORES_FILE).exists() and snapshot(folder) == before


def test_import_takes_documents_that_share_signal_ids_and_a_second_import_changes_nothing(tmp_path, capsys):
    def score(ident, value):
        return {"id": ident, "kind": "score", "name": "depth", "value": value, "tier": "reported",
                "observed_at": AT, "source": {"kind": "judge", "ref": "reviewer"}}

    first = {"run": "r1", "input": "case-01", "arm": "baseline", "steps": [
        {"step": "final", "model": {"id": "model-small", "family": "beta"}, "effort": "low"}]}
    (tmp_path / "in").mkdir()
    docs = []
    for run, value in ((first, 4), ({**first, "run": "r2", "arm": "strong"}, 9)):
        docs.append(tmp_path / "in" / f"{run['run']}.ocp.json")
        docs[-1].write_text(json.dumps(document(run, signals=[score("s1", value)])), encoding="utf-8")
    folder = tmp_path / "study"
    base = ("import", "--ocp", *docs, "--rubric", RUBRIC, "--out", folder)
    code, report = scores_json(capsys, *base)
    assert code == 0 and report["readings"] == {"new": 2, "unchanged": 0, "refused": 0} and report["refused"] == []
    assert [(row["id"], row["run"], row["value"]) for row in C.read_study(folder).scores] == [
        ("r1.s1", "r1", 4), ("r2.s1", "r2", 9)]
    before = snapshot(folder)
    code, report = scores_json(capsys, *base)
    assert code == 0 and report["readings"] == {"new": 0, "unchanged": 2, "refused": 0} and snapshot(folder) == before
    assert scores(capsys, "check", folder, "--rubric", RUBRIC)[0] == 0
    code, out, _ = scores(capsys, "export", folder)
    assert code == 0 and [[s["id"] for s in json.loads(line)["signals"]] for line in out.splitlines()] == [
        ["s1"], ["s1"]]


# ---------------------------------------------------------------------------------------------------
# Ids of records made by other tools (review 2 item 10, and preferences)

def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def other_score(ident: str, value, minute: int = 0, judge: str = "reviewer") -> dict:
    return {"id": ident, "kind": "score", "name": "depth", "value": value, "tier": "reported",
            "observed_at": f"2026-02-03T10:{minute:02d}:00-08:00", "source": {"kind": "judge", "ref": judge}}


def run_of(name: str, input_id: str = "case-01", arm: str = "baseline") -> dict:
    return {"run": name, "input": input_id, "arm": arm, "steps": [
        {"step": "final", "model": {"id": f"model-{arm}", "family": arm}, "effort": "low"}]}


def write_doc(path: Path, run: dict, **more) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document(run, **more)), encoding="utf-8")
    return path


def test_import_takes_runs_a_dot_b_and_a_whose_signal_ids_gave_one_id_in_0_4_4(tmp_path, capsys):
    docs = [write_doc(tmp_path / "in" / "ab.json", run_of("a.b"), signals=[other_score("c", 4, 1)]),
            write_doc(tmp_path / "in" / "a.json", run_of("a", arm="strong"), signals=[other_score("b.c", 9, 1)])]
    folder = tmp_path / "study"
    base = ("import", "--ocp", *docs, "--rubric", RUBRIC, "--out", folder)
    code, report = scores_json(capsys, *base)
    assert code == 0 and report["readings"] == {"new": 2, "unchanged": 0, "refused": 0} and report["refused"] == []
    assert [(row["id"], row["run"], row["value"]) for row in C.read_study(folder).scores] == [
        ("a%2Eb.c", "a.b", 4), ("a.b.c", "a", 9)]
    assert scores(capsys, "check", folder, "--rubric", RUBRIC)[0] == 0
    before = snapshot(folder)
    assert scores_json(capsys, *base)[1]["readings"] == {"new": 0, "unchanged": 2, "refused": 0}
    assert snapshot(folder) == before
    code, out, _ = scores(capsys, "export", folder)
    assert code == 0 and [[(s["id"], s["ext"][A.NS]["id"]) for s in json.loads(line)["signals"]]
                          for line in out.splitlines()] == [[("c", "a%2Eb.c")], [("b.c", "a.b.c")]]


def test_import_takes_preferences_of_another_tool_in_two_slates_and_refuses_a_taken_id(tmp_path, capsys):
    def pref(slate: str, winner: str) -> dict:
        return {"id": "f1", "slate": slate, "winner": winner, "observed_at": AT, "judge": {"kind": "model"}}

    runs = [run_of("r1"), run_of("r2", arm="strong"), run_of("r3", "case-02"), run_of("r4", "case-02", "strong")]
    slates = [{"id": "sl-1", "members": ["r1", "r2"]}, {"id": "sl-2", "members": ["r3", "r4"]}]
    docs = [write_doc(tmp_path / "in" / f"{run['run']}.json", run, slate=slate, preferences=[pref(slate["id"], win)])
            for run, slate, win in zip(runs, (slates[0], slates[0], slates[1], slates[1]), ("r2", "r2", "r3", "r3"))]
    folder = tmp_path / "study"
    code, report = scores_json(capsys, "import", "--ocp", *docs, "--rubric", RUBRIC, "--out", folder)
    assert code == 0 and report["pairs"] == {"new": 2, "unchanged": 0, "refused": 0}
    assert [(row["id"], row["a"], row["b"], row["verdict"]) for row in C.read_study(folder).pairs] == [
        ("sl-1.f1", "r1", "r2", "b"), ("sl-2.f1", "r3", "r4", "a")]
    assert scores(capsys, "check", folder, "--rubric", RUBRIC)[0] == 0
    # a later file with the same preference id in the same slate and another verdict is refused by id
    later = write_doc(tmp_path / "later" / "r1.json", runs[0], slate=slates[0], preferences=[pref("sl-1", "r1")])
    before = snapshot(folder)
    code, out, err = scores(capsys, "import", "--ocp", later, "--rubric", RUBRIC, "--out", folder)
    assert code == 1 and snapshot(folder) == before
    assert err == ('error: r1.json: the pair id "sl-1.f1" is already in the study with other content; give a changed '
                   "pair a new id, and name the one it corrects in replaces\n")


def test_import_refuses_a_record_that_a_0_4_4_study_holds_under_its_old_id(tmp_path, capsys):
    runs = [run_of("team/r 3"), run_of("r2", arm="strong")]
    slate = {"id": "sl-1", "members": ["team/r 3", "r2"]}
    pref = {"id": "f1", "slate": "sl-1", "winner": "r2", "observed_at": AT, "judge": {"kind": "model"}}
    docs = [write_doc(tmp_path / "in" / "t.json", runs[0], slate=slate, preferences=[pref],
                      signals=[other_score("s1", 4)]),
            write_doc(tmp_path / "in" / "r2.json", runs[1], slate=slate, preferences=[pref],
                      signals=[other_score("s1", 7)])]
    folder = tmp_path / "study"
    base = ("import", "--ocp", *docs, "--rubric", RUBRIC, "--out", folder)
    assert scores_json(capsys, *base)[0] == 0
    # the study as 0.4.4 wrote it: "team/r 3" and "s1" gave "team_r_3.s1", and a preference kept its own id
    old_ids = {"team%2Fr%203.s1": "team_r_3.s1", "r2.s1": "r2.s1"}
    edit_lines(folder, C.SCORES_FILE, lambda rows: [{**row, "id": old_ids[row["id"]], "call": old_ids[row["id"]]}
                                                    for row in rows])
    edit_lines(folder, C.PAIRS_FILE, lambda rows: [
        {**row, "id": "f1", "call": "f1", "ext": {A.KEPT: {"slate": "sl-1"}}} for row in rows])
    assert scores(capsys, "check", folder, "--rubric", RUBRIC)[0] == 0
    before = snapshot(folder)
    code, report = scores_json(capsys, *base)
    assert code == 1 and snapshot(folder) == before
    assert report["readings"] == {"new": 0, "unchanged": 1, "refused": 1}  # r2.s1 is the id 0.4.4 gave too
    assert report["refused"] == [
        {"table": "readings", "id": "team%2Fr%203.s1", "source": "t.json", "reason": (
            'the signal "s1" of the run "team/r 3" is already in the study as the reading "team_r_3.s1", and would '
            'now be the reading "team%2Fr%203.s1" (loopmath 0.4.4 named it otherwise); import the documents into a '
            "new study folder")},
        {"table": "pairs", "id": "sl-1.f1", "source": "t.json", "reason": (
            'the preference "f1" of the slate "sl-1" is already in the study as the pair "f1", and would now be the '
            'pair "sl-1.f1" (loopmath 0.4.4 named it otherwise); import the documents into a new study folder')}]
    # into a new folder the same documents are taken
    assert scores_json(capsys, "import", "--ocp", *docs, "--rubric", RUBRIC, "--out", tmp_path / "new")[0] == 0


LEGACY = Path(__file__).resolve().parent / "data" / "ocp-0-4-4"  # written by loopmath 0.4.4 itself
NEW_FOLDER = "(loopmath 0.4.4 named it otherwise); import the documents into a new study folder"
LEGACY_REFUSED = [
    {"table": "readings", "id": "a%2Eb.s1", "source": "ab.json", "reason": (
        'the signal "s1" of the run "a.b" is already in the study as the reading "a.b.s1", and would now be the '
        f'reading "a%2Eb.s1" {NEW_FOLDER}')},
    {"table": "pairs", "id": "slate-1.f1", "source": "ab.json", "reason": (
        'the preference "f1" of the slate "slate-1" is already in the study as the pair "f1", and would now be the '
        f'pair "slate-1.f1" {NEW_FOLDER}')},
    {"table": "pairs", "id": "eval%2Ecase-02.g1", "source": "r3.json", "reason": (
        'the preference "g1" of the slate "eval.case-02" is already in the study as the pair "g1", and would now '
        f'be the pair "eval%2Ecase-02.g1" {NEW_FOLDER}')}]


def legacy_documents(tmp_path: Path, change=None) -> list[Path]:
    """The documents the 0.4.4 study was made from, with `change(file name, run)` applied, in a new folder."""
    folder = tmp_path / "in"
    folder.mkdir()
    paths = []
    for path in sorted((LEGACY / "in").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if change is not None:
            change(path.name, doc["run"])
        paths.append(folder / path.name)
        paths[-1].write_text(json.dumps(doc), encoding="utf-8")
    return paths


def set_signal(**more):
    def change(name: str, run: dict) -> None:
        if name == "ab.json":
            run["signals"][0].update(more)
    return change


def set_preference(ident: str, **more):
    def change(name: str, run: dict) -> None:
        for preference in run.get("preferences", []):
            if preference["id"] == ident:
                preference.update(more)  # in every member's copy
    return change


LATER = "2026-02-03T11:00:00-08:00"
OTHER_EXT = {"org.example": {"batch": 2}}
# per change: the source as it is now, and as a new study reads it: (a.b's value, its time, f1's verdict,
# f1's judge, g1's verdict); the 0.4.4 study holds (4, 10:01, "a", "judge-1", "b")
CHANGES = {
    "unchanged": (None, (4, "2026-02-03T10:01:00-08:00", "a", "judge-1", "b")),
    "value": (set_signal(value=8), (8, "2026-02-03T10:01:00-08:00", "a", "judge-1", "b")),
    "signal metadata": (set_signal(observed_at=LATER, ext=OTHER_EXT), (4, LATER, "a", "judge-1", "b")),
    "verdict": (set_preference("f1", winner="r2"), (4, "2026-02-03T10:01:00-08:00", "b", "judge-1", "b")),
    "preference metadata": (set_preference("f1", judge={"kind": "model", "model": "judge-2"}, ext=OTHER_EXT),
                            (4, "2026-02-03T10:01:00-08:00", "a", "judge-2", "b")),
    "verdict on the default slate": (set_preference("g1", winner="r3"),
                                     (4, "2026-02-03T10:01:00-08:00", "a", "judge-1", "a")),
}


def test_the_0_4_4_study_holds_the_ids_0_4_4_gave(tmp_path, capsys):
    folder = Path(shutil.copytree(LEGACY / "study", tmp_path / "study"))
    study = C.read_study(folder)
    assert [(row["id"], row["call"], row["run"], row["value"], row["ext"]) for row in study.scores] == [
        ("a.b.s1", "a.b.s1", "a.b", 4, {A.KEPT: {"id": "s1"}}), ("r2.s1", "r2.s1", "r2", 7, {A.KEPT: {"id": "s1"}})]
    assert [(row["id"], row["call"], row["a"], row["b"], row["verdict"], row["ext"][A.KEPT].get("slate"))
            for row in study.pairs] == [("f1", "f1", "a.b", "r2", "a", "slate-1"), ("g1", "g1", "r3", "r4", "b", None)]
    assert scores(capsys, "check", folder, "--rubric", RUBRIC)[0] == 0


@pytest.mark.parametrize("change", list(CHANGES))
def test_import_refuses_a_record_of_a_0_4_4_study_under_its_old_id_whatever_its_content(tmp_path, capsys, change):
    edit, now = CHANGES[change]
    folder = Path(shutil.copytree(LEGACY / "study", tmp_path / "study"))
    docs = legacy_documents(tmp_path, edit)
    before = snapshot(folder)
    code, report = scores_json(capsys, "import", "--ocp", *docs, "--rubric", RUBRIC, "--out", folder)
    assert code == 1 and snapshot(folder) == before and report["written"] == []
    assert report["runs"] == {"new": 0, "unchanged": 4, "refused": 0}
    assert report["readings"] == {"new": 0, "unchanged": 1, "refused": 1}  # 0.4.4 named r2's signal "r2.s1" too
    assert report["pairs"] == {"new": 0, "unchanged": 0, "refused": 2}
    assert report["refused"] == LEGACY_REFUSED
    # a new folder takes the same documents, as they are now, under the ids 0.4.5 gives
    code, report = scores_json(capsys, "import", "--ocp", *docs, "--rubric", RUBRIC, "--out", tmp_path / "new")
    assert code == 0 and report["refused"] == []
    new = C.read_study(tmp_path / "new")
    assert [row["id"] for row in new.scores] == ["a%2Eb.s1", "r2.s1"]
    assert [row["id"] for row in new.pairs] == ["slate-1.f1", "eval%2Ecase-02.g1"]
    assert (new.scores[0]["value"], new.scores[0]["observed_at"], new.pairs[0]["verdict"], new.pairs[0]["judge"],
            new.pairs[1]["verdict"]) == now


def test_a_run_on_two_slates_goes_out_to_valid_documents_whole_and_judge_by_judge(tmp_path, capsys):
    from loopmath.ocp.conformance import validate_many

    def f1(slate: str, other: str, winner: str, judge: str) -> dict:
        return {"id": "f1", "slate": slate, "winner": winner, "members": ["r1", other], "observed_at": AT,
                "judge": {"kind": "model", "model": judge}}

    paths = [write_doc(tmp_path / "in" / "r1.json", run_of("r1", "input-1")),
             write_doc(tmp_path / "in" / "r2.json", run_of("r2", "input-1", "strong"),
                       preferences=[f1("slate-1", "r2", "r1", "judge-1")]),
             write_doc(tmp_path / "in" / "r3.json", run_of("r3", "input-1", "strong"),
                       preferences=[f1("slate-2", "r3", "r3", "judge-2")])]
    folder = tmp_path / "study"
    code, report = scores_json(capsys, "import", "--ocp", *paths, "--rubric", RUBRIC, "--out", folder)
    assert code == 0 and report["pairs"] == {"new": 2, "unchanged": 0, "refused": 0}
    study = C.read_study(folder)
    assert [(row["id"], row["a"], row["b"], row["judge"], row["verdict"]) for row in study.pairs] == [
        ("slate-1.f1", "r1", "r2", "judge-1", "a"), ("slate-2.f1", "r1", "r3", "judge-2", "b")]

    def export(*more) -> list[dict]:
        code, report = scores_json(capsys, "export", folder, *more)
        assert code == 0 and report["warnings"] == [] and report["not_used"] == []
        return report["runs"]

    one, two = {"id": "slate-1", "members": ["r1", "r2"]}, {"id": "slate-2", "members": ["r1", "r3"]}
    whole = {entry["run"]: entry for entry in export()}
    assert "slate" not in whole["r1"] and whole["r1"]["slates"] == [one, two]  # r1 is on both: no slate of its own
    assert [(p["slate"], p["id"], p["members"]) for p in whole["r1"]["preferences"]] == [
        ("slate-1", "f1", ["r1", "r2"]), ("slate-2", "f1", ["r1", "r3"])]
    assert (whole["r2"]["slate"], whole["r3"]["slate"]) == (one, two) and "slates" not in whole["r2"]
    original = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in paths}
    merged = {name: A.merge(original[name], whole[name]) for name in original}
    assert "slate" not in merged["r1"]["run"] and len(merged["r1"]["run"]["preferences"]) == 2
    assert (merged["r2"]["run"]["slate"], merged["r3"]["run"]["slate"]) == (one, two)
    # one judge at a time into r1's document: each part leaves r1 without a slate, as the whole study does
    doc = original["r1"]
    for judge, slates in (("judge-1", [one]), ("judge-2", [two])):
        (part,) = export("--run", "r1", "--judges", judge)
        assert "slate" not in part and part["slates"] == slates and len(part["preferences"]) == 1
        doc = A.merge(doc, part)
    assert doc == merged["r1"]
    assert [[f"{f.code} {f.message}" for f in found if f.level == "error"]
            for found in validate_many(list(merged.values()))] == [[], [], []]
    # and back: a new study reads the same two pairs, and the study they came from finds nothing new
    back = []
    for name, out in merged.items():
        back.append(tmp_path / "back" / f"{name}.json")
        back[-1].parent.mkdir(exist_ok=True)
        back[-1].write_text(json.dumps(out), encoding="utf-8")
    code, report = scores_json(capsys, "import", "--ocp", *back, "--rubric", RUBRIC, "--out", tmp_path / "again")
    assert code == 0 and report["refused"] == [] and C.read_study(tmp_path / "again").pairs == study.pairs
    # r1's document, which names no slate, goes back into the study it came from as nothing new (the slate
    # `merge` gives r2's and r3's documents makes their runs differ, as in 0.4.4)
    code, report = scores_json(capsys, "import", "--ocp", back[0], "--rubric", RUBRIC, "--out", folder)
    assert code == 0 and report["runs"] == {"new": 0, "unchanged": 1, "refused": 0}
    assert report["pairs"] == {"new": 0, "unchanged": 2, "refused": 0}


# ---------------------------------------------------------------------------------------------------
# Provenance of import, add and export (review 2 item 11)

def test_import_reports_the_digest_of_every_file_it_read_and_its_options(tmp_path, capsys):
    docs = [write_doc(tmp_path / "in" / f"r{n}.ocp.json", run_of(f"r{n}", arm="baseline" if n % 2 else "strong"),
                      signals=[other_score("s1", n % 10)]) for n in range(1, 11)]
    folder = tmp_path / "study"
    code, report = scores_json(capsys, "import", "--ocp", *docs[:2], "--rubric", RUBRIC, "--out", folder)
    assert code == 0 and report["inputs"] == {
        "ocp.1": {"file": "r1.ocp.json", "sha256": sha(docs[0])},
        "ocp.2": {"file": "r2.ocp.json", "sha256": sha(docs[1])},
        "rubric":{"file": "rubric.json", "sha256": sha(RUBRIC)}}
    assert report["options"] == {"allow_simulated": False, "pair_component": None, "rubric": "brief@1",
                                 "file_times": {}}
    # ten documents into the study as it is now: the roles sort in file order, and the study's tables are
    # recorded as they were before the import wrote them
    tables = {name: sha(folder / name) for name in (C.RUNS_FILE, C.SCORES_FILE)}
    code, report = scores_json(capsys, "import", "--ocp", *docs, "--rubric", RUBRIC, "--out", folder)
    assert code == 0 and report["readings"] == {"new": 8, "unchanged": 2, "refused": 0}
    assert list(report["inputs"]) == [*(f"ocp.{n:02d}" for n in range(1, 11)), "rubric", "study.runs", "study.scores"]
    assert report["inputs"]["ocp.10"] == {"file": "r10.ocp.json", "sha256": sha(docs[9])}
    assert (report["inputs"]["study.runs"], report["inputs"]["study.scores"]) == (
        {"file": C.RUNS_FILE, "sha256": tables[C.RUNS_FILE]}, {"file": C.SCORES_FILE, "sha256": tables[C.SCORES_FILE]})
    assert sha(folder / C.SCORES_FILE) != tables[C.SCORES_FILE]
    # two sources that differ by one byte give two reports that differ
    edited = tmp_path / "in2" / "r1.ocp.json"
    edited.parent.mkdir()
    edited.write_bytes(docs[0].read_bytes() + b"\n")
    first = scores_json(capsys, "import", "--ocp", docs[0], "--rubric", RUBRIC, "--out", folder)[1]["inputs"]
    second = scores_json(capsys, "import", "--ocp", edited, "--rubric", RUBRIC, "--out", folder)[1]["inputs"]
    assert first["ocp.1"]["file"] == second["ocp.1"]["file"] and first["ocp.1"]["sha256"] != second["ocp.1"]["sha256"]
    # tables, a labels-free form with options
    small = small_study(tmp_path)
    readings = write_csv(tmp_path / "r.csv", "id,run,component,judge,call,value,observed_at",
                         f"s1,r1,depth,reviewer,k1,4,{AT}")
    pairs = write_csv(tmp_path / "p.csv", "id,a,b,judge,call,verdict,observed_at", f"p1,r1,r2,reviewer,q1,a,{AT}")
    code, report = scores_json(capsys, "import", "--csv", readings, "--pairs-csv", pairs, "--rubric", RUBRIC,
                               "--pair-component", "depth", "--allow-simulated", "--out", small)
    assert code == 0 and report["inputs"] == {
        "csv": {"file": "r.csv", "sha256": sha(readings)}, "pairs_csv": {"file": "p.csv", "sha256": sha(pairs)},
        "rubric": {"file": "rubric.json", "sha256": sha(RUBRIC)},
        "study.runs": {"file": C.RUNS_FILE, "sha256": sha(small / C.RUNS_FILE)}}
    assert report["options"] == {"allow_simulated": True, "pair_component": "depth", "rubric": "brief@1",
                                 "file_times": {}}  # every row gave its own time


def test_add_reports_the_files_as_it_read_them_and_its_options(tmp_path, capsys):
    folder = copy_study(tmp_path)
    before = {name: sha(folder / name) for name in (C.RUNS_FILE, C.SCORES_FILE, C.PAIRS_FILE)}
    code, report = scores_json(capsys, "add", folder, "--rubric", RUBRIC, "--run", "baseline-01", "--component",
                               "depth", "--judge", "reviewer", "--value", "5")
    assert code == 0 and report["inputs"] == {
        "pairs": {"file": C.PAIRS_FILE, "sha256": before[C.PAIRS_FILE]},
        "rubric": {"file": "rubric.json", "sha256": sha(RUBRIC)},
        "runs": {"file": C.RUNS_FILE, "sha256": before[C.RUNS_FILE]},
        "scores": {"file": C.SCORES_FILE, "sha256": before[C.SCORES_FILE]}}
    assert sha(folder / C.SCORES_FILE) != before[C.SCORES_FILE]  # the digest is of the file before the reading
    assert report["options"] == {
        "run": "baseline-01", "pair": None, "step": "final", "component": "depth", "judge": "reviewer", "value": "5",
        "item": None, "of": None, "verdict": None, "shown_first": None, "call": "c-s0306",
        "observed_at": report["row"]["observed_at"], "replaces": None, "assignment": None, "id": "s0306",
        "defaults": ["id", "call", "observed_at"]}
    code, report = scores_json(capsys, "add", folder, "--rubric", RUBRIC, "--pair", "baseline-01", "strong-01",
                               "--judge", "reviewer", "--verdict", "b", "--call", "k9", "--observed-at", AT,
                               "--id", "mine")
    assert code == 0 and report["inputs"]["pairs"]["sha256"] == before[C.PAIRS_FILE]
    assert report["inputs"]["scores"]["sha256"] == sha(folder / C.SCORES_FILE)
    assert {key: report["options"][key] for key in ("pair", "component", "call", "observed_at", "id", "defaults")} == {
        "pair": ["baseline-01", "strong-01"], "component": "overall", "call": "k9", "observed_at": AT, "id": "mine",
        "defaults": ["component"]}


# ---------------------------------------------------------------------------------------------------
# export --judges

def test_export_keeps_to_the_judges_asked_for_and_reports_its_inputs(tmp_path, capsys):
    folder = copy_study(tmp_path)
    study = C.read_study(folder)
    code, report = scores_json(capsys, "export", folder, "--judges", "reviewer")
    assert code == 0 and report["warnings"] == [] and report["not_used"] == []
    signals = [s for entry in report["runs"] for s in entry["signals"]]
    prefs = {p["id"]: p for entry in report["runs"] for p in entry["preferences"]}
    assert {s["source"]["ref"] for s in signals} == {"reviewer"} and len(signals) == 49
    assert {p["judge"]["ext"][A.NS]["id"] for p in prefs.values()} == {"reviewer"} and len(prefs) == 6
    by_reviewer = ({row["run"] for row in study.scores if row["judge"] == "reviewer"}
                   | {row[side] for row in study.pairs if row["judge"] == "reviewer" for side in ("a", "b")})
    assert [entry["run"] for entry in report["runs"]] == [row["run"] for row in study.runs if row["run"] in by_reviewer]
    assert report["options"] == {"run": None, "judges": ["reviewer"], "persons": ["reviewer"]}
    assert report["inputs"] == {name.split(".")[0]: {"file": name, "sha256": sha(folder / name)}
                                for name in ("judges.json", C.PAIRS_FILE, C.RUNS_FILE, C.SCORES_FILE)}
    # repeated, and with --run; the text form prints the same lines
    code, report = scores_json(capsys, "export", folder, "--judges", "reviewer", "--judges", "judge-cheap", "reviewer")
    assert code == 0 and report["options"]["judges"] == ["reviewer", "judge-cheap"]
    assert sum(len(entry["signals"]) for entry in report["runs"]) == 49 + 128
    code, out, err = scores(capsys, "export", folder, "--judges", "judge-strong", "--run", "strong-01")
    assert code == 0 and err == "" and len(out.splitlines()) == 1
    entry = json.loads(out)
    assert {s["source"]["ref"] for s in entry["signals"]} == {"judge-strong"} and len(entry["signals"]) == sum(
        1 for row in study.scores if row["judge"] == "judge-strong" and row["run"] == "strong-01")
    assert scores(capsys, "export", folder, "--judges", "nobody") == (
        1, "", 'error: eval scores export: --judges: no reading or pair of the judge "nobody" in the study; use one '
               "of judge-strong, judge-cheap, reviewer\n")


def test_export_by_judges_warns_of_a_correction_that_crosses_judges(tmp_path, capsys):
    folder = small_study(tmp_path)
    base = {"schema": C.SCHEMA_SCORES, "run": "r1", "step": "final", "component": "depth", "rubric": "brief@1",
            "observed_at": AT}
    C.write_lines(folder / C.SCORES_FILE, C.SCHEMA_SCORES, [
        {**base, "id": "s1", "judge": "judge-a", "call": "k1", "value": 4},
        {**base, "id": "s2", "judge": "judge-b", "call": "k2", "value": 6, "replaces": "s1"},
        {**base, "id": "s3", "judge": "judge-a", "call": "k3", "value": 5}])
    code, out, err = scores(capsys, "export", folder, "--judges", "judge-a")
    assert code == 0 and [s["id"] for s in json.loads(out)["signals"]] == ["s1", "s3"]
    assert err == ('warning: scores.jsonl: the reading "s1" is corrected by "s2", a reading of the judge "judge-b", '
                   "which --judges leaves out: it is written as if it were not corrected\n")
    code, report = scores_json(capsys, "export", folder, "--judges", "judge-b")
    assert code == 0 and [s["id"] for s in report["runs"][0]["signals"]] == ["s2"]
    assert report["warnings"] == ['scores.jsonl: the reading "s2" corrects "s1", a reading of the judge "judge-a", '
                                  "which --judges leaves out"]
    assert scores(capsys, "export", folder, "--judges", "judge-a", "judge-b")[2] == ""


def test_import_reports_the_file_time_a_csv_gave_the_rows_that_had_none(tmp_path, capsys):
    seen = []
    for n, when in enumerate((1_700_000_000, 1_700_000_000 + 86_400)):  # the same bytes, modified a day apart
        form = write_csv(tmp_path / f"r{n}.csv", "id,run,component,judge,call,value", "s1,r1,depth,reviewer,k1,4")
        os.utime(form, (when, when))
        folder = small_study(tmp_path / f"s{n}")
        code, report = scores_json(capsys, "import", "--csv", form, "--rubric", RUBRIC, "--out", folder)
        assert code == 0
        seen.append((report, C.read_study(folder).scores[0]["observed_at"]))
    (first, at_first), (second, at_second) = seen
    assert [entry["sha256"] for entry in first["inputs"].values()] == [
        entry["sha256"] for entry in second["inputs"].values()]
    assert first["options"]["file_times"] == {"csv": at_first} and second["options"]["file_times"] == {"csv": at_second}
    assert _dt.datetime.fromisoformat(at_second) - _dt.datetime.fromisoformat(at_first) == _dt.timedelta(days=1)


def test_export_reports_the_items_file_the_study_reader_checks(tmp_path, capsys):
    folder = copy_study(tmp_path)
    C.write_lines(folder / C.ITEMS_FILE, C.SCHEMA_ITEMS, [{"id": "i1", "input": "case-01", "component": "depth"}])
    code, report = scores_json(capsys, "export", folder)
    assert code == 0 and sorted(report["inputs"]) == ["items", "judges", "pairs", "runs", "scores"]
    assert report["inputs"]["items"] == {"file": C.ITEMS_FILE, "sha256": sha(folder / C.ITEMS_FILE)}
    edit_lines(folder, C.ITEMS_FILE, lambda rows: [{key: value for key, value in row.items() if key != "input"}
                                                   for row in rows])
    assert scores(capsys, "export", folder)[0] == 1  # export reads it: a broken items file stops it
