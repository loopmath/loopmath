"""A given run document, row, task file, configuration file or flow document is read as JSON
(`loopmath.ocp.finite`): `NaN`, `Infinity`, `-Infinity` and a number past what a float holds are refused by file
and place, and nothing is written. Each command refuses such a file as it refuses one that is not JSON: the same
exit, and with `--json` the same envelope. Runs already stored are read as before.

`loopmath.ocp.finite` is imported inside each test that uses it, so this file collects on a build without it."""

from __future__ import annotations

import copy
import io
import json
import math
import shutil
from pathlib import Path

import pytest

from loopmath import cli, flows
from loopmath import harness_runs as HR
from loopmath.ocp import conformance
from loopmath.ocp.canonical import config_id, flow_digest
from loopmath.store import Store, fitjob

from ._common import EXAMPLES, ROOT

FLOW_FILE = EXAMPLES / "v0.4" / "lit-brief.flow.ocp.json"
PLAN = ROOT / "tests" / "fixtures" / "flows" / "plan.sample.json"
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
ANSWERS = json.loads((EXAMPLE / "answers.json").read_text(encoding="utf-8"))
INFERRED_ANSWERS = {**ANSWERS, "steps": [{k: s[k] for k in ("id", "reads", "writes")} for s in ANSWERS["steps"]]}
AT = "2026-11-03T11:06:00-08:00"
BIG = '"1e400 here"'  # written as 1e400 in the file's text
NOT_JSON = "{not json\n"
DOC_WORDS = ('run.ext["dev.example"].x: NaN is not a JSON number; attempts.0.cost.usd: Infinity is not a JSON '
             'number; attempts.1.cost.usd: 1e400 is past what a float holds')
ROW_WORDS = ('stages.0.cost.usd: Infinity is not a JSON number; stages.1.cost.usd: 1e400 is past what a float '
             'holds; ext["dev.example"].x: NaN is not a JSON number')


class _Pipe(io.StringIO):
    def isatty(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """No refit starts, and stdin is not a terminal."""
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})
    monkeypatch.setattr("sys.stdin", _Pipe(""))


def _stage(piece: str, role: str, model: str, effort: str, usd: float) -> dict:
    return {"role": role, "piece": piece, "agent": "api", "model": model, "effort": effort, "round": 1,
            "cost": {"usd": usd}}


def _row(rid: str = "q07-r1", **more) -> dict:
    """A logged run of the example's four steps, naming no flow."""
    return {"schema": HR.SCHEMA, "id": rid, "task_id": "q07", "arm": "base", "status": "done", "workflow": "lit-brief",
            "stages": [_stage("scout", "worker", "claude-sonnet-5-5", "default", 0.3),
                       _stage("extract", "worker", "claude-opus-5-5", "high", 1.0),
                       _stage("synthesize", "worker", "gpt-6-astra", "high", 0.6),
                       _stage("critic", "reviewer", "claude-haiku-5-5", "default", 0.05)],
            "outcome": {"name": "quality", "score": 0.78, "scale": "fraction"},
            "started_at": "2026-11-03T11:00:00-08:00", "ended_at": AT, **more}


def _flow_row(rid: str = "q07-f1") -> dict:
    """A run of the example flow's base arm, scored by its two judges."""
    row = _row(rid, flow={"id": "lit-brief", "version": 1})
    row["outcome"] = {"judges": [{"id": "j-claude", "score": 0.8, "observed_at": "2026-11-03T11:07:00-08:00"},
                                 {"id": "j-astra", "score": 0.7, "observed_at": "2026-11-03T11:08:00-08:00"}]}
    return row


def _bad_doc(doc: dict) -> dict:
    doc = copy.deepcopy(doc)
    doc["run"]["ext"] = {"dev.example": {"x": float("nan")}}
    doc["attempts"][0]["cost"]["usd"] = float("inf")
    doc["attempts"][1]["cost"]["usd"] = json.loads(BIG)
    return doc


def _bad_row(row: dict) -> dict:
    row = copy.deepcopy(row)
    row["stages"][0]["cost"]["usd"] = float("inf")
    row["stages"][1]["cost"]["usd"] = json.loads(BIG)
    row["ext"] = {"dev.example": {"x": float("nan")}}
    return row


def _text(value) -> str:
    """JSON text as Python's writer gives it (NaN and Infinity as tokens), with 1e400 written in."""
    return json.dumps(value).replace(BIG, "1e400")


def _strict(text: str):
    def refuse(token):
        raise AssertionError(f"{token} in written JSON")

    return json.loads(text, parse_constant=refuse)


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under `root` but a store's lock, by path."""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "lock"}


def _main(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _strings(value) -> list[str]:
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return [value] if isinstance(value, str) else []


@pytest.fixture
def lb(tmp_path) -> Path:
    """The example flow with a plan of two arms."""
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    shutil.copyfile(FLOW_FILE, folder / "flow.ocp.json")
    shutil.copyfile(PLAN, folder / "plan.json")
    return folder


def _refused_as_not_json(capsys, tmp_path: Path, argv: list[str], path: Path, text: str, words: str,
                         json_mode: bool) -> str:
    """`argv` reading `path`: with `text` it is refused naming `words` and writes nothing, with the exit (and with
    `--json` the envelope's members) it gives for a file that is not JSON. The refusal's stdout is returned."""
    argv = argv + (["--json"] if json_mode else [])
    shown = []
    for given in (NOT_JSON, text):
        path.write_text(given, encoding="utf-8")
        before = _tree(tmp_path)
        code, out, err = _main(capsys, *argv)
        assert _tree(tmp_path) == before
        shown.append((code, sorted(_strict(out)) if json_mode else None))
    assert code != 0 and shown[0] == shown[1]
    found = _strings(_strict(out)) if json_mode else err.splitlines()
    assert any(words in line for line in found), (words, out, err)
    return out


# ------------------------------------------------------------------ the reader

@pytest.mark.parametrize("text, words", [
    ("NaN", "the document: NaN is not a JSON number"),
    ("Infinity", "the document: Infinity is not a JSON number"),
    ("-Infinity", "the document: -Infinity is not a JSON number"),
    ("1e400", "the document: 1e400 is past what a float holds"),
    ("-1e400", "the document: -1e400 is past what a float holds"),
    ('{"a": {"b": NaN}}', "a.b: NaN is not a JSON number"),
    ("[1, [2, Infinity]]", "1.1: Infinity is not a JSON number"),
    ('{"stages": [{"cost": {"usd": 1E+400}}]}', "stages.0.cost.usd: 1E+400 is past what a float holds"),
    ('{"run": {"ext": {"dev.example": {"x": -Infinity}}}}',
     'run.ext["dev.example"].x: -Infinity is not a JSON number'),
    ('{"": NaN}', '[""]: NaN is not a JSON number'),
    ('{"12": [NaN]}', '["12"].0: NaN is not a JSON number'),
    ('{"a b": {"c": NaN}}', '["a b"].c: NaN is not a JSON number'),
    ('{"a[0]": NaN}', '["a[0]"]: NaN is not a JSON number'),
    ('{"say \\"x\\"": NaN}', '["say \\"x\\""]: NaN is not a JSON number'),
    ('{"line\\nbreak": NaN}', '["line\\nbreak"]: NaN is not a JSON number'),
], ids=["nan", "infinity", "minus infinity", "past a float", "past a float, negative", "in an object",
        "in a list", "a cost", "a member name with a dot", "empty name", "digits name", "white space name",
        "brackets name", "quotes name", "line break name"])
def test_the_reader_names_each_place_and_its_words(text, words):
    from loopmath.ocp import finite

    with pytest.raises(finite.NotFinite) as caught:
        finite.loads(text)
    assert isinstance(caught.value, ValueError)
    assert str(caught.value) == words and caught.value.places == [words]


def test_five_places_name_three_and_count_the_rest():
    from loopmath.ocp import finite

    with pytest.raises(finite.NotFinite) as caught:
        finite.loads('{"a": NaN, "b": [Infinity, 1e400], "c": {"d": -Infinity}, "e": -1e400}')
    assert str(caught.value) == ("a: NaN is not a JSON number; b.0: Infinity is not a JSON number; b.1: 1e400 is "
                                 "past what a float holds and 2 more")
    assert caught.value.places[3:] == ["c.d: -Infinity is not a JSON number", "e: -1e400 is past what a float holds"]


@pytest.mark.parametrize("text", ["1.7976931348623157e308", "5e-324", "1e-400", "-0.0", "1" + "0" * 400, '"NaN"',
                                  '{"Infinity": ["-Infinity", 0.5]}', '{"x":1,"x":2}'],
                         ids=["largest float", "smallest float", "under range", "negative zero",
                              "an integer of 401 digits", "a string", "names and strings", "a member given twice"])
def test_what_json_loads_takes_is_read_as_it_reads_it(text):
    from loopmath.ocp import finite

    got, want = finite.loads(text), json.loads(text)
    assert json.dumps(got) == json.dumps(want) and type(got) is type(want)


@pytest.mark.parametrize("text, words", [
    ('{"x":NaN,"x":1}', "x: NaN is not a JSON number"),
    ('{"x":Infinity,"x":null}', "x: Infinity is not a JSON number"),
    ('{"x":1e400,"x":2}', "x: 1e400 is past what a float holds"),
    ('{"x":{"a":-Infinity},"x":{}}', "x.a: -Infinity is not a JSON number"),
], ids=["nan", "infinity", "past a float", "in an object given again"])
def test_a_value_a_later_member_replaces_is_refused_too(text, words):
    """The token is in the text though the value read is the later member's."""
    from loopmath.ocp import finite

    with pytest.raises(finite.NotFinite) as caught:
        finite.loads(text)
    assert str(caught.value) == words and caught.value.places == [words]


# ------------------------------------------------------------------ the readers of given files

@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_run_import_refuses_a_document_by_file_and_place(json_mode, tmp_path, capsys):
    home = tmp_path / "home"
    good = tmp_path / "good.ocp.json"
    good.write_text(json.dumps(HR.to_ocp(_row("q07-r0"), observed_default=AT)), encoding="utf-8")
    assert _main(capsys, "run", "import", str(good), "--home", str(home))[0] == 0
    path = tmp_path / "run.ocp.json"
    _refused_as_not_json(capsys, tmp_path, ["run", "import", str(path), "--home", str(home)], path,
                         _text(_bad_doc(HR.to_ocp(_row(), observed_default=AT))), f"{path}: {DOC_WORDS}", json_mode)


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_flow_record_refuses_a_document_by_file_and_place(json_mode, lb, tmp_path, capsys):
    home = tmp_path / "home"
    path = tmp_path / "run.ocp.json"
    doc = HR.to_ocp(_flow_row(), flow=flows.load(lb), observed_default=AT)
    out = _refused_as_not_json(capsys, tmp_path, ["flow", "record", str(lb), "--ocp", str(path), "--home", str(home)],
                               path, _text(_bad_doc(doc)), DOC_WORDS if json_mode else f"{path}: {DOC_WORDS}",
                               json_mode)
    if json_mode:  # the file is the refused item
        assert json.loads(out)["refused"] == [{"item": str(path), "run": None, "reason": DOC_WORDS}]


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_flow_init_refuses_a_document_by_file_and_place(json_mode, tmp_path, capsys):
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps(INFERRED_ANSWERS), encoding="utf-8")
    path = tmp_path / "run.ocp.json"
    _refused_as_not_json(capsys, tmp_path, ["flow", "init", str(tmp_path / "f"), "--from-ocp", str(path),
                                            "--answers", str(answers)],
                         path, _text(_bad_doc(HR.to_ocp(_row(), observed_default=AT))), f"{path}: {DOC_WORDS}",
                         json_mode)


def _rows_argv(command: str, path: Path, tmp_path: Path, lb: Path) -> list[str]:
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps(INFERRED_ANSWERS), encoding="utf-8")
    home = str(tmp_path / "home")
    return {"flow record": ["flow", "record", str(lb), "--rows", str(path), "--home", home],
            "adapt": ["adapt", "harness-runs", str(path), "--out", str(tmp_path / "out")],
            "flow init": ["flow", "init", str(tmp_path / "f"), "--from-rows", str(path), "--answers", str(answers)],
            "gate": ["gate", "--runs", str(path), "--base", "arm:base", "--cand", "arm:other", "--home", home]}[command]


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
@pytest.mark.parametrize("command", ["flow record", "adapt", "flow init", "gate"])
def test_a_rows_reader_refuses_a_row_by_file_line_and_place(command, json_mode, lb, tmp_path, capsys):
    row = _flow_row if command == "flow record" else _row
    path = tmp_path / "rows.jsonl"
    text = json.dumps(row("q07-a")) + "\n" + _text(_bad_row(row("q07-b"))) + "\n"
    _refused_as_not_json(capsys, tmp_path, _rows_argv(command, path, tmp_path, lb), path, text,
                         f"{path}:2: {ROW_WORDS}", json_mode)


def test_a_known_member_gets_the_readers_words(tmp_path, capsys):
    """A row whose only such value is its cost: the reader refuses it first, not the rows checker."""
    row = {k: v for k, v in _row().items() if k not in ("stages", "workflow")}
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({**row, "cost": {"usd": float("nan")}}) + "\n", encoding="utf-8")
    code, out, err = _main(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "out"))
    assert code == 1 and not (tmp_path / "out").exists()
    assert err.splitlines() == [f"error: adapt harness-runs: {path}:1: cost.usd: NaN is not a JSON number"]


# ------------------------------------------------------------------ what is written

def test_what_run_import_and_flow_record_write_is_strict_json(lb, tmp_path, capsys):
    home = tmp_path / "home"
    plain, run = tmp_path / "plain.ocp.json", tmp_path / "run.ocp.json"
    plain.write_text(json.dumps(HR.to_ocp(_row(), observed_default=AT)), encoding="utf-8")
    run.write_text(json.dumps(HR.to_ocp(_flow_row(), flow=flows.load(lb), observed_default=AT)), encoding="utf-8")
    assert _main(capsys, "run", "import", str(plain), "--home", str(home))[0] == 0
    assert _main(capsys, "flow", "record", str(lb), "--ocp", str(run), "--home", str(home))[0] == 0
    index = (home / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    files = sorted((home / "runs").glob("*.ocp.json"))
    assert len(index) == 2 and len(files) == 2
    for line in index:
        _strict(line)
    for p in files:
        _strict(p.read_text(encoding="utf-8"))


def test_a_run_stored_with_nan_reads_as_before(lb, tmp_path, capsys):
    """A run stored with a NaN dollar cost (an older tool, or a store written by hand): flow status counts it as
    dollars past what a float holds, and compare as no dollar cost."""
    home = tmp_path / "home"
    store = Store(str(home))
    for task, usd in (("q07", 0.25), ("q08", float("nan"))):
        doc = HR.to_ocp({**_flow_row(f"{task}-f1"), "task_id": task}, flow=flows.load(lb), observed_default=AT)
        doc["attempts"][0]["cost"]["usd"] = usd
        store.import_run(doc, finished=True, check=False)
    assert "NaN" in "".join(p.read_text(encoding="utf-8") for p in (home / "runs").glob("*.ocp.json"))
    code, out, err = _main(capsys, "flow", "status", str(lb), "--home", str(home), "--json")
    assert code == 0, err
    status = json.loads(out)
    assert "1 of 2 runs has an attempt with dollars past what a float holds" in json.dumps(status)



def test_runs_stored_with_nan_fit_and_compare_as_before(tmp_path, capsys):
    """Plain runs, one stored with a NaN dollar cost: the fit leaves that attempt out of its cost rows (15 of 16), and
    compare counts the run as having no dollar cost."""
    home = tmp_path / "home"
    store = Store(str(home))
    for task, arm, usd in (("q07", "base", 0.3), ("q08", "base", 0.3), ("q07", "cand", 0.2),
                           ("q08", "cand", float("nan"))):
        doc = HR.to_ocp(_row(f"{task}-{arm}", task_id=task, arm=arm), observed_default=AT)
        doc["attempts"][0]["cost"]["usd"] = usd
        store.import_run(doc, finished=True, check=False)
    code, out, err = _main(capsys, "check-change", "from-runs", "--base-label", "arm=base", "--cand-label", "arm=cand",
                           "--rule", "tests", "--json", "--home", str(home))
    assert code == 0, err
    comparison = json.loads(out)["comparison"]
    assert comparison["counts"]["no_cost"] == {"base": 0, "cand": 1}
    assert "1 candidate run has no dollar cost; the cost ranges leave it out" in comparison["reasons"]
    code, out, err = _main(capsys, "fit", "--home", str(home), "--json")
    assert code == 0, err
    fit = json.loads(out)
    assert fit["fit"]["n_runs"]["user"] == 4 and fit["heads"]["cost"]["rows_by_source"]["user"] == 15


# ------------------------------------------------------------------ run start

def _start_argv(task: Path, home: Path) -> list[str]:
    return ["run", "start", "--task-file", str(task), "--workflow", "solo", "--set", "implement=api:claude-sonnet-5-5",
            "--source", "usual", "--home", str(home)]


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_run_start_refuses_a_task_file_by_file_and_place(json_mode, tmp_path, capsys):
    home, task = tmp_path / "home", tmp_path / "task.json"
    _refused_as_not_json(capsys, tmp_path, _start_argv(task, home), task,
                         '{"type": "feature", "repo": "acme/api", "title": "t", "difficulty": NaN}\n',
                         f"{task}: difficulty: NaN is not a JSON number", json_mode)


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_run_start_refuses_a_task_file_whose_nan_a_later_member_replaces(json_mode, tmp_path, capsys):
    home, task = tmp_path / "home", tmp_path / "task.json"
    _refused_as_not_json(capsys, tmp_path, _start_argv(task, home), task,
                         '{"type": "feature", "repo": "acme/api", "title": "t", "difficulty": NaN, '
                         '"difficulty": 0.5}\n', f"{task}: difficulty: NaN is not a JSON number", json_mode)
    assert _main(capsys, *_start_argv(task, home))[0] == 1 and not (home / "runs").exists()


def _configuration(temperature: float) -> dict:
    given = json.loads((EXAMPLE / "flow.ocp.json").read_text(encoding="utf-8"))["flow"]["configuration"]
    given["settings"]["scout"]["options"] = {"temperature": temperature}
    given["id"] = config_id(given["workflow"], given["settings"], resolve=conformance.catalog_resolver)
    return given


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_run_start_refuses_a_configuration_file_by_file_and_place(json_mode, tmp_path, capsys):
    home, task, cfg = tmp_path / "home", tmp_path / "task.json", tmp_path / "cfg.json"
    task.write_text('{"type": "feature", "repo": "acme/api", "title": "t"}\n', encoding="utf-8")
    argv = ["run", "start", "--task-file", str(task), "--config", str(cfg), "--source", "usual", "--home", str(home)]
    _refused_as_not_json(capsys, tmp_path, argv, cfg, _text(_configuration(float("inf"))),
                         f"{cfg}: settings.scout.options.temperature: Infinity is not a JSON number", json_mode)


def test_run_start_with_finite_values_stores_strict_json(tmp_path, capsys):
    home, task, cfg = tmp_path / "home", tmp_path / "task.json", tmp_path / "cfg.json"
    task.write_text('{"type": "feature", "repo": "acme/api", "title": "t", "difficulty": 0.5}\n', encoding="utf-8")
    cfg.write_text(json.dumps(_configuration(0.2)), encoding="utf-8")
    assert _main(capsys, *_start_argv(task, home))[0] == 0
    assert _main(capsys, "run", "start", "--task-file", str(task), "--config", str(cfg), "--source", "usual",
                 "--home", str(home))[0] == 0
    files = sorted((home / "runs").glob("*.ocp.json"))
    assert len(files) == 2
    for p in files:
        _strict(p.read_text(encoding="utf-8"))
    for line in (home / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines():
        _strict(line)


# ------------------------------------------------------------------ recommend

@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_recommend_refuses_a_task_file_before_a_recommendation_is_written(json_mode, tmp_path, capsys):
    """`run start --rec` writes the stored recommendation's task into the run file, so the task file is refused
    before `recs/` gains a file; with finite values the same store gives a recommendation."""
    home, task = tmp_path / "home", tmp_path / "task.json"
    assert _main(capsys, "fit", "--home", str(home))[0] == 0
    argv = ["recommend", "--task-file", str(task), "--home", str(home)]
    _refused_as_not_json(capsys, tmp_path, argv, task, '{"type": "bug_fix", "repo": "acme/api", "difficulty": NaN}\n',
                         f"{task}: difficulty: NaN is not a JSON number", json_mode)
    assert _main(capsys, *argv)[0] == 1 and sorted((home / "recs").glob("*")) == []
    task.write_text('{"type": "bug_fix", "repo": "acme/api", "difficulty": 0.5}\n', encoding="utf-8")
    assert _main(capsys, *argv)[0] == 0
    [rec] = (home / "recs").glob("*.json")
    _strict(rec.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the flow document

def _nan_flow(tmp_path: Path) -> Path:
    """The example folder, its flow.toml left out, its flow.ocp.json holding NaN in a step option, the
    configuration id and the digest computed over it."""
    folder = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, folder, ignore=shutil.ignore_patterns("flow.toml"))
    doc = json.loads((folder / "flow.ocp.json").read_text(encoding="utf-8"))
    c = doc["flow"]["configuration"]
    c["settings"]["scout"]["options"] = {"temperature": float("nan")}
    c["id"] = config_id(c["workflow"], c["settings"], resolve=conformance.catalog_resolver)
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    (folder / "flow.ocp.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return folder


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_flow_plan_refuses_a_flow_document_by_file_and_place(json_mode, tmp_path, capsys):
    folder = _nan_flow(tmp_path)
    path = folder / "flow.ocp.json"
    argv = ["flow", "plan", str(folder), "--home", str(tmp_path / "home")]
    _refused_as_not_json(capsys, tmp_path, argv, path, path.read_text(encoding="utf-8"),
                         f"{path}: flow.configuration.settings.scout.options.temperature: NaN is not a JSON number",
                         json_mode)
    assert not (folder / "plan.json").exists()


def test_the_example_plans_as_before(tmp_path, capsys):
    folder = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, folder)
    code, out, err = _main(capsys, "flow", "plan", str(folder), "--home", str(tmp_path / "home"), "--json")
    assert code == 0, err
    assert json.loads(out)["plan"]["id"] == "pln_97f47f3321db" and (folder / "plan.json").is_file()
