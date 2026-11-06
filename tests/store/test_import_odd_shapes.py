"""`run import` and `flow record --ocp` of a run naming the flow that `run import` cannot read before its strict
check: `run.signals` or `run.preferences` that is not a list, or `producer` that is not an object. Each is refused by
name (`run.signals is int, not a list`), exit 1, with no traceback, and nothing is written for it; a batch stores its
good files as before. A null member and a list holding a non-object are the strict check's to name, as before, and
so is `attempts` that is not a list (E010, E104). A stored run file nested deeper than the JSON reader reads is
refused by name, and the file is left as it is; finishing another run reads past it, as past any file it cannot
read."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
from store_helpers import cli

from loopmath.store import Store, fitjob
from loopmath.store.runs import dollars_past_float, is_finished
from tests.flows.test_flow_record import FLOW_FILE, PLAN, _doc, _row

# Each document, by the member set, its value and the refusal.
SHAPES = {"signals-5": ("run.signals", 5, "run.signals is int, not a list"),
          "preferences-5": ("run.preferences", 5, "run.preferences is int, not a list"),
          "producer-5": ("producer", 5, "producer is int, not an object"),
          "producer-list": ("producer", [1], "producer is list, not an object")}
# The strict check's, as on the base: the member set, its value, and the finding's value, type and place.
CONTROLS = {"signals-null": ("run.signals", None, None, "array", "$['run']['signals']"),
            "preferences-null": ("run.preferences", None, None, "array", "$['run']['preferences']"),
            "producer-null": ("producer", None, None, "object", "$['producer']"),
            "signals-list": ("run.signals", [5], 5, "object", "$['run']['signals'][0]")}
ATTEMPTS = {"5": (5, "int"), "true": (True, "bool"), "null": (None, "null"), "object": ({}, "dict")}
NESTED = "[" * 200000 + "]" * 200000  # deeper than the JSON reader reads
NONE_FINISHED = {"started": False, "reason": "no run was finished by this import; use --finish"}
NOTHING = "nothing was imported: fix the failed files and import them again"


@pytest.fixture(autouse=True)
def fits(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(str(home)) or {"started": True, "pid": 0})
    return calls


@pytest.fixture
def lb(tmp_path) -> Path:
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    shutil.copyfile(FLOW_FILE, folder / "flow.ocp.json")
    shutil.copyfile(PLAN, folder / "plan.json")
    return folder


@pytest.fixture
def docs(lb) -> dict:
    """Runs naming the flow, each with an end and not simulated: the one changed, and two good ones."""
    return {"doc": _doc(lb), "good": _doc(lb, _row("q07-r2")), "first": _doc(lb, _row("q07-r3"))}


def _write(path: Path, doc) -> str:
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return str(path)


def _tree(home: Path) -> dict[str, bytes]:
    """Every file of the store but its lock, by path."""
    if not home.exists():
        return {}
    return {p.relative_to(home).as_posix(): p.read_bytes() for p in sorted(home.rglob("*"))
            if p.is_file() and p.name != "lock"}


def _set(doc: dict, member: str, value) -> dict:
    out = copy.deepcopy(doc)
    *where, key = member.split(".")
    (out[where[0]] if where else out)[key] = value
    return out


def _brief(path: str, error: str) -> dict:
    """`run import --json --brief` of one file that failed."""
    return {"schema": "loopmath.run.import/1", "imported": 0, "new": 0, "unchanged": 0, "replaced": 0, "failed": 1,
            "refused": 0, "finished": 0, "already_finished": 0, "failures": [{"file": path, "error": error}],
            "more_failures": 0, "shipped_overlap": {}, "overlap_note": None, "fit": NONE_FINISHED, "next": NOTHING}


# ------------------------------------------------------------------ shapes `run import` cannot read


@pytest.mark.parametrize("shape", list(SHAPES))
def test_one_file_is_refused_by_name_and_nothing_is_written(shape, capsys, tmp_path, docs, fits):
    member, value, reason = SHAPES[shape]
    home, path = tmp_path / "lm", _write(tmp_path / "x.ocp.json", _set(docs["doc"], member, value))
    assert cli(capsys, home, "run", "import", path) == (1, "", f"error: {reason}\n")
    code, out, err = cli(capsys, home, "run", "import", path, "--json")
    assert code == 1 and err == f"error: {reason}\n"
    assert out == {"schema": "loopmath.run.import/1", "ok": False, "error": reason, "exit": 1}
    assert cli(capsys, home, "run", "import", path, "--json", "--brief") == (1, _brief(path, reason), "")
    assert _tree(home) == {} and fits == []


@pytest.mark.parametrize("shape", list(SHAPES))
def test_in_a_batch_the_file_is_named_and_the_good_one_stored(shape, capsys, tmp_path, docs):
    """The entry is the batch's generic failure (`exit` 1, no `refused`); the runs stored before keep their bytes,
    and the good file adds its run and its index row only."""
    member, value, reason = SHAPES[shape]
    home, bad_doc = tmp_path / "lm", _set(docs["doc"], member, value)
    good, bad = _write(tmp_path / "good.ocp.json", docs["good"]), _write(tmp_path / "bad.ocp.json", bad_doc)
    assert cli(capsys, home, "run", "import", _write(tmp_path / "first.ocp.json", docs["first"]))[0] == 0
    before = _tree(home)
    code, out, err = cli(capsys, home, "run", "import", good, bad, "--json")
    good_id = docs["good"]["run"]["id"]
    assert code == 1 and out["runs"] == [good_id] and (out["failed"], out["refused"]) == (1, 0)
    assert out["files"][1] == {"file": bad, "ok": False, "error": reason, "exit": 1}
    after, store = _tree(home), Store(home)
    assert not store.exists(bad_doc["run"]["id"])
    assert sorted(after) == sorted([*before, f"runs/{store.run_path(good_id).name}"])
    assert all(after[k] == before[k] for k in before if k != "runs/index.jsonl")
    old, new = before["runs/index.jsonl"], after["runs/index.jsonl"]
    assert new.startswith(old) and [json.loads(x)["run"] for x in new[len(old):].splitlines()] == [good_id]
    code, out, err = cli(capsys, tmp_path / "text", "run", "import", good, bad)
    assert code == 1 and out.splitlines()[0] == f"FAIL  {bad}: {reason}"


@pytest.mark.parametrize("mixed", [False, True], ids=["alone", "with-a-good-run"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_flow_record_refuses_it_by_name_and_records_nothing(shape, mixed, capsys, tmp_path, lb, docs, fits):
    member, value, reason = SHAPES[shape]
    home, bad_doc = tmp_path / "lm", _set(docs["doc"], member, value)
    bad = _write(tmp_path / "bad.ocp.json", bad_doc)
    given = [_write(tmp_path / "good.ocp.json", docs["good"]), bad] if mixed else [bad]
    plan = (lb / "plan.json").read_bytes()
    code, out, err = cli(capsys, home, "flow", "record", str(lb), "--ocp", *given)
    assert code == 1 and err == f"error: {bad}: {reason}\nnothing was recorded: 1 item refused\n"
    code, out, err = cli(capsys, home, "flow", "record", str(lb), "--ocp", *given, "--json")
    assert code == 1 and out == {"schema": "loopmath.flow-record/1", "flow": "lit-brief",
                                 "refused": [{"item": bad, "run": bad_doc["run"]["id"], "reason": reason}]}
    assert _tree(home) == {} and fits == [] and (lb / "plan.json").read_bytes() == plan


@pytest.mark.parametrize("control", list(CONTROLS))
def test_a_null_member_or_a_list_of_non_objects_is_the_strict_checks_as_before(control, capsys, tmp_path, lb, docs,
                                                                               fits):
    member, value, found, kind, at = CONTROLS[control]
    home, doc = tmp_path / "lm", _set(docs["doc"], member, value)
    path = _write(tmp_path / "x.ocp.json", doc)
    message = f"schema: {found!r} is not of type '{kind}'"
    error = f"strict OCP check failed: E010 {message}"
    assert cli(capsys, home, "run", "import", path) == (1, "", f"error: {error}\n  E010 {at} {message}\n")
    code, out, err = cli(capsys, home, "run", "import", path, "--json")
    assert code == 1 and out == {"schema": "loopmath.run.import/1", "ok": False, "error": error, "exit": 1,
                                 "validation": {"ok": False, "errors": [{"level": "error", "code": "E010", "path": at,
                                                                         "message": message}],
                                                "warnings": [], "checker": "loopmath.ocp"}}
    brief = cli(capsys, home, "run", "import", path, "--json", "--brief")
    assert brief == (1, _brief(path, f"{error}: E010 {at} {message}"), "")
    code, out, err = cli(capsys, home, "flow", "record", str(lb), "--ocp", path)
    assert code == 1 and err == f"error: {path}: E010 at {at}: {message}\nnothing was recorded: 1 item refused\n"
    assert _tree(home) == {} and fits == [] and json.loads(Path(path).read_text(encoding="utf-8")) == doc


# ------------------------------------------------------------------ attempts that is not a list


@pytest.mark.parametrize("given", list(ATTEMPTS))
def test_attempts_that_is_not_a_list_is_named_by_the_strict_check(given, capsys, tmp_path, lb, docs, fits):
    """The words before the dollar check walked `attempts`; that check reads no attempts from it."""
    value, kind = ATTEMPTS[given]
    home, doc = tmp_path / "lm", _set(docs["doc"], "attempts", value)
    assert dollars_past_float(doc) is None
    path = _write(tmp_path / "x.ocp.json", doc)
    message = f"schema: {value!r} is not of type 'array'"
    error = f"strict OCP check failed: E010 {message}; E104 attempts is {kind}, not a list"
    lines = f"  E010 $['attempts'] {message}\n  E104 $['attempts'] attempts is {kind}, not a list\n"
    assert cli(capsys, home, "run", "import", path) == (1, "", f"error: {error}\n{lines}")
    brief = cli(capsys, home, "run", "import", path, "--json", "--brief")
    assert brief == (1, _brief(path, f"{error}: E010 $['attempts'] {message}"), "")
    code, out, err = cli(capsys, home, "flow", "record", str(lb), "--ocp", path)
    assert code == 1 and err == (f"error: {path}: E010 at $['attempts']: {message}; E104 at $['attempts']: attempts "
                                 f"is {kind}, not a list\nnothing was recorded: 1 item refused\n")
    assert _tree(home) == {} and fits == []


# ------------------------------------------------------------------ a stored run file the reader cannot read


def _overwritten(capsys, home: Path, path: str, run: str, content: str) -> Path:
    """The run of `path` stored open, then its stored file replaced by `content`."""
    assert cli(capsys, home, "run", "import", path)[0] == 0
    stored = Store(home).run_path(run)
    stored.write_text(content, encoding="utf-8")
    return stored


def test_a_stored_run_file_nested_past_the_reader_is_refused_by_name(capsys, tmp_path, lb, docs, fits):
    home, run = tmp_path / "lm", docs["doc"]["run"]["id"]
    path, good = _write(tmp_path / "doc.ocp.json", docs["doc"]), _write(tmp_path / "good.ocp.json", docs["good"])
    stored = _overwritten(capsys, home, path, run, NESTED)
    reason = f"run {run}: the stored run file {stored} nests deeper than the JSON reader can read; nothing was written"
    before = _tree(home)
    assert cli(capsys, home, "run", "import", path) == (1, "", f"error: {reason}\n")
    code, out, err = cli(capsys, home, "run", "import", path, "--json")
    assert code == 1 and out == {"schema": "loopmath.run.import/1", "ok": False, "error": reason, "exit": 1}
    assert cli(capsys, home, "run", "import", path, "--json", "--brief") == (1, _brief(path, reason), "")
    code, out, err = cli(capsys, home, "flow", "record", str(lb), "--ocp", good, path)
    assert code == 1 and err == f"error: {path}: {reason}\nnothing was recorded: 1 item refused\n"
    assert _tree(home) == before and fits == []
    code, out, err = cli(capsys, home, "run", "import", good, path, "--json")
    assert code == 1 and out["files"][1] == {"file": path, "ok": False, "error": reason, "exit": 1}
    assert out["runs"] == [docs["good"]["run"]["id"]] and stored.read_text(encoding="utf-8") == NESTED


def test_a_stored_run_file_that_is_not_json_keeps_the_readers_words(capsys, tmp_path, docs):
    home, run = tmp_path / "lm", docs["doc"]["run"]["id"]
    path = _write(tmp_path / "doc.ocp.json", docs["doc"])
    _overwritten(capsys, home, path, run, "not json\n")
    before = _tree(home)
    assert cli(capsys, home, "run", "import", path) == (1, "", "error: Expecting value: line 1 column 1 (char 0)\n")
    assert _tree(home) == before


# ------------------------------------------------------------------ finishing runs over a stored file nested too deep


def _finished_beside(home: Path, good: str, nested: str, stored: Path) -> None:
    """The good run finished, in the index and in its stored file; the nested run open, its file as it was."""
    store = Store(home)
    assert {r["run"]: r["state"] for r in store.runs()} == {nested: "open", good: "finished"}
    assert is_finished(store.run_doc(good)) and stored.read_text(encoding="utf-8") == NESTED


@pytest.mark.parametrize("way", ["text", "json", "json-brief"])
def test_a_finish_batch_names_the_nested_run_and_finishes_the_good_one(way, capsys, tmp_path, docs, fits):
    """Finishing the good run reads the store's finished runs, and that read skips the nested file as it skips any
    file it cannot read: the batch ends as a batch with one failed file does, the nested run's entry the generic
    failed one (`exit` 1)."""
    home, run, good_id = tmp_path / "lm", docs["doc"]["run"]["id"], docs["good"]["run"]["id"]
    path, good = _write(tmp_path / "doc.ocp.json", docs["doc"]), _write(tmp_path / "good.ocp.json", docs["good"])
    stored = _overwritten(capsys, home, path, run, NESTED)
    reason = f"run {run}: the stored run file {stored} nests deeper than the JSON reader can read; nothing was written"
    before = list(fits)
    code, out, err = cli(capsys, home, "run", "import", path, good, "--finish",
                         *{"text": [], "json": ["--json"], "json-brief": ["--json", "--brief"]}[way])
    assert code == 1 and err == ""
    if way == "json":
        assert out["files"][0] == {"file": path, "ok": False, "error": reason, "exit": 1}
        assert (out["runs"], out["failed"], out["refused"], out["files"][1]["file"]) == ([good_id], 1, 0, good)
    else:
        assert out == ({"schema": "loopmath.run.import/1", "imported": 1, "new": 1, "unchanged": 0, "replaced": 0,
                        "failed": 1, "refused": 0, "finished": 1, "already_finished": 0,
                        "failures": [{"file": path, "error": reason}], "more_failures": 0, "shipped_overlap": {},
                        "overlap_note": None, "fit": {"started": True},
                        "next": "loopmath posterior --html once the background fit ends (loopmath status --json: "
                                "fit.running)"} if way == "json-brief" else
                       f"FAIL  {path}: {reason}\n{good_id}  finished\n1 run imported (1 new), 1 failed; 1 finished\n"
                       "refit: started in the background (pid 0)\n")
    _finished_beside(home, good_id, run, stored)
    assert fits == [*before, str(home)]


def test_finishing_a_run_over_a_store_holding_the_nested_file_finishes_it(capsys, tmp_path, docs):
    home, run, good_id = tmp_path / "lm", docs["doc"]["run"]["id"], docs["good"]["run"]["id"]
    stored = _overwritten(capsys, home, _write(tmp_path / "doc.ocp.json", docs["doc"]), run, NESTED)
    code, out, err = cli(capsys, home, "run", "import", _write(tmp_path / "good.ocp.json", docs["good"]), "--finish")
    assert code == 0 and err == "" and out.splitlines()[:2] == [good_id, f"run {good_id} finished"]
    _finished_beside(home, good_id, run, stored)


@pytest.mark.parametrize("content", ["not json\n", NESTED], ids=["not-json", "nested"])
def test_the_finished_runs_read_skips_a_stored_file_it_cannot_read(content, capsys, tmp_path, docs):
    """Two runs stored finished and one open, then one finished run's file replaced: a file that is not JSON is
    skipped, as before, and a nested one is skipped the same way; the other finished run is yielded as stored and
    the open one is not."""
    home, run = tmp_path / "lm", docs["doc"]["run"]["id"]
    good_id, open_id = docs["good"]["run"]["id"], docs["first"]["run"]["id"]
    given = [_write(tmp_path / "doc.ocp.json", docs["doc"]), _write(tmp_path / "good.ocp.json", docs["good"])]
    assert cli(capsys, home, "run", "import", *given, "--finish")[0] == 0
    assert cli(capsys, home, "run", "import", _write(tmp_path / "first.ocp.json", docs["first"]))[0] == 0
    store = Store(home)
    kept = store.run_doc(good_id)
    assert [d["run"]["id"] for d in store.finished_docs()] == sorted([run, good_id])
    store.run_path(run).write_text(content, encoding="utf-8")
    assert list(store.finished_docs()) == [kept]
    assert {r["run"]: r["state"] for r in store.runs()}[open_id] == "open"


@pytest.mark.parametrize("content", ["not json\n", NESTED], ids=["not-json", "nested"])
def test_the_finished_runs_read_goes_past_an_unreadable_file_that_comes_first(content, capsys, tmp_path, docs):
    """As above, with the finished run whose file is read first replaced: the read goes on past it and yields the
    other finished run."""
    home = tmp_path / "lm"
    given = [_write(tmp_path / f"{k}.ocp.json", docs[k]) for k in ("doc", "good")]
    assert cli(capsys, home, "run", "import", *given, "--finish")[0] == 0
    store = Store(home)
    first, later = sorted((docs[k]["run"]["id"] for k in ("doc", "good")), key=store.run_path)
    assert [d["run"]["id"] for d in store.finished_docs()] == [first, later]  # read in file order
    kept = store.run_doc(later)
    store.run_path(first).write_text(content, encoding="utf-8")
    assert list(store.finished_docs()) == [kept]
