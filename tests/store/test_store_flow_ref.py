"""The store keeps one flow per run id: a document for a stored run that names a flow (`run.flow`, its id and
version) is refused by name before anything is written when it names another flow id or version, or none
(`FlowChanged`), by `Store.import_run` and by `run import`, `--replace` or not, so no run file, index row, pending
signal or receipt changes. Versions read as OCP reads them (1 and 1.0 alike); a stored run that names no flow is
filled as before; the same flow with other content is the content checks' business, as before."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from store_helpers import cli, finished_doc

from loopmath import flows
from loopmath import harness_runs as HR
from loopmath.store import Store, fitjob
from loopmath.store import runs as R
from loopmath.store.home import FlowChanged, StoreError
from loopmath.types import Signal

ROOT = Path(__file__).resolve().parents[2]
FLOW_FILE = ROOT / "spec" / "examples" / "v0.4" / "lit-brief.flow.ocp.json"
PLAN = ROOT / "tests" / "fixtures" / "flows" / "plan.sample.json"
AT = "2026-11-03T11:06:00-08:00"
LIT = {"id": "lit-brief", "version": 1}
BEFORE = {"config": "cfg_aaaaaaaaaaaa", "p_success": {"mean": 0.7, "lo": 0.6, "hi": 0.8, "level": 0.8},
          "cost": {"usd": {"mean": 1.0, "lo": 0.5, "hi": 1.5, "level": 0.8},
                   "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}},
          "ell": {"usd": {"mean": 1.2, "lo": 0.6, "hi": 1.8, "level": 0.8},
                  "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}},
          "rounds": {"mean": 1, "lo": 1, "hi": 1, "level": 0.8}, "per_piece": {}, "support": 3}


@pytest.fixture(autouse=True)
def fits(monkeypatch):
    """The refits started, one entry each."""
    started: list[str] = []

    def spawn(home, **kw):
        started.append(str(home))
        return {"started": True, "pid": 0}

    monkeypatch.setattr(fitjob, "spawn_fit", spawn)
    return started


@pytest.fixture
def lb(tmp_path) -> Path:
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    shutil.copyfile(FLOW_FILE, folder / "flow.ocp.json")
    shutil.copyfile(PLAN, folder / "plan.json")
    return folder


def _flow_run(lb: Path, rid: str = "q07-r1", task: str = "q07") -> dict:
    """A run of the example flow (lit-brief version 1, arm base) as `adapt harness-runs` writes it: no simulate
    marker."""
    stage = {"agent": "api", "effort": "default", "round": 1, "cost": {"usd": 0.3}}
    row = {"schema": HR.SCHEMA, "id": rid, "task_id": task, "arm": "base", "status": "done", "workflow": "lit-brief",
           "flow": dict(LIT), "started_at": "2026-11-03T11:00:00-08:00", "ended_at": AT,
           "stages": [{**stage, "role": "worker", "piece": "scout", "model": "claude-sonnet-5-5"},
                      {**stage, "role": "worker", "piece": "extract", "model": "claude-opus-5-5", "effort": "high"},
                      {**stage, "role": "worker", "piece": "synthesize", "model": "gpt-6-astra", "effort": "high"},
                      {**stage, "role": "reviewer", "piece": "critic", "model": "claude-haiku-5-5"}],
           "outcome": {"judges": [{"id": "j-claude", "score": 0.8, "observed_at": AT}]}}
    return HR.to_ocp(row, flow=flows.load(lb), observed_default=AT)


def _with(doc: dict, **flow) -> dict:
    """`doc` with `run.flow`'s members set; a member given as None is dropped."""
    out = copy.deepcopy(doc)
    out["run"]["flow"].update(flow)
    for k in [k for k, v in flow.items() if v is None]:
        del out["run"]["flow"][k]
    return out


def _tree(home: Path) -> dict[str, str]:
    """Every file of the store but its lock: the run files, the index, pending signals and receipts."""
    return {p.relative_to(home).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(home.rglob("*")) if p.is_file() and p.name != "lock"} if home.exists() else {}


def _stored(home: Path, doc: dict, *, open_: bool = False) -> Store:
    """`doc` stored (finished, or open with a pending signal), with the receipt `run start --rec` writes."""
    store = Store(home)
    run = doc["run"]["id"]
    store.import_run(doc, finished=not open_)
    if open_:
        assert store.add_signal(run, Signal(id="sig_pending1", run=run, kind="verdict", name="tests", value="pass",
                                            tier="verified", source={"kind": "ci"}, observed_at=AT)) == "pending"
    store.write_receipt({"id": f"rct_{run}", "rec": None, "run": run, "fit": "fit_1", "before": BEFORE})
    return store


def _shown(key: dict | None) -> str:
    return "no flow" if key is None else f"flow {json.dumps(key['id'])} version {json.dumps(key['version'])}"


def _refusal(run: str, given: dict | None, stored: dict = LIT) -> str:
    return (f"run {run} is stored for {_shown(stored)}; this document names {_shown(given)}: the store does not "
            "replace a run of another flow; nothing was written")


def _write(path: Path, doc: dict) -> str:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ the flow key


@pytest.mark.parametrize(("a", "b", "same"), [
    (None, None, True), (LIT, None, False), (None, LIT, False), (LIT, dict(LIT), True),
    (LIT, {"id": "lit-brief", "version": 1.0}, True), ({"id": "lit-brief", "version": 1.0}, LIT, True),
    (LIT, {"id": "lit-brief-2", "version": 1}, False), (LIT, {"id": "lit-brief", "version": 2}, False),
    (LIT, {"id": "lit-brief", "version": True}, False), ({"id": "x", "version": 2}, {"id": "x", "version": 2.5}, False),
    ({"id": "x", "version": 2.5}, {"id": "x", "version": 2.5}, True), (LIT, {"id": "lit-brief", "version": "1"}, False),
    ({"id": "1", "version": 1}, {"id": 1, "version": 1}, False),
    ({"id": 1, "version": 1}, {"id": 1.0, "version": 1}, False),
    ({"id": "x", "version": 2 ** 53}, {"id": "x", "version": 2 ** 53 + 1}, False),
    ({"id": "x", "version": 2 ** 53 + 1}, {"id": "x", "version": float(2 ** 53)}, False),
    ({"id": "x", "version": 10 ** 400}, {"id": "x", "version": 10 ** 400}, True),
    ({"id": "x", "version": None}, {"id": "x", "version": None}, True),
    (LIT, {"id": "lit-brief", "version": None}, False),
    ({"id": {"member": True}, "version": 1}, {"id": {"member": 1}, "version": 1}, False),
    ({"id": "x", "version": {"member": True}}, {"id": "x", "version": {"member": 1}}, False),
    ({"id": [False], "version": 1}, {"id": [0], "version": 1}, False),
    ({"id": "x", "version": [False]}, {"id": "x", "version": [0]}, False),
    ({"id": {"a": [{"b": True}]}, "version": 1}, {"id": {"a": [{"b": 1}]}, "version": 1}, False),
    ({"id": "x", "version": [1]}, {"id": "x", "version": [1.0]}, False),
    ({"id": {"a": 1}, "version": 1}, {"id": {"a": 1, "b": 2}, "version": 1}, False),
    ({"id": "x", "version": [1]}, {"id": "x", "version": [1, 1]}, False),
    ({"id": {"a": 1, "b": [1, "x", None]}, "version": 1}, {"id": {"b": [1, "x", None], "a": 1}, "version": 1}, True),
    ({"id": "x", "version": [1, {"m": False}]}, {"id": "x", "version": [1, {"m": False}]}, True)],
    ids=["none", "stored-none", "given-none", "equal", "int-float", "float-int", "renamed", "re-versioned", "bool",
         "not-whole", "not-whole-equal", "string", "id-type", "id-int-float", "2**53", "2**53-float", "10**400",
         "no-version", "version-dropped", "object-id-bool", "object-version-bool", "array-id-bool",
         "array-version-bool", "deep-bool", "array-int-float", "object-extra-member", "array-longer",
         "object-equal", "array-equal"])
def test_the_flow_key_reads_versions_as_ocp_does_and_ids_by_type(a, b, same):
    """The whole-number rule is for the version itself; an object or array is the same only member by member and
    item by item, each of the same JSON type (so `true` is not 1 inside it either)."""
    assert R.same_flow(a, b) is same


def test_a_key_nested_past_the_recursion_limit_is_compared_without_recursion():
    def nested(leaf):
        value = leaf
        for _ in range(3 * sys.getrecursionlimit()):
            value = [value]
        return {"id": "x", "version": value}

    assert R.same_flow(nested(True), nested(True)) is True
    assert R.same_flow(nested(True), nested(1)) is False


def test_the_flow_key_is_the_id_and_version_as_held():
    assert R.flow_key({"run": {"flow": {"id": "f", "version": 1.0, "digest": "d", "arm": "x"}}}) == {"id": "f",
                                                                                                    "version": 1.0}
    assert R.flow_key({"run": {"flow": {}}}) == {"id": None, "version": None}
    assert [R.flow_key(d) for d in ({"run": {"flow": "f"}}, {"run": {}}, {}, {"run": "x"})] == [None] * 4


# ------------------------------------------------------------------ Store.import_run


GIVEN = pytest.mark.parametrize(("change", "given", "check"), [
    ({"id": "lit-brief-2"}, {"id": "lit-brief-2", "version": 1}, True),
    ({"version": 2}, {"id": "lit-brief", "version": 2}, True),
    ("drop", None, False), ("string", None, False)], ids=["renamed", "re-versioned", "no-flow", "flow-a-string"])


def _given(doc: dict, change) -> dict:
    if isinstance(change, dict):
        return _with(doc, **change)
    out = copy.deepcopy(doc)
    if change == "drop":
        del out["run"]["flow"]
    else:
        out["run"]["flow"] = "lit-brief"
    return out


@GIVEN
@pytest.mark.parametrize("finished", [True, False], ids=["finished", "open"])
@pytest.mark.parametrize("stored_open", [False, True], ids=["stored-finished", "stored-open"])
def test_store_import_run_refuses_a_run_of_another_flow_and_writes_nothing(change, given, check, finished,
                                                                           stored_open, lb, tmp_path):
    home, doc = tmp_path / "lm", _flow_run(lb)
    store = _stored(home, doc, open_=stored_open)
    before = _tree(home)
    with pytest.raises(FlowChanged) as caught:
        store.import_run(_given(doc, change), finished=finished, check=check)
    run = doc["run"]["id"]
    assert (str(caught.value), caught.value.run, caught.value.stored, caught.value.given) == (
        _refusal(run, given), run, LIT, given)
    assert _tree(home) == before


@pytest.mark.parametrize(("stored", "given"), [(1, 1.0), (1.0, 1), (10 ** 400, 10 ** 400)],
                         ids=["int-then-float", "float-then-int", "past-a-float"])
def test_a_whole_version_written_either_way_is_the_same_flow_and_replaced(stored, given, lb, tmp_path):
    """With the strict check on both sides."""
    home, doc = tmp_path / "lm", _flow_run(lb)
    Store(home).import_run(_with(doc, version=stored))
    new = _with(doc, version=given, digest="flw_000000000000")
    Store(home).import_run(new)
    held = Store(home).run_doc(doc["run"]["id"])["run"]["flow"]
    assert (type(held["version"]), held["version"], held["digest"]) == (type(given), given, "flw_000000000000")


@pytest.mark.parametrize(("stored", "given"), [
    ({"version": 1}, {"version": True}), ({"version": 1}, {"version": 2}), ({"version": 2}, {"version": 2.5}),
    ({"version": 1}, {"version": "1"}), ({"id": "1"}, {"id": 1}), ({"version": 2 ** 53}, {"version": 2 ** 53 + 1}),
    ({"version": 1}, {"version": None}), ({"id": {"member": True}}, {"id": {"member": 1}}),
    ({"version": {"member": True}}, {"version": {"member": 1}}), ({"id": [False]}, {"id": [0]}),
    ({"version": [False]}, {"version": [0]})],
    ids=["bool", "other-number", "not-whole", "string", "id-type", "2**53", "version-dropped", "object-id",
         "object-version", "array-id", "array-version"])
def test_a_version_or_id_of_another_value_or_type_is_another_flow(stored, given, lb, tmp_path):
    """Stored and given with `check=False`, as only such documents hold these types. The keys compare as JSON, so
    `true` and 1 differ in the assertion too."""
    home, doc = tmp_path / "lm", _flow_run(lb)
    store = Store(home)
    store.import_run(_with(doc, **stored), check=False)
    before = _tree(home)
    with pytest.raises(FlowChanged) as caught:
        store.import_run(_with(doc, **given), check=False)
    assert json.dumps(caught.value.stored) == json.dumps(R.flow_key(_with(doc, **stored)))
    assert json.dumps(caught.value.given) == json.dumps(R.flow_key(_with(doc, **given))) and _tree(home) == before


@pytest.mark.parametrize("key", [{"id": {"a": [1, {"b": True}]}}, {"version": [1, {"m": False}]}],
                         ids=["object-id", "array-version"])
def test_the_same_compound_key_given_again_is_the_same_flow_and_replaced(key, lb, tmp_path):
    """`check=False` on both sides; the key is held as given."""
    home, doc = tmp_path / "lm", _flow_run(lb)
    Store(home).import_run(_with(doc, **key), check=False)
    new = _with(doc, **copy.deepcopy(key), digest="flw_000000000000")
    Store(home).import_run(new, check=False)
    held = Store(home).run_doc(doc["run"]["id"])["run"]["flow"]
    assert json.dumps(held) == json.dumps(new["run"]["flow"]) and held["digest"] == "flw_000000000000"


def test_the_same_flow_with_another_digest_and_arm_is_replaced_as_before(lb, tmp_path):
    home, doc = tmp_path / "lm", _flow_run(lb)
    Store(home).import_run(doc)
    new = _with(doc, digest="flw_000000000000", arm="extract-sonnet")
    Store(home).import_run(new)
    assert Store(home).run_doc(doc["run"]["id"])["run"]["flow"] == new["run"]["flow"]


@pytest.mark.parametrize("given", ["flow-run", "plain-run"])
def test_a_stored_run_that_names_no_flow_is_filled_as_before(given, lb, tmp_path):
    """The producer's fill of a run `run start` made, which names no flow."""
    home, doc = tmp_path / "lm", _flow_run(lb)
    bare = copy.deepcopy(doc)
    del bare["run"]["flow"]
    Store(home).import_run(bare, finished=False, check=False)
    new = doc if given == "flow-run" else finished_doc(doc["run"]["id"], usd=0.1)
    Store(home).import_run(new, finished=False)
    assert R.flow_key(Store(home).run_doc(doc["run"]["id"])) == (LIT if given == "flow-run" else None)


def test_a_new_run_id_is_stored_as_before(lb, tmp_path):
    home = tmp_path / "lm"
    Store(home).import_run(_flow_run(lb))
    other = _with(_flow_run(lb, "q08-r1", "q08"), id="lit-brief-2")
    Store(home).import_run(other)
    assert R.flow_key(Store(home).run_doc(other["run"]["id"])) == {"id": "lit-brief-2", "version": 1}


@pytest.mark.parametrize("held", ["not json {", "[1, 2]"], ids=["not-json", "not-an-object"])
def test_a_stored_run_file_that_cannot_be_read_is_refused_by_name(held, lb, tmp_path):
    home, doc = tmp_path / "lm", _flow_run(lb)
    store = _stored(home, doc)
    store.run_path(doc["run"]["id"]).write_text(held, encoding="utf-8")
    before = _tree(home)
    with pytest.raises(StoreError) as caught:
        store.import_run(doc)
    assert type(caught.value) is StoreError and str(caught.value) == (
        f"run {doc['run']['id']}: the stored run file cannot be read, so the store cannot tell which flow it holds; "
        "nothing was written")
    assert _tree(home) == before


@pytest.mark.parametrize("check", [True, False], ids=["checked", "unchecked"])
def test_a_stored_run_file_nested_past_the_reader_is_refused_by_name(check, lb, tmp_path):
    home, doc = tmp_path / "lm", _flow_run(lb)
    store = _stored(home, doc)
    store.run_path(doc["run"]["id"]).write_text("[" * 200000 + "]" * 200000, encoding="utf-8")
    before = _tree(home)
    with pytest.raises(StoreError) as caught:
        store.import_run(doc, check=check)
    assert type(caught.value) is StoreError and str(caught.value) == (
        f"run {doc['run']['id']}: the stored run file cannot be read, so the store cannot tell which flow it holds; "
        "nothing was written")
    assert _tree(home) == before


# ------------------------------------------------------------------ run import


@pytest.mark.parametrize("more", [[], ["--replace"], ["--finish"], ["--replace", "--finish"]],
                         ids=["plain", "replace", "finish", "replace-finish"])
@pytest.mark.parametrize("json_", [False, True], ids=["text", "json"])
def test_run_import_of_one_file_refuses_it_with_no_offer_of_replace(more, json_, capsys, lb, tmp_path, fits):
    home, doc = tmp_path / "lm", _flow_run(lb)
    _stored(home, doc)
    before = _tree(home)
    given = {"id": "lit-brief-2", "version": 1}
    path = _write(tmp_path / "renamed.ocp.json", _with(doc, **given))
    code, out, err = cli(capsys, home, "run", "import", path, *more, *(["--json"] if json_ else []))
    text = _refusal(doc["run"]["id"], given)
    assert code == 1 and f"error: {text}\n" == err
    if json_:
        assert out == {"schema": "loopmath.run.import/1", "ok": False, "error": text, "exit": 1}
    else:
        assert out == ""
    assert "--replace" not in json.dumps(out) + err and _tree(home) == before and fits == []


def test_run_import_refuses_it_over_an_open_run_never_imported(capsys, lb, tmp_path, fits):
    """Today's silent path: an open run that was never imported is the producer's to fill, of any flow."""
    home, doc = tmp_path / "lm", _flow_run(lb)
    _stored(home, doc, open_=True)
    before = _tree(home)
    path = _write(tmp_path / "renamed.ocp.json", _with(doc, id="lit-brief-2"))
    code, out, err = cli(capsys, home, "run", "import", path)
    assert (code, out, err) == (1, "", f"error: {_refusal(doc['run']['id'], {'id': 'lit-brief-2', 'version': 1})}\n")
    assert _tree(home) == before and fits == []


def _batch(lb: Path, tmp_path: Path) -> tuple[Path, list[str], dict, dict]:
    """A store holding a finished flow run, and a folder of two files: that run renamed, and a good new run."""
    home, doc = tmp_path / "lm", _flow_run(lb)
    _stored(home, doc)
    src = tmp_path / "ocp"
    src.mkdir()
    good = _flow_run(lb, "q08-r1", "q08")
    files = [_write(src / "a.ocp.json", _with(doc, id="lit-brief-2")), _write(src / "b.ocp.json", good)]
    return home, files, doc, good


@pytest.mark.parametrize("given", ["folder", "files"])
def test_in_a_batch_the_run_is_refused_by_run_and_flows_and_the_rest_finished(given, capsys, lb, tmp_path, fits):
    home, files, doc, good = _batch(lb, tmp_path)
    run = doc["run"]["id"]
    held = Store(home).run_path(run).read_bytes()
    args = [str(Path(files[0]).parent)] if given == "folder" else files
    code, out, err = cli(capsys, home, "run", "import", *args, "--finish", "--json")
    assert code == 1 and out["runs"] == [good["run"]["id"]]
    assert (out["failed"], out["refused"], out["finished"]) == (1, 1, 1) and "hint" not in out
    [failed] = [f for f in out["files"] if not f["ok"]]
    assert failed == {"file": files[0], "ok": False, "error": _refusal(run, {"id": "lit-brief-2", "version": 1}),
                      "exit": 1, "refused": {"run": run, "flow": {"stored": LIT,
                                                                  "given": {"id": "lit-brief-2", "version": 1}}}}
    assert R.is_finished(Store(home).run_doc(good["run"]["id"])) and Store(home).run_path(run).read_bytes() == held
    assert len(fits) == 1 and out["fit"]["started"] is True


@pytest.mark.parametrize("form", ["text", "brief"])
def test_a_batch_names_the_refused_file_in_text_and_brief_with_no_hint(form, capsys, lb, tmp_path, fits):
    home, files, doc, good = _batch(lb, tmp_path)
    text = _refusal(doc["run"]["id"], {"id": "lit-brief-2", "version": 1})
    code, out, err = cli(capsys, home, "run", "import", *files, "--finish",
                         *(["--json", "--brief"] if form == "brief" else []))
    assert code == 1 and len(fits) == 1
    if form == "brief":
        assert (out["imported"], out["failed"], out["refused"], out["finished"]) == (1, 1, 1, 1)
        assert out["failures"] == [{"file": files[0], "error": text}] and "hint" not in out
    else:
        lines = out.splitlines()
        assert f"FAIL  {files[0]}: {text}" in lines and not any(line.startswith("hint:") for line in lines)
        assert "--replace" not in out


def test_a_batch_of_only_the_refused_run_starts_no_refit(capsys, lb, tmp_path, fits):
    home, files, doc, good = _batch(lb, tmp_path)
    Path(files[1]).rename(tmp_path / "kept-aside.ocp.json")
    before = _tree(home)
    code, out, err = cli(capsys, home, "run", "import", str(Path(files[0]).parent), "--finish", "--json")
    assert code == 1 and out["runs"] == [] and (out["failed"], out["refused"]) == (1, 1)
    assert out["fit"]["started"] is False and fits == [] and _tree(home) == before


def test_run_import_of_a_new_run_and_of_it_unchanged_reads_as_before(capsys, lb, tmp_path):
    home, doc = tmp_path / "lm", _flow_run(lb)
    path = _write(tmp_path / "run.ocp.json", doc)
    run, again = doc["run"]["id"], (f"next: stored open: import again with --finish to finish it (loopmath run import "
                                     f"{path} --finish); outcomes added since are kept\n")
    assert cli(capsys, home, "run", "import", path) == (0, f"{run}\n{again}", "")
    assert cli(capsys, home, "run", "import", path) == (0, f"{run}  (already in the store, unchanged)\n{again}", "")
    code, out, err = cli(capsys, home, "run", "import", path, "--json")
    assert (code, out["import"], out["already_in_store"], out["state"]) == (0, "unchanged", True, "open")
