"""The store and flows: it keeps OCP 0.4 runs that name a flow (`run.flow`) or state `control.checks`, refuses a
flow document by name, and `Store.flow_runs` gives one flow's runs, of one arm when asked, in start order."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from store_helpers import cli, finished_doc

from loopmath.store import Store, StoreError, ValidationFailed, fitjob

ROOT = Path(__file__).resolve().parents[2]
V04 = ROOT / "spec" / "examples" / "v0.4"
FLOW_DOC = V04 / "lit-brief.flow.ocp.json"
FLOW_RUNS = (V04 / "lit-brief.run-1.ocp.json", V04 / "lit-brief.run-2.ocp.json")
CHECKS_RUN = V04 / "two-checks.ocp.json"
NEITHER = ("the store takes OCP 0.4 run documents that name a flow (run.flow) or state control.checks; this one "
           "does neither, so write it as OCP 0.3")
AT_0 = "2026-11-03T09:00:00-08:00"
AT_1 = "2026-11-03T10:00:00-08:00"
AT_1_CET = "2026-11-03T19:00:00+01:00"  # the same instant as AT_1
AT_2_CET = "2026-11-03T19:01:00+01:00"  # one minute after AT_1
AT_EARLY_CET = "2026-11-03T17:59:00+01:00"  # a minute before AT_0, though after it as a string


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


@pytest.fixture(autouse=True)
def spawned(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(str(home)) or {"started": True, "pid": 0})
    return calls


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _flow_run(run: str, started: str, arm: str | None = "base", flow: str = "lit-brief") -> dict:
    doc = _read(FLOW_RUNS[0])
    doc["run"]["id"] = run
    doc["run"]["started_at"] = started
    doc["run"]["flow"]["id"] = flow
    if arm is None:
        del doc["run"]["flow"]["arm"]
    else:
        doc["run"]["flow"]["arm"] = arm
    return doc


# ------------------------------------------------------------------ what the store takes

@pytest.mark.parametrize("path", [*FLOW_RUNS, CHECKS_RUN], ids=lambda p: p.name)
def test_the_store_takes_0_4_runs_with_a_flow_or_checks(path, home):
    doc = _read(path)
    run = Store(home).import_run(doc)
    stored = Store(home).run_doc(run)
    assert run == doc["run"]["id"] and stored["ocp"] == "0.4"
    assert stored["run"].get("flow") == doc["run"].get("flow")
    assert stored["run"]["configuration"] == doc["run"]["configuration"]


@pytest.mark.parametrize("path", [*FLOW_RUNS, CHECKS_RUN], ids=lambda p: p.name)
def test_run_import_takes_them(path, capsys, home):
    code, out, err = cli(capsys, home, "run", "import", str(path), "--json")
    assert code == 0, err
    assert out["run"] == _read(path)["run"]["id"] and out["import"] == "new"


def test_run_import_refuses_a_flow_document_by_name(capsys, home):
    code, out, err = cli(capsys, home, "run", "import", str(FLOW_DOC))
    assert code == 1
    assert f"{FLOW_DOC} is a flow document (kind 'flow'), not a run: the store keeps runs; give it to the flow " \
           "commands as FLOW" in err
    assert Store(home).index_rows() == {}


@pytest.mark.parametrize("check", [True, False])
def test_import_run_refuses_a_flow_document_even_unchecked(check, home):
    with pytest.raises(StoreError) as caught:
        Store(home).import_run(_read(FLOW_DOC), check=check)
    assert str(caught.value) == ("this is a flow document (kind 'flow'), not a run: the store keeps runs; give it "
                                 "to the flow commands")
    assert not isinstance(caught.value, ValidationFailed) and Store(home).index_rows() == {}


def test_a_0_4_run_with_neither_a_flow_nor_checks_is_refused(capsys, home, tmp_path):
    doc = {**finished_doc("run-n"), "ocp": "0.4"}
    with pytest.raises(ValidationFailed) as caught:
        Store(home).import_run(doc)
    assert caught.value.result["errors"][0]["message"] == NEITHER
    assert str(caught.value) == f"strict OCP check failed: E005 {NEITHER}"
    path = tmp_path / "neither.ocp.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    code, out, err = cli(capsys, home, "run", "import", str(path))
    assert code == 1
    assert f"{path} declares OCP '0.4' but names no flow (run.flow) and states no control.checks; run import reads " \
           "0.3, and 0.4 runs that do either" in err
    assert Store(home).index_rows() == {}


def test_another_version_is_named_with_what_import_reads(capsys, home, tmp_path):
    path = tmp_path / "future.ocp.json"
    path.write_text(json.dumps({**finished_doc("run-f"), "ocp": "9.9"}), encoding="utf-8")
    code, out, err = cli(capsys, home, "run", "import", str(path))
    assert code == 1
    assert f"{path} declares OCP '9.9'; run import reads 0.3, and 0.4 runs that name a flow or state checks (0.1 " \
           "and 0.2 are migrated)" in err


# ------------------------------------------------------------------ the index and flow_runs

def test_index_rows_hold_flow_and_arm_only_for_flow_runs(home):
    store = Store(home)
    for doc in (_read(FLOW_RUNS[0]), _read(CHECKS_RUN), finished_doc("run-plain"), _flow_run("run-x", AT_0, None)):
        store.import_run(doc)
    rows = store.index_rows()
    assert (rows[_read(FLOW_RUNS[0])["run"]["id"]]["flow"], rows[_read(FLOW_RUNS[0])["run"]["id"]]["arm"]) == \
           ("lit-brief", "base")
    assert (rows["run-x"]["flow"], rows["run-x"]["arm"]) == ("lit-brief", "base")  # absent reads as base
    assert not {"flow", "arm"} & set(rows["run_ex_two_checks"]) and not {"flow", "arm"} & set(rows["run-plain"])


def _imported(home: Path, docs: list[dict]) -> Store:
    store = Store(home)
    for doc in docs:
        store.import_run(doc)
    return store


def _ids(rows: list[dict]) -> list[str]:
    return [r["run"] for r in rows]


def test_flow_runs_gives_one_flow_in_start_order_by_instant(home):
    store = _imported(home, [_flow_run("run-late", AT_2_CET), _flow_run("run-early", AT_1),
                             _flow_run("run-other-flow", AT_0, flow="other-flow"),
                             _flow_run("run-arm", AT_0, arm="sonnet-extract"), _flow_run("run-first", AT_EARLY_CET)])
    assert _ids(store.flow_runs("lit-brief")) == ["run-first", "run-arm", "run-early", "run-late"]
    assert _ids(store.flow_runs("lit-brief", "base")) == ["run-first", "run-early", "run-late"]
    assert _ids(store.flow_runs("lit-brief", arm="sonnet-extract")) == ["run-arm"]
    assert _ids(store.flow_runs("other-flow")) == ["run-other-flow"] and store.flow_runs("no-such-flow") == []
    rows = store.flow_runs("lit-brief")
    assert all(r["flow"] == "lit-brief" for r in rows)
    assert [r["arm"] for r in rows] == ["base", "sonnet-extract", "base", "base"]
    assert store.run_doc(rows[1]["run"])["run"]["flow"]["arm"] == "sonnet-extract"


@pytest.mark.parametrize("order", [("run-a", "run-b"), ("run-b", "run-a")])
def test_equal_instants_keep_index_order(order, home):
    starts = {"run-a": AT_1, "run-b": AT_1_CET}
    store = _imported(home, [_flow_run(run, starts[run]) for run in order])
    assert _ids(store.flow_runs("lit-brief")) == list(order)


def test_an_absent_arm_reads_as_base(home):
    store = _imported(home, [_flow_run("run-named", AT_1), _flow_run("run-unnamed", AT_2_CET, arm=None),
                             _flow_run("run-empty", AT_0, arm="")])
    assert _ids(store.flow_runs("lit-brief", "base")) == ["run-empty", "run-named", "run-unnamed"]
    assert {r["arm"] for r in store.flow_runs("lit-brief")} == {"base"}


def test_a_replaced_run_moves_with_its_new_start(home):
    store = _imported(home, [_flow_run("run-a", AT_1), _flow_run("run-b", AT_2_CET)])
    store.import_run(_flow_run("run-a", "2026-11-03T11:00:00-08:00"))
    assert _ids(store.flow_runs("lit-brief")) == ["run-b", "run-a"]


@pytest.mark.parametrize("flow", ["", None, 3])
def test_flow_runs_needs_a_flow_id(flow, home):
    with pytest.raises(ValueError, match=r"^flow: expected a flow id, got "):
        Store(home).flow_runs(flow)


def test_flow_runs_on_an_empty_store(home):
    assert Store(home).flow_runs("lit-brief") == []
    assert Store(home).flow_runs("lit-brief", "base") == []
