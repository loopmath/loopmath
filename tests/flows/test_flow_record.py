"""`loopmath flow record`: a flow's runs into the store, from rows or OCP run documents. Every item is checked
against the flow and the store before anything is written, each refusal in its own words, and one refusal writes
nothing; the runs are stored finished, and one refit starts when a run was new. `run import --allow-simulated` is
pinned in tests/store/test_run_import_simulated.py and `flow score` in test_flow_score.py."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from loopmath import cli, flows
from loopmath import harness_runs as HR
from loopmath.belief.outcome import outcome_evidence
from loopmath.belief.rules import stored_rule
from loopmath.flows import record, simulate
from loopmath.flows.next import next_runs
from loopmath.ocp.canonical import flow_digest
from loopmath.store import Store, fitjob

ROOT = Path(__file__).resolve().parents[2]
FLOW_FILE = ROOT / "spec" / "examples" / "v0.4" / "lit-brief.flow.ocp.json"
PLAN = ROOT / "tests" / "fixtures" / "flows" / "plan.sample.json"
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
PLAIN_ROWS = ROOT / "spec" / "examples" / "harness-runs" / "auto-research-loop.jsonl"
AT = "2026-11-03T11:06:00-08:00"
DIGEST = "flw_774ab7a14c50"
BASE_ID = "cfg_9bc4f6e876f9"
SONNET_ID = "cfg_b8de56a265ea"
SIMULATE = "dev.loopmath.simulate"
FLOW_EXT = "dev.loopmath.flow"
NO_FLOW = "names no flow; flow record takes this flow's rows (convert others with loopmath adapt harness-runs)"
FLAG = "--allow-unplanned records it under arm 'unplanned'"
RECORD_KEYS = ["schema", "flow", "digest", "recorded", "findings", "advisories", "fit", "plan"]
RECORDED_KEYS = ["run", "arm", "input", "import", "ended_at_from"]


@pytest.fixture(autouse=True)
def fits(monkeypatch):
    """The refits started, by home; none runs."""
    calls: list[Path] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(Path(home)) or {"started": True, "pid": 0})
    return calls


@pytest.fixture
def lb(tmp_path) -> Path:
    """The example flow with a plan of two arms, base and extract-sonnet."""
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    shutil.copyfile(FLOW_FILE, folder / "flow.ocp.json")
    shutil.copyfile(PLAN, folder / "plan.json")
    return folder


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "home"


def _stage(piece, role, model, effort, usd) -> dict:
    return {"role": role, "piece": piece, "agent": "api", "model": model, "effort": effort, "round": 1,
            "cost": {"usd": usd}}


def _row(rid: str = "q07-r1", task: str = "q07", **over) -> dict:
    """One pass of the flow's four pieces on `task`, scored by its two model judges."""
    row = {"schema": HR.SCHEMA, "id": rid, "task_id": task, "arm": "base", "status": "done", "workflow": "lit-brief",
           "flow": {"id": "lit-brief", "version": 1},
           "stages": [_stage("scout", "worker", "claude-sonnet-5-5", "default", 0.3),
                      _stage("extract", "worker", "claude-opus-5-5", "high", 1.0),
                      _stage("synthesize", "worker", "gpt-6-astra", "high", 0.6),
                      _stage("critic", "reviewer", "claude-haiku-5-5", "default", 0.05)],
           "outcome": {"judges": [{"id": "j-claude", "score": 0.8, "observed_at": "2026-11-03T11:07:00-08:00"},
                                  {"id": "j-astra", "score": 0.7, "observed_at": "2026-11-03T11:08:00-08:00"}]},
           "started_at": "2026-11-03T11:00:00-08:00", "ended_at": AT}
    row.update(over)
    return row


def _doc(folder: Path, row: dict | None = None, **kw) -> dict:
    return HR.to_ocp(row or _row(), flow=flows.load(folder), observed_default=AT, **kw)


def _write(path: Path, doc) -> str:
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return str(path)


def _rows(path: Path, *rows: dict) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _flow(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["flow", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _record(capsys, folder: Path, home: Path, *argv: str) -> tuple[int, str, str]:
    return _flow(capsys, "record", str(folder), "--home", str(home), *argv)


def _store(home: Path) -> dict[str, bytes]:
    """Every file of the store but its lock, by path."""
    if not home.exists():
        return {}
    return {p.relative_to(home).as_posix(): p.read_bytes() for p in sorted(home.rglob("*"))
            if p.is_file() and p.name != "lock"}


def _refused(err: str) -> list[str]:
    return [line.removeprefix("error: ") for line in err.splitlines() if line.startswith("error: ")]


# Rows -------------------------------------------------------------------------------------------------------------

def test_rows_are_recorded_finished_with_the_flow(capsys, tmp_path, lb, home, fits):
    rows = _rows(tmp_path / "rows.jsonl", _row(), _row("q07-r2", "q08"))
    code, out, err = _record(capsys, lb, home, "--rows", rows)
    assert code == 0 and err == ""
    store = Store(home)
    ids = [row["run"] for row in store.flow_runs("lit-brief")]
    assert len(ids) == 2
    assert out.splitlines() == [f"recorded {ids[0]} arm base input q07",
                                f"recorded {ids[1]} arm base input q08",
                                "2 runs recorded, 0 already recorded",
                                "refit: started in the background (pid 0)"]
    assert fits == [home]
    flow = flows.load(lb)
    for rid in ids:
        doc = store.run_doc(rid)
        assert doc["run"]["flow"] == {"id": "lit-brief", "version": 1, "digest": DIGEST, "arm": "base"}
        assert doc["run"]["acceptance_rule"] == flow.flow.acceptance_rule
        assert doc["run"]["configuration"]["id"] == BASE_ID and doc["run"]["ended_at"]
        assert [e["type"] for e in doc["events"]].count("run_finished") == 1
        assert store.index_rows()[rid]["state"] == "finished"
    assert sorted(d["run"]["id"] for d in store.finished_docs()) == sorted(ids)


def test_the_json_report_has_its_pinned_keys(capsys, tmp_path, lb, home):
    rows = _rows(tmp_path / "rows.jsonl", _row())
    code, out, err = _record(capsys, lb, home, "--rows", rows, "--json")
    report = json.loads(out)
    assert code == 0 and err == "" and list(report) == RECORD_KEYS
    assert report["schema"] == "loopmath.flow-record/1" and report["flow"] == "lit-brief"
    assert report["digest"] == DIGEST and report["findings"] == [] and report["advisories"] == []
    assert [list(r) for r in report["recorded"]] == [RECORDED_KEYS]
    assert report["recorded"][0]["arm"] == "base" and report["recorded"][0]["input"] == "q07"
    assert report["recorded"][0]["import"] == "new" and report["recorded"][0]["ended_at_from"] == "run"
    assert report["fit"] == {"started": True, "pid": 0}


def test_a_row_that_names_no_flow_is_refused_by_name(capsys, tmp_path, lb, home, fits):
    plain = json.loads(PLAIN_ROWS.read_text(encoding="utf-8").splitlines()[0])
    assert "flow" not in plain
    rows = _rows(tmp_path / "rows.jsonl", _row(), plain)
    code, out, err = _record(capsys, lb, home, "--rows", rows)
    assert code == 1 and out == ""
    assert _refused(err) == [f"{rows}:2: {NO_FLOW}"]
    assert err.splitlines()[-1] == "nothing was recorded: 1 item refused"
    assert _store(home) == {} and fits == []


def test_the_row_adapt_refuses_is_stored_by_the_command_it_names(capsys, tmp_path, lb, home):
    rows = _rows(tmp_path / "rows.jsonl", _row())
    code = cli.main(["adapt", "harness-runs", rows, "--out", str(tmp_path / "ocp")])
    err = capsys.readouterr().err
    assert code == 1 and "give the flow to convert it (loopmath flow record FLOW --rows FILE)" in err
    code, out, err = _flow(capsys, "record", str(lb), "--rows", rows, "--home", str(home))
    assert code == 0 and out.startswith("recorded run_hr_") and len(Store(home).flow_runs("lit-brief")) == 1


def test_rows_keep_their_run_ids_slate_split_and_plan(capsys, tmp_path, lb, home):
    slate = {"id": "slt_q07", "members": ["run_lit-brief.q07:r1", "run_lit-brief.q07:r2"], "base_commit": "4f1c2a9",
             "isolated": True, "blinded": True}
    pair = [_row(rid, slate=copy.deepcopy(slate), flow={"id": "lit-brief", "version": 1, "split": "train",
                                                         "plan": "pln_lit-brief_0001"})
            for rid in slate["members"]]
    code, out, err = _record(capsys, lb, home, "--rows", _rows(tmp_path / "rows.jsonl", *pair))
    assert code == 0, err
    for rid in slate["members"]:
        run = Store(home).run_doc(rid)["run"]
        assert run["slate"]["id"] == "slt_q07"
        assert run["flow"] == {"id": "lit-brief", "version": 1, "digest": DIGEST, "arm": "base", "plan":
                               "pln_lit-brief_0001", "split": "train"}


def test_a_kept_run_id_at_the_cap_is_stored_and_one_past_it_refused(capsys, tmp_path, lb, home):
    at_cap, past = "run_" + "a" * (HR.MAX_KEPT_RUN_ID - 4), "run_" + "b" * (HR.MAX_KEPT_RUN_ID - 3)
    code, out, err = _record(capsys, lb, home, "--rows", _rows(tmp_path / "ok.jsonl", _row(at_cap)))
    assert code == 0 and Store(home).exists(at_cap)
    rows = _rows(tmp_path / "long.jsonl", _row(past))
    code, out, err = _record(capsys, lb, home, "--rows", rows)
    assert code == 1 and _refused(err) == [
        f"{rows}:1: id: run id {past!r} is 186 characters; a row naming a flow keeps a run id of at most 185 "
        "(MAX_KEPT_RUN_ID), so every id made from it fits 200"]


def test_two_rows_with_one_kept_run_id_are_refused(capsys, tmp_path, lb, home):
    rows = _rows(tmp_path / "rows.jsonl", _row("run_q07.one"), _row("run_q07.one", "q08"))
    code, out, err = _record(capsys, lb, home, "--rows", rows)
    assert code == 1 and _refused(err) == [
        f"{rows}:2: run run_q07.one is also given by {rows}:1; give each run once"]
    assert _store(home) == {}


# OCP documents ----------------------------------------------------------------------------------------------------

def test_a_document_is_recorded_once_then_reads_already_recorded(capsys, tmp_path, lb, home, fits):
    doc = _doc(lb)
    path = _write(tmp_path / "run.json", doc)
    before = Path(path).read_bytes()
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    rid = doc["run"]["id"]
    assert code == 0 and out.splitlines()[0] == f"recorded {rid} arm base input q07"
    stored = _store(home)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    assert code == 0 and out.splitlines() == [f"already recorded {rid} arm base input q07",
                                              "0 runs recorded, 1 already recorded",
                                              "refit: not started (no run was new)"]
    code, out, err = _record(capsys, lb, home, "--ocp", path, "--json")
    report = json.loads(out)
    assert report["recorded"] == [{"run": rid, "arm": "base", "input": "q07", "import": "unchanged",
                                   "ended_at_from": None}]
    assert report["fit"] == {"started": False, "reason": "no run was new"}
    assert _store(home) == stored and fits == [home] and Path(path).read_bytes() == before
    code = cli.main(["run", "import", path, "--home", str(home), "--json"])
    assert code == 0 and json.loads(capsys.readouterr().out)["import"] == "unchanged"


def test_a_simulated_pair_from_the_plan_is_stored_with_its_slate(capsys, tmp_path, home):
    folder = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, folder)
    assert _flow(capsys, "plan", str(folder), "--home", str(home))[0] == 0
    flow = flows.load(folder)
    truth = json.loads((EXAMPLE / "truth.json").read_text(encoding="utf-8"))
    docs = simulate.simulate(flow, arm="base", n=0, seed=5, truth=truth, specs=next_runs(flow, n=2, seed=1))
    files = [_write(tmp_path / f"r{i}.json", d) for i, d in enumerate(docs)]
    code, out, err = _record(capsys, folder, home, "--ocp", *files, "--allow-simulated")
    assert code == 0, err
    runs = [Store(home).run_doc(d["run"]["id"])["run"] for d in docs]
    assert len({r["slate"]["id"] for r in runs}) == 1
    for run, doc in zip(runs, docs, strict=True):
        assert run["flow"]["digest"] == flow.digest and run["flow"]["plan"] == flow.plan.id
        assert run["flow"]["split"] == doc["run"]["flow"]["split"]
        assert run["acceptance_rule"] == flow.flow.acceptance_rule


def _variants(folder: Path) -> dict[str, dict]:
    """Documents each refused by one of record's own checks, before strict and cross-file validation."""
    base = _doc(folder)
    out = {}
    for name, edit in {
        "another flow": lambda d: d["run"]["flow"].update(id="other-flow"),
        "another version": lambda d: d["run"]["flow"].update(version=2),
        "another digest": lambda d: d["run"]["flow"].update(digest="flw_000000000000"),
        "no digest": lambda d: d["run"]["flow"].pop("digest"),
        "an input not listed": lambda d: d["run"]["task"].update(id="q99"),
        "another rule": lambda d: d["run"]["acceptance_rule"]["score"].update(target=0.8),
        "no run.flow": lambda d: d["run"].pop("flow"),
        "an id the store refuses": lambda d: d["run"].update(id="_run"),
    }.items():
        doc = copy.deepcopy(base)
        edit(doc)
        out[name] = doc
    return out


REASONS = {
    "another flow": "run.flow names flow 'other-flow' version 1; the flow given is flow 'lit-brief' version 1 ({path})",
    "another version": ("run.flow names flow 'lit-brief' version 2; the flow given is flow 'lit-brief' version 1 "
                        "({path})"),
    "another digest": (f"run.flow.digest 'flw_000000000000' is not the flow's digest {DIGEST}: the run was made for "
                       "another revision of the flow"),
    "no digest": f"run.flow gives no digest; the flow's is {DIGEST}",
    "an input not listed": "task.id 'q99' is not among the flow's inputs.ids (20 ids)",
    "another rule": ("run.acceptance_rule (score 'quality') is not the flow's (score 'quality'); they differ in score; "
                     "a run of the flow is judged by the flow's rule"),
    "no run.flow": "an OCP '0.4' run with no run.flow object is no run of a flow; store it with loopmath run import",
    "an id the store refuses": ("run.id '_run' cannot be stored: a run id it starts with a letter or digit, has no "
                                "control characters and fits in 200 characters as a file name"),
}


@pytest.mark.parametrize("name", list(REASONS))
def test_each_check_refuses_by_name_and_writes_nothing(name, capsys, tmp_path, lb, home, fits):
    doc = _variants(lb)[name]
    path = _write(tmp_path / "run.json", doc)
    before = Path(path).read_bytes()
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    reason = REASONS[name].format(path=lb / "flow.ocp.json")
    assert code == 1 and out == "" and _refused(err) == [f"{path}: {reason}"]
    code, out, err = _record(capsys, lb, home, "--ocp", path, "--json")
    assert code == 1 and json.loads(out) == {"schema": "loopmath.flow-record/1", "flow": "lit-brief",
                                             "refused": [{"item": path, "run": doc["run"]["id"], "reason": reason}]}
    assert _store(home) == {} and fits == [] and Path(path).read_bytes() == before


def test_an_ocp_0_3_run_is_pointed_to_run_import(capsys, tmp_path, lb, home):
    doc = json.loads((ROOT / "spec" / "examples" / "v0.3" / "full-fields.ocp.json").read_text(encoding="utf-8"))
    path = _write(tmp_path / "run.json", doc)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    assert code == 1 and _refused(err) == [
        f"{path}: an OCP '0.3' run with no run.flow object is no run of a flow; store it with loopmath run import"]


def test_one_bad_item_stores_nothing_and_every_bad_item_is_named(capsys, tmp_path, lb, home, fits):
    variants = _variants(lb)
    good = _write(tmp_path / "good.json", _doc(lb, _row("q08-r1", "q08")))
    bad = [_write(tmp_path / "digest.json", variants["another digest"]),
           _write(tmp_path / "input.json", variants["an input not listed"])]
    missing = str(tmp_path / "missing.json")
    code, out, err = _record(capsys, lb, home, "--ocp", good, *bad, missing)
    assert code == 1 and out == ""
    assert [line.split(": ", 1)[0] for line in _refused(err)] == [missing, *bad]
    assert err.splitlines()[-1] == "nothing was recorded: 3 items refused"
    assert _store(home) == {} and fits == []


def test_dollars_no_float_holds_are_refused_by_name_before_the_good_run_is_stored(capsys, tmp_path, lb, home, fits):
    """The store refuses to finish a run whose attempt's dollars are an integer no float holds; record checks that
    on every item first, so the good run before it is not stored either."""
    good = _write(tmp_path / "good.json", _doc(lb, _row("q08-r1", "q08")))
    doc = _doc(lb)
    doc["attempts"][0]["cost"]["usd"] = 10 ** 400
    big = _write(tmp_path / "big.json", doc)
    reason = "attempts[0].cost.usd is past what a float holds, so the run cannot be recorded finished"
    code, out, err = _record(capsys, lb, home, "--ocp", good, big)
    assert code == 1 and out == "" and _refused(err) == [f"{big}: {reason}"]
    assert err.splitlines()[-1] == "nothing was recorded: 1 item refused"
    code, out, err = _record(capsys, lb, home, "--ocp", good, big, "--json")
    assert code == 1 and json.loads(out) == {"schema": "loopmath.flow-record/1", "flow": "lit-brief",
                                             "refused": [{"item": big, "run": doc["run"]["id"], "reason": reason}]}
    assert _store(home) == {} and fits == []


def test_a_document_given_twice_is_refused_even_when_identical(capsys, tmp_path, lb, home):
    doc = _doc(lb)
    one, two = _write(tmp_path / "a.json", doc), _write(tmp_path / "b.json", doc)
    code, out, err = _record(capsys, lb, home, "--ocp", one, two)
    assert code == 1 and _refused(err) == [f"{two}: run {doc['run']['id']} is also given by {one}; give each run once"]
    assert _store(home) == {}


# The acceptance rule --------------------------------------------------------------------------------------------

def test_a_document_with_no_rule_gets_the_flows_and_reads_unchanged_after(capsys, tmp_path, lb, home):
    doc = _doc(lb)
    del doc["run"]["acceptance_rule"]
    path = _write(tmp_path / "run.json", doc)
    before = Path(path).read_bytes()
    assert _record(capsys, lb, home, "--ocp", path)[0] == 0
    stored = Store(home).run_doc(doc["run"]["id"])
    assert stored["run"]["acceptance_rule"] == flows.load(lb).flow.acceptance_rule
    assert Path(path).read_bytes() == before
    snapshot = _store(home)
    assert _record(capsys, lb, home, "--ocp", path)[1].startswith("already recorded ")
    assert cli.main(["run", "import", path, "--home", str(home), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["import"] == "unchanged"
    assert _store(home) == snapshot


@pytest.mark.parametrize(("a", "b", "equal"), [
    (1, 1.0, True), ({"x": [1, 2.0]}, {"x": [1.0, 2]}, True), (True, 1, False), (False, 0.0, False),
    (True, True, True), ({"x": 1}, {"x": 1, "y": None}, False), ([1], [1, 1], False), ("1", 1, False),
    (None, None, True)])
def test_rules_compare_as_json_values(a, b, equal):
    assert record._json_equal(a, b) is equal and record._json_equal(b, a) is equal


def test_a_rule_with_an_unknown_member_names_it(capsys, tmp_path, lb, home):
    doc = _doc(lb)
    doc["run"]["acceptance_rule"]["note"] = "mine"
    path = _write(tmp_path / "run.json", doc)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    assert code == 1 and _refused(err) == [
        f"{path}: run.acceptance_rule (score 'quality') is not the flow's (score 'quality'); they differ in note; a "
        "run of the flow is judged by the flow's rule"]


# Arms: planned, unplanned, no plan, stale plan --------------------------------------------------------------------

def _own(folder: Path) -> dict:
    """A run of arm `unplanned` that named `opus-only`, with its own configuration from its stages."""
    doc = _doc(folder, _row(arm="opus-only"), unplanned=True)
    assert doc["run"]["flow"]["arm"] == "unplanned" and doc["run"]["configuration"]["id"] not in (BASE_ID, SONNET_ID)
    assert doc["run"]["flow"]["ext"] == {FLOW_EXT: {"named_arm": "opus-only"}}
    return doc


def _arm(doc: dict, arm: str | None, ext=None) -> dict:
    doc = copy.deepcopy(doc)
    if arm is None:
        del doc["run"]["flow"]["arm"]
    else:
        doc["run"]["flow"]["arm"] = arm
    if ext is None:
        doc["run"]["flow"].pop("ext", None)
    else:
        doc["run"]["flow"]["ext"] = ext
    return doc


def test_a_known_arm_with_its_configuration_is_recorded(capsys, tmp_path, lb, home):
    for name, doc in (("sonnet", _doc(lb, _row(arm="extract-sonnet"))), ("none", _arm(_doc(lb, _row("q07-r2")), None))):
        path = _write(tmp_path / f"{name}.json", doc)
        code, out, err = _record(capsys, lb, home, "--ocp", path, "--json")
        assert code == 0, err
        arm = "extract-sonnet" if name == "sonnet" else "base"
        assert json.loads(out)["recorded"][0]["arm"] == arm
        assert Store(home).run_doc(doc["run"]["id"])["run"]["flow"].get("arm", "base") == arm


@pytest.mark.parametrize("flag", [[], ["--allow-unplanned"]])
def test_an_arm_with_another_arms_configuration_is_refused_flag_or_not(flag, capsys, tmp_path, lb, home):
    path = _write(tmp_path / "run.json", _arm(_doc(lb), "extract-sonnet"))
    code, out, err = _record(capsys, lb, home, "--ocp", path, *flag)
    assert code == 1 and _refused(err) == [
        f"{path}: run.flow.arm is 'extract-sonnet', but run.configuration.id {BASE_ID!r} is arm 'base''s "
        "configuration; the arm and the configuration must agree"]
    assert _store(home) == {}


def _unplanned_cases(folder: Path) -> dict[str, tuple[dict, str, str | None]]:
    """(document, the refusal without the flag after the file name, the named_arm stored with it)."""
    own = _own(folder)
    cid = own["run"]["configuration"]["id"]
    plan = folder / "plan.json"
    return {
        "base, no arm's configuration": (
            _arm(own, "base"), f"run.configuration.id {cid!r} is no arm's configuration (arm 'base' has "
                               f"{BASE_ID!r}); {FLAG}", "base"),
        "no arm, no arm's configuration": (
            _arm(own, None), f"run.configuration.id {cid!r} is no arm's configuration (arm 'base' has "
                             f"{BASE_ID!r}); {FLAG}", None),
        "an arm the plan lacks": (
            _arm(own, "opus-only"), f"run.flow.arm 'opus-only' is not an arm of {plan} (arms: base, extract-sonnet); "
                                    f"{FLAG}", "opus-only"),
        "unplanned, its named arm kept": (own, f"run.flow.arm is 'unplanned' (it named arm 'opus-only'); {FLAG}",
                                          "opus-only"),
        "unplanned, none named": (_arm(own, "unplanned"), f"run.flow.arm is 'unplanned'; {FLAG}", None),
    }


@pytest.mark.parametrize("case", ["base, no arm's configuration", "no arm, no arm's configuration",
                                  "an arm the plan lacks", "unplanned, its named arm kept", "unplanned, none named"])
def test_an_unplanned_run_needs_the_flag_and_keeps_the_arm_it_named(case, capsys, tmp_path, lb, home):
    doc, reason, named = _unplanned_cases(lb)[case]
    path = _write(tmp_path / "run.json", doc)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    assert code == 1 and _refused(err) == [f"{path}: {reason}"] and _store(home) == {}
    code, out, err = _record(capsys, lb, home, "--ocp", path, "--allow-unplanned", "--json")
    assert code == 0, err
    assert json.loads(out)["recorded"][0]["arm"] == "unplanned"
    run = Store(home).run_doc(doc["run"]["id"])["run"]
    assert run["flow"]["arm"] == "unplanned" and run["configuration"] == doc["run"]["configuration"]
    assert run["flow"].get("ext") == ({FLOW_EXT: {"named_arm": named}} if named is not None else None)
    assert run["labels"] == doc["run"]["labels"]  # a document's labels are kept as written


def test_other_flow_ext_members_are_kept_beside_the_named_arm(capsys, tmp_path, lb, home):
    ext = {FLOW_EXT: {"note": [1, 2]}, "dev.example.x": {"y": True}}
    doc = _arm(_own(lb), "opus-only", copy.deepcopy(ext))
    path = _write(tmp_path / "run.json", doc)
    assert _record(capsys, lb, home, "--ocp", path, "--allow-unplanned")[0] == 0
    stored = Store(home).run_doc(doc["run"]["id"])["run"]["flow"]["ext"]
    assert stored == {FLOW_EXT: {"note": [1, 2], "named_arm": "opus-only"}, "dev.example.x": {"y": True}}


@pytest.mark.parametrize("held", ["haiku-only", None, 7])
def test_a_named_arm_that_differs_is_refused(held, capsys, tmp_path, lb, home, fits):
    doc = _arm(_own(lb), "opus-only", {FLOW_EXT: {"named_arm": held}})
    path = _write(tmp_path / "run.json", doc)
    code, out, err = _record(capsys, lb, home, "--ocp", path, "--allow-unplanned")
    assert code == 1 and _refused(err) == [
        f"{path}: run.flow.ext['dev.loopmath.flow'].named_arm is {held!r}, but the run names arm 'opus-only'"]
    assert _store(home) == {} and fits == []


def test_an_equal_named_arm_is_kept_with_the_rest_of_the_namespace(capsys, tmp_path, lb, home):
    ext = {FLOW_EXT: {"named_arm": "opus-only", "note": 1}, "dev.example.x": {}}
    doc = _arm(_own(lb), "opus-only", copy.deepcopy(ext))
    assert _record(capsys, lb, home, "--ocp", _write(tmp_path / "run.json", doc), "--allow-unplanned")[0] == 0
    assert Store(home).run_doc(doc["run"]["id"])["run"]["flow"]["ext"] == ext


NOT_OBJECTS = [None, 7, [], "kept"]


def _named(folder: Path, rid: str, ext) -> dict:
    """A run naming arm `opus-only`, which the plan lacks, with its own configuration and `run.flow.ext` as given."""
    doc = _doc(folder, _row(rid, arm="opus-only"), unplanned=True)
    doc["run"]["flow"]["arm"] = "opus-only"
    doc["run"]["flow"]["ext"] = ext
    return doc


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_a_flow_ext_that_is_no_object_is_refused_by_name_and_the_batch_writes_nothing(as_json, capsys, tmp_path, lb,
                                                                                       home, fits):
    good = _write(tmp_path / "good.json", _doc(lb, _row("q08-r1", "q08")))
    docs = [_named(lb, f"q07-r{n}", ext) for n, ext in enumerate(NOT_OBJECTS, 1)]
    bad = [_write(tmp_path / f"bad{n}.json", doc) for n, doc in enumerate(docs, 1)]
    given = {path: Path(path).read_bytes() for path in (good, *bad)}
    code, out, err = _record(capsys, lb, home, "--ocp", good, *bad, "--allow-unplanned", *(["--json"] * as_json))
    reasons = [f"run.flow.ext {ext!r} is not an object" for ext in NOT_OBJECTS]
    assert code == 1 and _refused(err) == [f"{path}: {reason}" for path, reason in zip(bad, reasons)]
    assert err.splitlines()[-1] == "nothing was recorded: 4 items refused"
    assert (json.loads(out) if as_json else out) == (
        {"schema": "loopmath.flow-record/1", "flow": "lit-brief",
         "refused": [{"item": path, "run": doc["run"]["id"], "reason": reason}
                     for path, doc, reason in zip(bad, docs, reasons)]} if as_json else "")
    assert _store(home) == {} and fits == [] and {path: Path(path).read_bytes() for path in given} == given


@pytest.mark.parametrize("value", NOT_OBJECTS)
def test_a_flow_namespace_that_is_no_object_is_refused_not_replaced_by_the_named_arm(value, capsys, tmp_path, lb, home,
                                                                                   fits):
    doc = _named(lb, "q07-r1", {FLOW_EXT: value, "dev.example.x": {"y": True}})
    path = _write(tmp_path / "run.json", doc)
    before = Path(path).read_bytes()
    reason = (f"run.flow.ext['dev.loopmath.flow'] {value!r} is not an object, so the arm the run named, 'opus-only', "
              "cannot be kept there as named_arm")
    code, out, err = _record(capsys, lb, home, "--ocp", path, "--allow-unplanned")
    assert code == 1 and out == "" and _refused(err) == [f"{path}: {reason}"]
    code, out, err = _record(capsys, lb, home, "--ocp", path, "--allow-unplanned", "--json")
    assert code == 1 and json.loads(out)["refused"] == [{"item": path, "run": doc["run"]["id"], "reason": reason}]
    assert _store(home) == {} and fits == [] and Path(path).read_bytes() == before


@pytest.mark.parametrize("value", NOT_OBJECTS)
def test_a_flow_namespace_that_record_does_not_write_to_is_kept_as_given(value, capsys, tmp_path, lb, home):
    """A known arm, and a run that names `unplanned` itself, keep no named arm, so the value is the producer's."""
    known = _doc(lb, _row("q08-r1", "q08"))
    known["run"]["flow"]["ext"] = {FLOW_EXT: value}
    plain = _named(lb, "q07-r1", {FLOW_EXT: value})
    plain["run"]["flow"]["arm"] = "unplanned"
    paths = [_write(tmp_path / "known.json", known), _write(tmp_path / "plain.json", plain)]
    code, out, err = _record(capsys, lb, home, "--ocp", *paths, "--allow-unplanned")
    assert code == 0, err
    for doc in (known, plain):
        assert Store(home).run_doc(doc["run"]["id"])["run"]["flow"]["ext"] == {FLOW_EXT: value}


def test_an_unplanned_run_with_a_known_configuration_is_refused(capsys, tmp_path, lb, home):
    path = _write(tmp_path / "run.json", _arm(_doc(lb), "unplanned"))
    code, out, err = _record(capsys, lb, home, "--ocp", path, "--allow-unplanned")
    assert code == 1 and _refused(err) == [
        f"{path}: run.flow.arm is 'unplanned', but run.configuration.id {BASE_ID!r} is arm 'base''s configuration; "
        "name arm 'base'"]


def test_with_no_plan_an_arm_needs_flow_plan_or_the_flag(capsys, tmp_path, lb, home):
    doc = _doc(lb, _row(arm="extract-sonnet"))
    (lb / "plan.json").unlink()
    path = _write(tmp_path / "run.json", doc)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    assert code == 1 and _refused(err) == [
        f"{path}: run.flow.arm is 'extract-sonnet', but there is no plan.json in {lb}; run loopmath flow plan, or "
        f"{FLAG}"]
    code, out, err = _record(capsys, lb, home, "--ocp", path, "--allow-unplanned")
    assert code == 0, err
    run = Store(home).run_doc(doc["run"]["id"])["run"]
    assert run["flow"]["arm"] == "unplanned" and run["flow"]["ext"] == {FLOW_EXT: {"named_arm": "extract-sonnet"}}


@pytest.mark.parametrize("flag", [[], ["--allow-unplanned"]])
def test_a_stale_plan_is_refused_naming_flow_plan_flag_or_not(flag, capsys, tmp_path, lb, home):
    docs = [_doc(lb, _row(arm="extract-sonnet")), _arm(_own(lb), "opus-only"), _own(lb)]
    plan = json.loads((lb / "plan.json").read_text(encoding="utf-8"))
    plan["flow"]["digest"] = "flw_000000000000"
    (lb / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    stale = (f"{lb / 'plan.json'} is stale: it was made for flow digest flw_000000000000, the flow's digest is "
             f"{DIGEST}; run loopmath flow plan")
    for n, doc in enumerate(docs):
        path = _write(tmp_path / f"run{n}.json", doc)
        code, out, err = _record(capsys, lb, home, "--ocp", path, *flag)
        assert code == 1 and _refused(err) == [f"{path}: {stale}"]
    rows = _rows(tmp_path / "rows.jsonl", _row(arm="opus-only"))
    code, out, err = _record(capsys, lb, home, "--rows", rows, *flag)
    assert code == 1 and _refused(err) == [f"{rows}:1: {stale}"]
    assert _store(home) == {}


def test_an_unplanned_row_needs_the_flag_and_runs_its_own_stages(capsys, tmp_path, lb, home):
    rows = _rows(tmp_path / "rows.jsonl", _row(arm="opus-only"))
    code, out, err = _record(capsys, lb, home, "--rows", rows)
    assert code == 1 and _refused(err) == [f"{rows}:1: run.flow.arm is 'unplanned' (it named arm 'opus-only'); {FLAG}"]
    code, out, err = _record(capsys, lb, home, "--rows", rows, "--allow-unplanned")
    assert code == 0, err
    doc = Store(home).run_doc(Store(home).flow_runs("lit-brief")[0]["run"])
    assert doc["run"]["flow"]["arm"] == "unplanned"
    assert doc["run"]["flow"]["ext"] == {FLOW_EXT: {"named_arm": "opus-only"}}
    assert doc["run"]["configuration"] == _own(lb)["run"]["configuration"]
    assert [a["node"] for a in doc["attempts"]] == ["scout", "extract", "synthesize", "critic"]


def test_an_unplanned_row_with_a_piece_the_flow_lacks_is_refused_flag_or_not(capsys, tmp_path, lb, home):
    row = _row(arm="extract-sonnet")
    row["stages"][1]["piece"] = "writer"
    rows = _rows(tmp_path / "rows.jsonl", row)
    for flag in ([], ["--allow-unplanned"]):
        code, out, err = _record(capsys, lb, home, "--rows", rows, *flag)
        assert code == 1 and _refused(err) == [
            f"{rows}:1: stages.1.piece: 'writer' is not a piece of flow 'lit-brief' arm 'extract-sonnet' (pieces: "
            "scout, extract, synthesize, critic)"]


# Simulated runs -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("marker", [{}, {"seed": 5}])
def test_a_simulated_document_needs_allow_simulated(marker, capsys, tmp_path, lb, home):
    doc = _doc(lb)
    doc["run"].setdefault("ext", {})[SIMULATE] = marker
    path = _write(tmp_path / "run.json", doc)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    assert code == 1 and _refused(err) == [
        f"{path}: run {doc['run']['id']} is simulated (run.ext['dev.loopmath.simulate']); --allow-simulated records "
        "simulated runs"]
    assert _store(home) == {}
    assert _record(capsys, lb, home, "--ocp", path, "--allow-simulated")[0] == 0
    assert Store(home).run_doc(doc["run"]["id"])["run"]["ext"][SIMULATE] == marker


@pytest.mark.parametrize("marker", ["yes", [1], None])
def test_a_marker_that_is_not_an_object_is_no_simulated_run(marker, capsys, tmp_path, lb, home):
    doc = _doc(lb)
    doc["run"].setdefault("ext", {})[SIMULATE] = marker
    code, out, err = _record(capsys, lb, home, "--ocp", _write(tmp_path / "run.json", doc))
    assert code == 0, err


def test_a_simulated_row_needs_allow_simulated(capsys, tmp_path, lb, home):
    rows = _rows(tmp_path / "rows.jsonl", _row(ext={SIMULATE: {"seed": 1}}))
    code, out, err = _record(capsys, lb, home, "--rows", rows)
    assert code == 1 and "is simulated (run.ext['dev.loopmath.simulate']); --allow-simulated records" in err
    assert _record(capsys, lb, home, "--rows", rows, "--allow-simulated")[0] == 0


# A run already in the store ---------------------------------------------------------------------------------------

def test_a_changed_copy_of_a_recorded_run_is_refused_and_the_stored_run_kept(capsys, tmp_path, lb, home):
    doc = _doc(lb)
    assert _record(capsys, lb, home, "--ocp", _write(tmp_path / "a.json", doc))[0] == 0
    stored = _store(home)
    doc["run"]["signals"][0]["value"] = 0.1
    path = _write(tmp_path / "b.json", doc)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    [line] = _refused(err)
    assert code == 1 and line.startswith(f"{path}: run {doc['run']['id']} is recorded with other content (")
    assert line.endswith("); flow record does not replace a recorded run")
    assert _store(home) == stored


def test_an_open_run_that_run_start_made_is_not_filled(capsys, tmp_path, lb, home):
    code = cli.main(["run", "start", "--type", "research", "--repo", "r", "--workflow", "solo", "--set",
                     "implement=codex:gpt-6-sol:low", "--source", "user_edit", "--home", str(home), "--json"])
    rid = json.loads(capsys.readouterr().out)["run"]
    assert code == 0
    doc = _doc(lb)
    doc["run"]["id"] = rid
    path = _write(tmp_path / "run.json", doc)
    stored = _store(home)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    assert code == 1 and _refused(err)[0].startswith(f"{path}: run {rid} is recorded with other content (")
    assert _store(home) == stored


def test_a_recorded_run_given_again_under_the_flow_renamed_is_refused_by_record_itself(capsys, tmp_path, lb, home):
    """The run id kept, `run.flow` the renamed folder's id and digest: record's checks pass the reference and its
    stored-content check refuses the run, naming run.flow.id, before the store's own refusal of another flow."""
    doc = _doc(lb)
    assert _record(capsys, lb, home, "--ocp", _write(tmp_path / "a.json", doc))[0] == 0
    stored = _store(home)
    renamed = tmp_path / "lit-brief-2"
    renamed.mkdir()
    flow = json.loads(FLOW_FILE.read_text(encoding="utf-8"))
    flow["flow"]["id"] = "lit-brief-2"
    flow["flow"]["digest"] = flow_digest(flow["flow"])
    (renamed / "flow.ocp.json").write_text(json.dumps(flow), encoding="utf-8")
    again = _doc(renamed, _row(flow={"id": "lit-brief-2", "version": 1}))
    again["run"]["id"] = rid = doc["run"]["id"]
    assert (again["run"]["flow"]["id"], again["run"]["flow"]["digest"]) == ("lit-brief-2", flow["flow"]["digest"])
    path = _write(tmp_path / "b.json", again)
    code, out, err = _record(capsys, renamed, home, "--ocp", path)
    [line] = _refused(err)
    assert code == 1 and line.startswith(f"{path}: run {rid} is recorded with other content (")
    assert "run.flow.id" in line.split(" (", 1)[1].split(")")[0].split(", ")
    code, out, err = _record(capsys, renamed, home, "--ocp", path, "--json")
    [refused] = json.loads(out)["refused"]
    assert code == 1 and (refused["item"], refused["run"]) == (path, rid)
    assert refused["reason"].startswith(f"run {rid} is recorded with other content (") and "another flow" not in err
    assert _store(home) == stored


def test_a_recorded_run_reads_unchanged_after_a_late_score(capsys, tmp_path, lb, home, fits):
    doc = _doc(lb)
    path = _write(tmp_path / "run.json", doc)
    assert _record(capsys, lb, home, "--ocp", path)[0] == 0
    code, out, err = _flow(capsys, "score", str(lb), "--run", doc["run"]["id"], "--judge", "person", "0.9",
                           "--home", str(home))
    assert code == 0 and "late signal" in out, err
    snapshot = _store(home)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    assert code == 0 and out.startswith("already recorded ") and _store(home) == snapshot
    assert cli.main(["run", "import", path, "--home", str(home), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["import"] == "unchanged" and _store(home) == snapshot


# The end, the inputs, the refit, a failed write -----------------------------------------------------------------

def test_the_end_comes_from_the_run_then_its_attempts_and_a_run_with_neither_is_refused(capsys, tmp_path, lb, home,
                                                                                         fits):
    docs = {"run": _doc(lb, _row("q07-a")), "attempts": _doc(lb, _row("q07-b")), "none": _doc(lb, _row("q07-c"))}
    assert docs["run"]["run"]["ended_at"] == AT  # the row's end, written by the converter
    del docs["attempts"]["run"]["ended_at"], docs["none"]["run"]["ended_at"]
    docs["attempts"]["attempts"][-1]["ended_at"] = "2026-11-03T11:05:00-08:00"
    files = [_write(tmp_path / f"{k}.json", d) for k, d in docs.items()]
    code, out, err = _record(capsys, lb, home, "--ocp", *files)  # good items before the one with no end
    assert code == 1 and out == "" and _store(home) == {} and fits == []
    assert _refused(err) == [f"{files[2]}: run {docs['none']['run']['id']} states no end (run.ended_at, or an "
                             "attempt's ended_at); flow record stores a run finished at its own end"]
    code, out, err = _record(capsys, lb, home, "--ocp", *files[:2], "--json")
    assert code == 0, err
    assert [r["ended_at_from"] for r in json.loads(out)["recorded"]] == ["run", "attempts"]
    store = Store(home)
    ends = [store.run_doc(d["run"]["id"])["run"]["ended_at"] for d in list(docs.values())[:2]]
    assert ends == [AT, "2026-11-03T11:05:00-08:00"]


def test_a_document_without_a_run_end_ends_at_its_latest_attempt_not_its_last(capsys, tmp_path, lb, home):
    doc = _doc(lb)
    del doc["run"]["ended_at"]
    for attempt, end in zip(doc["attempts"], ["11:01", "11:05", "11:03", "11:04"], strict=True):
        attempt["ended_at"] = f"2026-11-03T{end}:00-08:00"
    code, out, err = _record(capsys, lb, home, "--ocp", _write(tmp_path / "run.json", doc), "--json")
    assert code == 0 and json.loads(out)["recorded"][0]["ended_at_from"] == "attempts", err
    assert Store(home).run_doc(doc["run"]["id"])["run"]["ended_at"] == "2026-11-03T11:05:00-08:00"


def test_the_example_runs_and_the_spec_row_are_recorded_at_their_own_ends(capsys, tmp_path, lb, home):
    files = [ROOT / "spec" / "examples" / "v0.4" / f"lit-brief.run-{n}.ocp.json" for n in (1, 2)]
    code, out, err = _record(capsys, lb, home, "--ocp", *map(str, files), "--json")
    assert code == 0, err
    recorded = json.loads(out)["recorded"]
    assert [(r["arm"], r["input"], r["import"], r["ended_at_from"]) for r in recorded] == [
        ("base", "q07", "new", "run")] * 2
    for path, r in zip(files, recorded, strict=True):
        given = json.loads(path.read_text(encoding="utf-8"))["run"]
        assert r["run"] == given["id"] and Store(home).run_doc(r["run"])["run"]["ended_at"] == given["ended_at"]
    text = (ROOT / "spec" / "harness-runs.md").read_text(encoding="utf-8")
    section = text[text.index("## 10. Rows that name a flow"):]
    row = json.loads(section.split("```json\n", 1)[1].split("```", 1)[0])
    other = tmp_path / "other"
    code, out, err = _record(capsys, lb, other, "--rows", _rows(tmp_path / "rows.jsonl", row), "--json")
    assert code == 0, err
    [r] = json.loads(out)["recorded"]
    assert r["ended_at_from"] == "run" and Store(other).run_doc(r["run"])["run"]["ended_at"] == row["ended_at"]


LAST = "2026-11-03T11:05:30-08:00"


def _ends(row: dict, last: str | None = None, earlier: str | None = None, own: bool = True) -> dict:
    """`row` with its last stage's end, an earlier stage's end, and its own end kept or dropped."""
    if last is not None:
        row["stages"][-1]["ended_at"] = last
    if earlier is not None:
        row["stages"][0]["ended_at"] = earlier
    if not own:
        del row["ended_at"]
    return row


@pytest.mark.parametrize(("case", "end"), [("the row's own over its last stage's", AT), ("its last stage's", LAST)])
def test_a_row_is_stored_finished_at_its_own_end(case, end, capsys, tmp_path, lb, home):
    row = _ends(_row(), last=LAST, own=case != "its last stage's")
    code, out, err = _record(capsys, lb, home, "--rows", _rows(tmp_path / "rows.jsonl", row), "--json")
    assert code == 0, err
    [recorded] = json.loads(out)["recorded"]
    assert recorded["ended_at_from"] == "run"
    run = Store(home).run_doc(recorded["run"])["run"]
    assert run["ended_at"] == end and run["started_at"] == "2026-11-03T11:00:00-08:00"


@pytest.mark.parametrize("earlier", [None, "2026-11-03T11:02:00-08:00"], ids=["no end", "an earlier stage's only"])
@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_a_row_that_states_no_end_is_refused_and_the_good_row_before_it_is_not_stored(earlier, as_json, capsys,
                                                                                      tmp_path, lb, home, fits):
    bad = _ends(_row("q08-r1", "q08"), earlier=earlier, own=False)
    rows = _rows(tmp_path / "rows.jsonl", _row(), bad)
    code, out, err = _record(capsys, lb, home, "--rows", rows, *(["--json"] * as_json))
    reason = ("row q08-r1 states no end (ended_at, or its last stage's ended_at); flow record stores a run finished "
              "at its own end")
    assert code == 1 and _refused(err) == [f"{rows}:2: {reason}"]
    assert err.splitlines()[-1] == "nothing was recorded: 1 item refused"
    assert (json.loads(out) if as_json else out) == (
        {"schema": "loopmath.flow-record/1", "flow": "lit-brief", "refused": [{"item": f"{rows}:2", "run": "q08-r1",
                                                                                "reason": reason}]}
        if as_json else "")
    assert _store(home) == {} and fits == []


def test_a_revert_a_month_after_a_rows_end_is_read_against_that_end(capsys, tmp_path, lb, home):
    """A September row recorded in November is stored with the row's end, and a revert 30 days on is read against
    it (z 1.0); against a November end, a time of recording, the same signals read as a late revert (z 0.0)."""
    row = _row(started_at="2026-09-01T11:00:00-08:00", ended_at="2026-09-01T11:06:00-08:00")
    for judge in row["outcome"]["judges"]:
        judge["observed_at"] = judge["observed_at"].replace("2026-11-03", "2026-09-01")
    code, out, err = _record(capsys, lb, home, "--rows", _rows(tmp_path / "rows.jsonl", row), "--json")
    assert code == 0, err
    doc = Store(home).run_doc(json.loads(out)["recorded"][0]["run"])
    assert doc["run"]["ended_at"] == "2026-09-01T11:06:00-08:00"
    source = {"kind": "orchestrator", "ref": "test"}
    doc["run"]["signals"] += [
        {"id": "sig_v", "kind": "verdict", "name": "claims-check", "value": "pass", "observed_at":
         "2026-09-01T11:09:00-08:00", "tier": "reported", "source": source},
        {"id": "sig_e", "kind": "event", "name": "revert", "value": "revert",
         "observed_at": "2026-10-01T11:00:00-08:00", "tier": "reported", "source": source}]
    now = datetime.fromisoformat("2026-12-01T00:00:00-08:00")
    evidence = outcome_evidence(doc, stored_rule(doc), now=now)
    assert (evidence.z, list(evidence.reasons)) == (1.0, [])
    doc["run"]["ended_at"] = "2026-11-05T06:54:51Z"
    evidence = outcome_evidence(doc, stored_rule(doc), now=now)
    assert (evidence.z, list(evidence.reasons)) == (0.0, ["late:revert"])


# run.flow.version and a stale plan's arms -------------------------------------------------------------------------

def test_a_flow_version_true_is_refused_in_records_words_and_1_0_is_recorded_as_given(capsys, tmp_path, lb, home,
                                                                                      fits):
    true, real = _doc(lb), _doc(lb, _row("q08-r1", "q08"))
    true["run"]["flow"]["version"], real["run"]["flow"]["version"] = True, 1.0
    path = _write(tmp_path / "true.json", true)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    assert code == 1 and out == "" and _refused(err) == [
        f"{path}: run.flow names flow 'lit-brief' version True; the flow given is flow 'lit-brief' version 1 "
        f"({lb / 'flow.ocp.json'})"]
    assert _store(home) == {} and fits == []
    code, out, err = _record(capsys, lb, home, "--ocp", _write(tmp_path / "real.json", real))
    assert code == 0, err
    version = Store(home).run_doc(real["run"]["id"])["run"]["flow"]["version"]
    assert version == 1 and type(version) is float


def test_under_a_stale_plan_a_base_run_with_a_stale_arms_configuration_is_no_arms(capsys, tmp_path, lb, home):
    """A stale plan's arms are not known configurations: the run is no arm's, recorded `unplanned` with the flag."""
    doc = _arm(_doc(lb, _row(arm="extract-sonnet")), "base")
    assert doc["run"]["configuration"]["id"] == SONNET_ID
    plan = json.loads((lb / "plan.json").read_text(encoding="utf-8"))
    plan["flow"]["digest"] = "flw_000000000000"
    (lb / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    path = _write(tmp_path / "run.json", doc)
    code, out, err = _record(capsys, lb, home, "--ocp", path)
    assert code == 1 and _refused(err) == [
        f"{path}: run.configuration.id {SONNET_ID!r} is no arm's configuration (arm 'base' has {BASE_ID!r}); {FLAG}"]
    assert _store(home) == {}
    code, out, err = _record(capsys, lb, home, "--ocp", path, "--allow-unplanned", "--json")
    assert code == 0 and json.loads(out)["recorded"][0]["arm"] == "unplanned", err
    run = Store(home).run_doc(doc["run"]["id"])["run"]
    assert run["flow"]["arm"] == "unplanned" and run["flow"]["ext"] == {FLOW_EXT: {"named_arm": "base"}}


def test_with_no_inputs_listed_the_input_is_recorded_unchecked(capsys, tmp_path, home):
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    flow = json.loads(FLOW_FILE.read_text(encoding="utf-8"))
    del flow["flow"]["inputs"]["ids"], flow["flow"]["inputs"]["holdout"]
    flow["flow"]["digest"] = flow_digest(flow["flow"])
    (folder / "flow.ocp.json").write_text(json.dumps(flow), encoding="utf-8")
    doc = _doc(folder, _row(task="q99"))
    path = _write(tmp_path / "run.json", doc)
    code, out, err = _record(capsys, folder, home, "--ocp", path)
    rid = doc["run"]["id"]
    assert code == 0 and f"{rid}: input q99 not checked: the flow lists no inputs.ids" in out.splitlines()
    code, out, err = _record(capsys, folder, home, "--ocp", path, "--json")
    assert json.loads(out)["advisories"] == [{"run": rid, "input": "q99", "advisory": "inputs-not-listed"}]


def test_one_refit_for_a_batch_with_a_new_run(capsys, tmp_path, lb, home, fits):
    first = _write(tmp_path / "a.json", _doc(lb))
    assert _record(capsys, lb, home, "--ocp", first)[0] == 0 and fits == [home]
    second = _write(tmp_path / "b.json", _doc(lb, _row("q08-r1", "q08")))
    code, out, err = _record(capsys, lb, home, "--ocp", first, second, "--json")
    assert code == 0 and [r["import"] for r in json.loads(out)["recorded"]] == ["unchanged", "new"]
    assert fits == [home, home]


def test_a_failed_write_names_the_run_it_stopped_at_and_those_stored(capsys, tmp_path, lb, home, monkeypatch):
    docs = [_doc(lb, _row(f"q0{n}-r1", f"q0{n}")) for n in (1, 2, 3)]
    files = [_write(tmp_path / f"r{n}.json", d) for n, d in enumerate(docs)]
    real, calls = Store.import_run, []

    def import_run(self, doc, **kw):
        calls.append(doc["run"]["id"])
        if len(calls) == 2:
            raise OSError("no space left on device")
        return real(self, doc, **kw)

    monkeypatch.setattr(Store, "import_run", import_run)
    code, out, err = _record(capsys, lb, home, "--ocp", *files)
    first, second = docs[0]["run"]["id"], docs[1]["run"]["id"]
    assert code == 1 and out == "" and _refused(err) == [
        f"stopped at run {second} (no space left on device); stored before it: {first}"]
    assert [Store(home).exists(d["run"]["id"]) for d in docs] == [True, False, False]


def test_a_failed_check_at_the_write_keeps_its_words(tmp_path, lb, home, monkeypatch):
    from loopmath.store.home import StoreError, ValidationFailed

    path = _write(tmp_path / "run.json", _doc(lb))
    store = Store(home)
    ready, _ = record.prepare(flows.load(lb), store, files=[path])

    def import_run(self, doc, **kw):
        raise ValidationFailed({"errors": [{"code": "E000", "message": "refused"}]})

    monkeypatch.setattr(Store, "import_run", import_run)
    with pytest.raises(StoreError) as caught:
        record.commit(store, ready)
    assert str(caught.value) == (f"stopped at run {ready[0]['run']} (strict OCP check failed: E000 refused); stored "
                                 "before it: none")


def test_a_run_recorded_with_other_content_while_record_ran_stops_it_before_any_write(tmp_path, lb, home):
    raced = _doc(lb)
    files = [_write(tmp_path / "new.json", _doc(lb, _row("q08-r1", "q08"))), _write(tmp_path / "raced.json", raced)]
    store = Store(home)
    ready, _ = record.prepare(flows.load(lb), store, files=files)
    other = copy.deepcopy(raced)
    other["run"]["signals"][0]["value"] = 0.1
    store.import_run(other)
    snapshot = _store(home)
    with pytest.raises(record.Refused) as caught:
        record.commit(store, ready)
    [item] = caught.value.refused
    rid = raced["run"]["id"]
    assert item["item"] == files[1] and item["run"] == rid
    assert item["reason"].startswith(f"run {rid} was recorded with other content while this ran (")
    assert item["reason"].endswith("); nothing was written")
    assert _store(home) == snapshot


# A row's flow digest ----------------------------------------------------------------------------------------------

OTHER = "flw_000000000000"


def _revision(folder: Path) -> str:
    return (f"flow.digest: this row names flow 'lit-brief' version 1 digest {OTHER!r}, but the flow given has digest "
            f"{DIGEST} ({folder / 'flow.ocp.json'}): the row was made for another revision of the flow")


def _hashes(*folders: Path) -> dict[str, str]:
    """The sha256 of every file under `folders`, by path."""
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for folder in folders for p in sorted(folder.rglob("*"))
            if p.is_file()}


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_a_row_stating_the_flows_digest_is_recorded(as_json, capsys, tmp_path, lb, home, fits):
    rows = _rows(tmp_path / "rows.jsonl", _row(flow={"id": "lit-brief", "version": 1, "digest": DIGEST}))
    code, out, err = _record(capsys, lb, home, "--rows", rows, *(["--json"] * as_json))
    assert code == 0 and err == "" and fits == [home]
    [rid] = [r["run"] for r in Store(home).flow_runs("lit-brief")]
    assert Store(home).run_doc(rid)["run"]["flow"]["digest"] == DIGEST
    assert (json.loads(out)["recorded"][0]["run"] if as_json else out.splitlines()[0]) == (
        rid if as_json else f"recorded {rid} arm base input q07")


def _digest_cases() -> dict[str, tuple[list[dict], list[str]]]:
    """Rows files whose last row states another digest, by case: the rows, then the flags given."""
    other = {"id": "lit-brief", "version": 1, "digest": OTHER}
    return {"another digest": ([_row(flow=other)], []),
            "after a good row": ([_row(), _row("q08-r1", "q08", flow=other)], []),
            "unplanned": ([_row(arm="opus-only", flow=other)], ["--allow-unplanned"])}


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize("case", list(_digest_cases()))
def test_a_row_stating_another_digest_is_refused_and_no_file_changes(case, as_json, capsys, tmp_path, lb, home, fits):
    """The store already holds a run. The refusal leaves each of its files and each file of the flow's folder,
    plan.json with them, as they were, stores no row of the file and starts no refit."""
    code, out, err = _record(capsys, lb, home, "--rows", _rows(tmp_path / "first.jsonl", _row("q09-r1", "q09")))
    assert code == 0 and fits == [home]
    first = [r["run"] for r in Store(home).flow_runs("lit-brief")]
    before = _hashes(home, lb)
    assert str(lb / "plan.json") in before and str(lb / "flow.ocp.json") in before
    given, flags = _digest_cases()[case]
    rows = _rows(tmp_path / "rows.jsonl", *given)
    code, out, err = _record(capsys, lb, home, "--rows", rows, *flags, *(["--json"] * as_json))
    item, reason = f"{rows}:{len(given)}", _revision(lb)
    assert code == 1 and _refused(err) == [f"{item}: {reason}"]
    assert err.splitlines()[-1] == "nothing was recorded: 1 item refused"
    assert (json.loads(out) if as_json else out) == (
        {"schema": "loopmath.flow-record/1", "flow": "lit-brief",
         "refused": [{"item": item, "run": given[-1]["id"], "reason": reason}]} if as_json else "")
    assert _hashes(home, lb) == before and fits == [home]
    assert [r["run"] for r in Store(home).flow_runs("lit-brief")] == first
