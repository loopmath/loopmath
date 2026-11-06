"""What `flow record` and `flow score` do to plan.json after their store write, as `plan` in their reports
(`WRITER_PLAN_KEYS`, null with no plan.json): the plan brought up to date and saved when it changed; a plan that is
not this revision's, or a pending score, skipped by name with plan.json left as it is; a store that cannot be read
again, a refusal of `advance` or plan.json that cannot be written, exit 1 with what was stored kept and shown first.
Also `flow plan` and `flow next` in a folder that cannot be written, `flow plan --replace` when the new plan cannot
be written once the old one is kept, and a replaced score as B's fit reads it back from the store. The merge
check's runs are simulated with seed 1 from the example's truth with no merge loss: 24 pairs adopt."""

from __future__ import annotations

import copy
import errno
import json
import os
import shutil
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.flows import commands, judges, load, simulate
from loopmath.flows import plan as plan_mod
from loopmath.flows.commands import PENDING, WRITER_PLAN_KEYS, _usable_plan
from loopmath.flows.folder import save_plan
from loopmath.flows.next import next_runs
from loopmath.store import Store, fitjob
from loopmath.store.home import StoreError
from tests.flows.test_flow_score import _doc as _row_doc

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
TRUTH = json.loads((EXAMPLE / "truth.json").read_text(encoding="utf-8"))
ARM = "merge-extract-synthesize"  # the merge check, its experiment and its arm


@pytest.fixture(autouse=True)
def no_fit_job(monkeypatch):
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})


@pytest.fixture
def lb(tmp_path, capsys) -> Path:
    """The example flow with the plan `flow plan` makes."""
    folder = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, folder)
    assert _flow(capsys, "plan", str(folder), "--home", str(tmp_path / "home"))[0] == 0
    return folder


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def locked():
    """Makes a folder 0555 for the test, and writable again after it."""
    if os.geteuid() == 0:
        pytest.skip("root writes into a 0555 folder")
    held = []

    def lock(folder: Path) -> None:
        held.append((folder, folder.stat().st_mode))
        folder.chmod(0o555)

    yield lock
    for folder, mode in held:
        folder.chmod(mode)


def _flow(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["flow", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _strict(text: str):
    def refuse(name):
        raise AssertionError(f"not standard JSON: {name}")

    return json.loads(text, parse_constant=refuse)


def _bytes(root: Path) -> dict[str, bytes]:
    """Every file under `root` but a lock, by path."""
    if not root.exists():
        return {}
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "lock"}


def _merge(folder: Path, last: int, first: int = 1) -> list[dict]:
    """The merge check's pairs `first` to `last`."""
    flow = load(folder)
    truth = copy.deepcopy(TRUTH)
    for effect in truth["quality"]["effects"]:
        if effect["edit"]["kind"] == "merge":
            effect["quality"] = 0.0
    specs = [s for s in next_runs(flow, n=1000, seed=1) if s.experiment == ARM
             and first <= int(s.run_id.split(".")[-2]) <= last]
    return simulate.simulate(flow, arm="base", n=0, seed=1, truth=truth, specs=specs)


def _unscored(doc: dict) -> dict:
    doc["run"]["signals"] = [s for s in doc["run"]["signals"] if s.get("kind") != "score"]
    return doc


def _stored(home: Path, docs: list[dict], open_run: str | None = None) -> None:
    store = Store(home)
    for doc in docs:
        store.import_run(doc, finished=doc["run"]["id"] != open_run)


def _paths(tmp_path: Path, docs: list[dict], tag: str) -> list[str]:
    folder = tmp_path / tag
    folder.mkdir()
    for doc in docs:
        (folder / f"{doc['run']['id']}.json").write_text(json.dumps(doc), encoding="utf-8")
    return [str(folder / f"{doc['run']['id']}.json") for doc in docs]


def _record(capsys, folder: Path, home: Path, paths: list[str], *more: str) -> tuple[int, str, str]:
    return _flow(capsys, "record", str(folder), "--ocp", *paths, "--allow-simulated", "--home", str(home), *more)


def _score(capsys, folder: Path, home: Path, run: str, *more: str, judge: str = "j-claude",
           value: str = "0.6") -> tuple[int, str, str]:
    return _flow(capsys, "score", str(folder), "--run", run, "--judge", judge, value, "--home", str(home), *more)


def _advanced(folder: Path, home: Path):
    """The folder's plan as `advance` brings it up to date on the store's runs (pending signals merged)."""
    flow, store = load(folder), Store(home)
    docs = [store.run_doc(row["run"]) for row in store.flow_runs(flow.flow.id)]
    quality, fit = plan_mod.quality_and_fit(flow.flow, plan_mod.flow_runs(flow, docs), None, None)
    return plan_mod.advance(flow, docs, quality=quality, fit=fit)


def _state(plan) -> tuple[str | None, str | None]:
    [exp] = [e for e in plan.experiments if e.id == ARM]
    return exp.state, exp.verdict


def _phase(folder: Path, **given) -> dict:
    return {"id": load(folder).plan.id, "advanced": False, "skipped": None, "problem": None, **given}


def _last_pair(lb: Path, home: Path, tmp_path: Path, verb: str, open_last: bool = False) -> tuple[list[str], list]:
    """The store and plan just before the merge check's last readable pair, the argv that makes it and the runs it
    writes to: `record` stores pairs 1 to 23 and records pair 24; `score` stores all 24 pairs, run 024.b with no
    score (open when `open_last`), and scores it. The plan is brought up to date and saved first."""
    docs = _merge(lb, 24)
    last = next(d for d in docs if d["run"]["id"].endswith(".024.b"))
    if verb == "record":
        _stored(home, [d for d in docs if ".024." not in d["run"]["id"]])
        argv = ["record", str(lb), "--ocp", *_paths(tmp_path, [d for d in docs if ".024." in d["run"]["id"]],
                                                    "last"), "--allow-simulated"]
    else:
        _unscored(last)
        _stored(home, docs, open_run=last["run"]["id"] if open_last else None)
        argv = ["score", str(lb), "--run", last["run"]["id"], "--judge", "j-claude", "0.6"]
    save_plan(lb, _advanced(lb, home))
    assert _state(load(lb).plan) == ("running", "more")
    made = [d["run"]["id"] for d in docs if ".024." in d["run"]["id"]] if verb == "record" else [last["run"]["id"]]
    return argv + ["--home", str(home)], made


# The plan brought up to date --------------------------------------------------------------------------------------

@pytest.mark.parametrize("verb", ["record", "score"])
@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_the_last_pair_recorded_or_scored_late_writes_the_checks_verdict(as_json, verb, capsys, lb, home, tmp_path):
    argv, _ = _last_pair(lb, home, tmp_path, verb)
    phase = _phase(lb, advanced=True)
    code, out, err = _flow(capsys, *argv, *(["--json"] * as_json))
    assert code == 0
    assert _state(load(lb).plan) == ("done", "adopt") and load(lb).plan == _advanced(lb, home)
    line = f"{lb / 'plan.json'} brought up to date"
    if as_json:
        report = _strict(out)
        assert list(report["plan"]) == list(WRITER_PLAN_KEYS) and report["plan"] == phase and err == line + "\n"
        assert verb == "record" or report["state"] == "late"
    else:
        assert out.splitlines()[-1] == line and err == ""


def test_an_unchanged_re_record_brings_a_plan_behind_the_store_up_to_date_and_then_changes_nothing(capsys, lb, home,
                                                                                                    tmp_path):
    paths = _paths(tmp_path, _merge(lb, 24), "all")
    behind = (lb / "plan.json").read_bytes()
    assert _record(capsys, lb, home, paths)[0] == 0
    (lb / "plan.json").write_bytes(behind)  # as a plan phase that failed leaves it
    assert _advanced(lb, home) != load(lb).plan
    paths = paths[-2:]
    phase = _phase(lb, advanced=True)
    code, out, err = _record(capsys, lb, home, paths, "--json")
    report = _strict(out)
    assert code == 0 and [r["import"] for r in report["recorded"]] == ["unchanged", "unchanged"]
    assert report["plan"] == phase and err == f"{lb / 'plan.json'} brought up to date\n"
    assert _state(load(lb).plan) == ("done", "adopt")
    held = (lb / "plan.json").read_bytes()
    code, out, err = _record(capsys, lb, home, paths, "--json")
    assert code == 0 and _strict(out)["plan"] == _phase(lb) and err == ""
    code, out, err = _record(capsys, lb, home, paths)
    assert code == 0 and err == "" and "0 runs recorded, 2 already recorded" in out.splitlines()
    assert "plan.json" not in out and (lb / "plan.json").read_bytes() == held


# Skipped ----------------------------------------------------------------------------------------------------------

def test_with_no_plan_json_the_writers_report_no_plan_and_print_no_plan_line(capsys, lb, home, tmp_path):
    (lb / "plan.json").unlink()
    doc = _row_doc(lb)
    [path] = _paths(tmp_path, [doc], "row")
    code, out, err = _record(capsys, lb, home, [path], "--json")
    assert code == 0 and err == "" and _strict(out)["plan"] is None
    code, out, err = _score(capsys, lb, home, doc["run"]["id"], "--json", judge="person", value="0.7")
    assert code == 0 and err == "" and _strict(out)["plan"] is None and _strict(out)["state"] == "late"
    code, out, err = _score(capsys, lb, home, doc["run"]["id"], "--replace", value="0.9")
    assert code == 0 and err == "" and "plan" not in out
    assert not (lb / "plan.json").exists()


@pytest.mark.parametrize("verb", ["record", "score"])
@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_a_stale_plan_is_skipped_by_name_and_left_as_it_is(as_json, verb, capsys, lb, home, tmp_path):
    plan = lb / "plan.json"
    held = json.loads(plan.read_text(encoding="utf-8"))
    held["flow"]["digest"] = "flw_000000000000"
    plan.write_text(json.dumps(held), encoding="utf-8")
    before = plan.read_bytes()
    with pytest.raises(ValueError) as caught:
        _usable_plan(load(lb))
    words = str(caught.value)
    doc = _row_doc(lb)
    [path] = _paths(tmp_path, [doc], "row")
    more = ["--json"] * as_json
    if verb == "score":
        assert _record(capsys, lb, home, [path])[0] == 0
        code, out, err = _score(capsys, lb, home, doc["run"]["id"], *more, judge="person", value="0.7")
    else:
        code, out, err = _record(capsys, lb, home, [path], *more)
    assert code == 0 and plan.read_bytes() == before
    if as_json:
        assert _strict(out)["plan"] == {"id": held["id"], "advanced": False, "skipped": words, "problem": None}
        assert err == words + "\n"
    else:
        assert out.splitlines()[-1] == words and err == ""


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_a_pending_score_leaves_the_plan_though_it_completes_the_last_pair(as_json, capsys, lb, home, tmp_path):
    argv, _ = _last_pair(lb, home, tmp_path, "score", open_last=True)
    before = (lb / "plan.json").read_bytes()
    phase = _phase(lb, skipped=PENDING)
    code, out, err = _flow(capsys, *argv, *(["--json"] * as_json))
    assert code == 0 and (lb / "plan.json").read_bytes() == before
    moved = _advanced(lb, home)  # the control: advance on these documents would finish the check
    assert moved != load(lb).plan and _state(moved)[0] == "done" and _state(load(lb).plan) == ("running", "more")
    if as_json:
        report = _strict(out)
        assert report["state"] == "pending" and report["plan"] == phase and err == PENDING + "\n"
    else:
        assert out.splitlines()[-1] == PENDING and err == ""


def test_a_refused_record_runs_no_plan_phase(capsys, lb, home, tmp_path):
    before = _bytes(lb)
    paths = _paths(tmp_path, _merge(lb, 1), "simulated")
    code, out, err = _flow(capsys, "record", str(lb), "--ocp", *paths, "--home", str(home), "--json")
    report = _strict(out)
    assert code == 1 and report["refused"] and "plan" not in report
    assert err.endswith("nothing was recorded: 2 items refused\n") and _bytes(lb) == before


# Failures after the store write -----------------------------------------------------------------------------------

def _raise(exc: Exception):
    def raising(*args, **kwargs):
        raise exc

    return raising


FAILURES = {
    "store": (lambda mp: mp.setattr(commands, "_store_docs", _raise(StoreError("the index cannot be read"))),
              "plan.json not brought up to date: the store could not be read again: the index cannot be read; "
              "loopmath flow next brings it up to date once the store can be read"),
    "advance": (lambda mp: mp.setattr(plan_mod, "advance", _raise(ValueError("advance refused"))),
                "plan.json not brought up to date: advance refused"),
    "save": (lambda mp: mp.setattr(commands, "save_plan", _raise(OSError(errno.EROFS, "Read-only file system"))),
             "plan.json not brought up to date: {plan} cannot be written: Read-only file system; loopmath flow next "
             "brings it up to date once the folder can be written"),
}


def _kept(capsys, lb, home, verb, as_json, code, out, err, problem, ids, before):
    """Exit 1 with `problem`, what was stored kept and shown first, plan.json's bytes as they were."""
    assert code == 1 and err == f"error: {problem}\n" and (lb / "plan.json").read_bytes() == before
    assert "nothing was" not in out + err
    store = Store(home)
    if verb == "record":
        assert all(store.run_doc(i)["run"]["id"] == i for i in ids)
    else:
        assert [s["value"] for s in store.run_doc(ids[-1])["run"]["signals"] if s.get("kind") == "score"] == [0.6]
    if as_json:
        report = _strict(out)
        assert report["plan"] == _phase(lb, problem=problem)
        if verb == "record":
            assert report["schema"] == "loopmath.flow-record/1" and [r["run"] for r in report["recorded"]] == ids
        else:
            assert report["schema"] == "loopmath.flow-score/1" and report["state"] == "late"
    else:
        lines = out.splitlines()
        if verb == "record":
            assert [line.split(" arm ")[0] for line in lines[:2]] == [f"recorded {i}" for i in ids]
            assert "2 runs recorded, 0 already recorded" in lines
        else:
            assert lines[0].startswith(f"scored run {ids[-1]} judge j-claude value 0.6 (signal ")


@pytest.mark.parametrize("verb", ["record", "score"])
@pytest.mark.parametrize("failure", list(FAILURES))
@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_a_failure_after_the_store_write_keeps_what_was_stored_and_exits_1(as_json, failure, verb, capsys, lb, home,
                                                                           tmp_path, monkeypatch):
    argv, ids = _last_pair(lb, home, tmp_path, verb)
    before = (lb / "plan.json").read_bytes()
    apply, problem = FAILURES[failure]
    apply(monkeypatch)
    code, out, err = _flow(capsys, *argv, *(["--json"] * as_json))
    _kept(capsys, lb, home, verb, as_json, code, out, err, problem.format(plan=lb / "plan.json"), ids, before)


# A folder that cannot be written ----------------------------------------------------------------------------------

def test_plan_and_next_in_a_folder_that_cannot_be_written_refuse_by_name_and_write_nothing(capsys, tmp_path, lb,
                                                                                            home, locked):
    fresh = tmp_path / "fresh"
    shutil.copytree(EXAMPLE, fresh)
    before = _bytes(fresh)
    locked(fresh)
    code, out, err = _flow(capsys, "plan", str(fresh), "--home", str(home))
    assert code == 1
    assert err == f"error: {fresh / 'plan.json'} cannot be written: Permission denied\nnothing was written\n"
    assert "Traceback" not in out and _bytes(fresh) == before
    _stored(home, _merge(lb, 2))
    assert _advanced(lb, home) != load(lb).plan
    before = _bytes(lb)
    locked(lb)
    code, out, err = _flow(capsys, "next", str(lb), "--home", str(home))
    assert code == 1 and out == ""
    assert err == f"error: {lb / 'plan.json'} cannot be written: Permission denied\nnothing was written\n"
    assert _bytes(lb) == before


@pytest.mark.parametrize("verb", ["record", "score"])
def test_record_and_score_in_a_folder_that_cannot_be_written_keep_what_was_stored(verb, capsys, lb, home, tmp_path,
                                                                                  locked):
    argv, ids = _last_pair(lb, home, tmp_path, verb)
    before, held = (lb / "plan.json").read_bytes(), _bytes(lb)
    locked(lb)
    code, out, err = _flow(capsys, *argv, "--json")
    problem = (f"plan.json not brought up to date: {lb / 'plan.json'} cannot be written: Permission denied; loopmath "
               "flow next brings it up to date once the folder can be written")
    _kept(capsys, lb, home, verb, True, code, out, err, problem, ids, before)
    assert _bytes(lb) == held


def test_plan_replace_that_cannot_write_the_new_plan_keeps_the_old_one_and_names_its_copy(capsys, lb, home,
                                                                                         monkeypatch, locked):
    """Runs of the plan are recorded, so `--replace` keeps the old plan as plan.<id>.json first: with save_plan
    failing after that copy, the copy stays and is named and plan.json keeps its bytes; with the folder 0555, the
    copy cannot be made and nothing is written."""
    _stored(home, _merge(lb, 1))
    old = load(lb).plan
    plan, backup = lb / "plan.json", lb / f"plan.{old.id}.json"
    held, before = plan.read_bytes(), _bytes(lb)
    argv = ["plan", str(lb), "--margin", "0.15", "--replace", "--home", str(home)]
    with monkeypatch.context() as mp:
        mp.setattr(commands, "save_plan", _raise(OSError(errno.ENOSPC, "No space left on device")))
        code, out, err = _flow(capsys, *argv)
    assert code == 1 and err == (f"error: kept plan {old.id} as {backup}; {plan} cannot be written: No space left on "
                                 f"device; plan.json still holds plan {old.id}\n")
    assert "nothing was written" not in out and _bytes(lb) == {**before, backup.name: held}
    backup.rename(lb.parent / backup.name)
    locked(lb)
    code, out, err = _flow(capsys, *argv)
    assert code == 1 and err == f"error: plan {old.id} cannot be kept as {backup}: Permission denied\n{plan} is " \
                                "unchanged\n"
    assert _bytes(lb) == before


# A replaced score read back ---------------------------------------------------------------------------------------

def _numbers(fit: dict) -> dict:
    """What a fit holds of the scores it read: its judges, run noise and each run's q."""
    return {"s_run": fit["s_run"], "judges": [(j["id"], j["n"], j["bias"], j["noise"]) for j in fit["judges"]],
            "runs": {k: (v["q"], v["sd"]) for k, v in fit["runs"].items()}}


@pytest.mark.parametrize("state", ["late", "pending"])
def test_a_replaced_score_is_what_bs_fit_reads_back_from_the_store(state, capsys, lb, home):
    """Pairs 1 and 2; run 001.a finished (a late replacement) or open (a pending one, in the store's overlay). The
    documents read back give the fit and the unpooled value of a copy with only the replaced score left out."""
    docs = _merge(lb, 2)
    rid = next(d["run"]["id"] for d in docs if d["run"]["id"].endswith(".001.a"))
    _stored(home, docs, open_run=rid if state == "pending" else None)
    flow, store = load(lb).flow, Store(home)
    earlier = [store.run_doc(d["run"]["id"]) for d in docs]
    first = judges.unpooled_quality(flow, next(d for d in earlier if d["run"]["id"] == rid), judges.fit(flow, earlier))
    code, out, err = _score(capsys, lb, home, rid, "--replace", "--json", value="0.95")
    report = _strict(out)
    assert code == 0 and report["state"] == state and len(report["replaced"]) == 1
    read = [store.run_doc(d["run"]["id"]) for d in docs]
    finished = {d["run"]["id"]: d for d in store.finished_docs()}
    back = next(d for d in read if d["run"]["id"] == rid)
    if state == "late":
        assert finished[rid] == back
    else:
        assert rid not in finished and store.pending_signals(rid)
    control = copy.deepcopy(back)
    control["run"]["signals"] = [s for s in control["run"]["signals"] if s.get("id") not in report["replaced"]]
    assert len(control["run"]["signals"]) == len(back["run"]["signals"]) - 1
    controls = [control if d["run"]["id"] == rid else d for d in read]
    fit, fit_control = judges.fit(flow, read), judges.fit(flow, controls)
    assert _numbers(fit) == _numbers(fit_control)
    got = judges.unpooled_quality(flow, back, fit)
    assert got == judges.unpooled_quality(flow, control, fit_control) and got != first
