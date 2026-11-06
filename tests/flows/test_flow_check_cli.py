"""`loopmath flow check FLOW --arm ARM`: C's check of one arm on the store's runs of the flow, on the plan brought up
to date and not saved, with one judges fit. Every verdict exits 0 and a refusal 1; `--json` is
`loopmath.flow-check/1`, the check's members but `given` in field order, `extra` nested as the check gives it. Runs
are the merge check's pairs simulated from the example's truth with seed 1, its loss stated per case."""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.flows import check as check_mod
from loopmath.flows import judges, load, simulate
from loopmath.flows import plan as plan_mod
from loopmath.flows.next import next_runs
from loopmath.flows.records import CheckResult
from loopmath.store import Store, fitjob

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
TRUTH = json.loads((EXAMPLE / "truth.json").read_text(encoding="utf-8"))
ARM = "merge-extract-synthesize"
CHECK_KEYS = ["schema"] + [f.name for f in dataclasses.fields(CheckResult) if f.name != "given"]


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


def _flow(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["flow", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _check(capsys, folder: Path, home: Path, arm: str, *more: str) -> tuple[int, str, str]:
    return _flow(capsys, "check", str(folder), "--arm", arm, "--home", str(home), *more)


def _strict(text: str):
    def refuse(name):
        raise AssertionError(f"not standard JSON: {name}")

    return json.loads(text, parse_constant=refuse)


def _truth(loss: float) -> dict:
    truth = copy.deepcopy(TRUTH)
    for effect in truth["quality"]["effects"]:
        if effect["edit"]["kind"] == "merge":
            effect["quality"] = -loss
    return truth


def _merge_runs(folder: Path, loss: float, pairs: int, change=None) -> list[dict]:
    """The merge check's first `pairs` pairs, simulated with seed 1 at quality loss `loss`, each run changed by
    `change(run, pair index)` when given."""
    flow = load(folder)
    specs = [s for s in next_runs(flow, n=1000, seed=1) if s.experiment == ARM
             and int(s.run_id.split(".")[-2]) <= pairs]
    docs = simulate.simulate(flow, arm="base", n=0, seed=1, truth=_truth(loss), specs=specs)
    for doc in docs:
        if change is not None:
            change(doc["run"], int(doc["run"]["id"].split(".")[-2]))
    return docs


def _stored(home: Path, docs: list[dict]) -> list[dict]:
    store = Store(home)
    for doc in docs:
        store.import_run(doc)
    return [store.run_doc(d["run"]["id"]) for d in docs]


def _expected(folder: Path, docs: list[dict], arm: str = ARM) -> CheckResult:
    """C's check on the stored documents, its plan brought up to date as advance gives it."""
    flow = load(folder)
    quality, fit = plan_mod.quality_and_fit(flow.flow, plan_mod.flow_runs(flow, docs), None, None)
    advanced = plan_mod.advance(flow, docs, quality=quality, fit=fit)
    return check_mod.check(dataclasses.replace(flow, plan=advanced), arm, docs, quality=quality, fit=fit)


def _files(folder: Path, home: Path) -> dict[str, bytes]:
    return {p.relative_to(p.anchor).as_posix(): p.read_bytes() for root in (folder, home) if root.exists()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "lock"}


# The verdicts ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("verdict", "loss", "pairs"), [("adopt", 0.0, 24), ("reject", 0.3, 24), ("more", 0.0, 4)])
def test_each_verdict_exits_0_with_the_checks_members_and_writes_nothing(verdict, loss, pairs, capsys, lb, home):
    docs = _stored(home, _merge_runs(lb, loss, pairs))
    want = _expected(lb, docs)
    before = _files(lb, home)
    code, out, err = _check(capsys, lb, home, ARM, "--json")
    report = _strict(out)
    assert code == 0 and err == "" and list(report) == CHECK_KEYS
    assert report == {"schema": "loopmath.flow-check/1",
                      **json.loads(json.dumps({k: getattr(want, k) for k in CHECK_KEYS[1:]}))}
    assert (report["verdict"], report["pairs"], report["pair_cap"]) == (verdict, pairs, 24)
    code, out, err = _check(capsys, lb, home, ARM)
    lines = out.splitlines()
    assert code == 0 and err == ""
    assert lines[:2] == [f"arm {ARM} against base: {verdict}, {pairs} pairs of 24",
                         f"P(quality within margin 0.1) {want.p_quality:.3f}, P(cheaper per run) {want.p_cost:.3f}; "
                         f"threshold {want.threshold:g}"]
    more = [f"more pairs: {want.more_pairs}"] if verdict == "more" else []
    assert lines[2:] == more + list(want.extra["reasons"])
    assert _files(lb, home) == before


def test_one_judges_fit_serves_the_whole_check(capsys, lb, home, monkeypatch):
    _stored(home, _merge_runs(lb, 0.0, 4))
    calls, real = [], judges.fit
    monkeypatch.setattr(judges, "fit", lambda flow, docs, *a, **kw: calls.append(len(docs)) or real(flow, docs, *a,
                                                                                                     **kw))
    assert _check(capsys, lb, home, ARM, "--json")[0] == 0
    assert calls == [8]


# The plan the check reads -----------------------------------------------------------------------------------------

def test_the_check_reads_the_plan_brought_up_to_date_and_does_not_save_it(capsys, lb, home):
    """The screen's runs are stored, the saved plan not advanced: the subset check's arm is in the plan advance
    gives, not in the saved one (where it is a screen arm), and check and status read it alike."""
    flow = load(lb)
    docs = _stored(home, simulate.simulate(flow, arm="base", n=0, seed=5, truth=TRUTH,
                                           specs=next_runs(flow, n=32, seed=1)))
    quality, fit = plan_mod.quality_and_fit(flow.flow, plan_mod.flow_runs(flow, docs), None, None)
    advanced = plan_mod.advance(flow, docs, quality=quality, fit=fit)
    [subset] = [e for e in advanced.experiments if e.kind == "subset"]
    [saved] = [e for e in flow.plan.experiments if e.kind == "subset"]
    arm = subset.arms[0]
    assert saved.arms == () and arm in flow.plan.experiments[0].arms
    before = _files(lb, home)
    code, out, err = _check(capsys, lb, home, arm, "--json")
    report = _strict(out)
    assert code == 0, err
    assert report == {"schema": "loopmath.flow-check/1",
                      **json.loads(json.dumps({k: getattr(_expected(lb, docs, arm), k) for k in CHECK_KEYS[1:]}))}
    assert (report["arm"], report["verdict"], report["pairs"]) == (arm, "more", 0)
    code, out, err = _flow(capsys, "status", str(lb), "--home", str(home), "--json")
    [row] = [e for e in _strict(out)["experiments"] if e["id"] == subset.id]
    assert row["arms"] == [arm] and row["pairs"] == report["pairs"]
    assert _files(lb, home) == before


# Runs the check leaves out ----------------------------------------------------------------------------------------

def test_a_run_of_the_other_split_is_listed_as_excluded(capsys, lb, home):
    def other_split(run, pair):
        if pair == 2 and run["id"].endswith(".a"):
            run["flow"]["split"] = "train" if run["flow"]["split"] == "holdout" else "holdout"

    docs = _stored(home, _merge_runs(lb, 0.0, 4, other_split))
    want = _expected(lb, docs)
    [gone] = want.extra["excluded"]
    assert want.pairs == 3 and "split" in gone["reason"]
    code, out, err = _check(capsys, lb, home, ARM, "--json")
    assert code == 0 and _strict(out)["extra"]["excluded"] == [gone] and _strict(out)["pairs"] == 3
    code, out, err = _check(capsys, lb, home, ARM)
    assert code == 0 and f"excluded pair {gone['pair']} run {gone['run']}: {gone['reason']}" in out.splitlines()


def test_a_run_of_another_plan_is_not_counted_and_not_listed(capsys, lb, home):
    def other_plan(run, pair):
        if pair == 2 and run["id"].endswith(".a"):
            run["flow"]["plan"] = "pln_other"

    docs = _stored(home, _merge_runs(lb, 0.0, 4, other_plan))
    want = _expected(lb, docs)
    assert want.pairs == 3 and want.extra["excluded"] == []
    code, out, err = _check(capsys, lb, home, ARM, "--json")
    assert code == 0 and (_strict(out)["pairs"], _strict(out)["extra"]["excluded"]) == (3, [])
    code, out, err = _check(capsys, lb, home, ARM)
    assert code == 0 and not any(line.startswith("excluded pair ") for line in out.splitlines())


# Refusals ---------------------------------------------------------------------------------------------------------

def _c_refusal(folder: Path, arm: str) -> str:
    with pytest.raises(ValueError) as caught:
        _expected(folder, [], arm)
    return str(caught.value)


@pytest.mark.parametrize("which", ["base", "an arm not in the plan", "a screen arm"])
def test_an_arm_the_check_refuses_exits_1_in_its_words(which, capsys, lb, home):
    arm = {"base": "base", "an arm not in the plan": "no-such-arm",
           "a screen arm": load(lb).plan.experiments[0].arms[0]}[which]
    reason = _c_refusal(lb, arm)
    before = _files(lb, home)
    code, out, err = _check(capsys, lb, home, arm)
    assert code == 1 and out == "" and err == f"error: {reason}\n"
    code, out, err = _check(capsys, lb, home, arm, "--json")
    assert code == 1 and json.loads(out) == {"schema": "loopmath.error/1", "ok": False, "command": "flow check",
                                             "error": reason, "exit": 1}
    assert _files(lb, home) == before and not home.exists()


def test_with_no_plan_or_a_stale_one_the_check_is_refused(capsys, lb, home):
    plan = lb / "plan.json"
    stale = json.loads(plan.read_text(encoding="utf-8"))
    plan.unlink()
    code, out, err = _check(capsys, lb, home, ARM)
    assert code == 1 and err == f"error: there is no plan.json in {lb}; run loopmath flow plan first\n"
    stale["flow"]["digest"] = "flw_000000000000"
    plan.write_text(json.dumps(stale), encoding="utf-8")
    code, out, err = _check(capsys, lb, home, ARM)
    assert code == 1 and err == (f"error: {plan} is stale: it was made for flow digest flw_000000000000, the flow's "
                                 f"digest is {load(lb).digest}; run loopmath flow plan\n")
    assert not home.exists()


@pytest.mark.parametrize("change", [{"version": 2}, {"id": "other"}], ids=["version", "flow-id"])
def test_a_plan_made_for_another_flow_id_or_version_is_refused_not_checked(change, capsys, lb, home):
    """As `flow next` refuses it: the plan's digest matches, but it was made for another revision of the flow."""
    _stored(home, _merge_runs(lb, 0.0, 4))
    plan, digest = lb / "plan.json", load(lb).digest
    made = json.loads(plan.read_text(encoding="utf-8"))
    made["flow"].update(change)
    plan.write_text(json.dumps(made), encoding="utf-8")
    before = _files(lb, home)
    reason = (f"{plan} was made for flow {made['flow']['id']} version {made['flow']['version']} (digest {digest}), "
              f"and the flow is flow lit-brief version 1 (digest {digest}); run loopmath flow plan")
    code, out, err = _check(capsys, lb, home, ARM)
    assert (code, out, err) == (1, "", f"error: {reason}\n")
    code, out, err = _check(capsys, lb, home, ARM, "--json")
    assert code == 1 and json.loads(out) == {"schema": "loopmath.error/1", "ok": False, "command": "flow check",
                                             "error": reason, "exit": 1}
    assert _files(lb, home) == before
