"""`loopmath flow plan`: the example's plan as printed and under `--json`, the margin table, the price line, and
what a second `flow plan` does to a plan already there: keep it, replace it, or refuse while runs of it are
recorded, unless `--replace` keeps it as `plan.<id>.json`. The store is only read."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.flows import commands, load, simulate
from loopmath.flows.commands import write_flow_document
from loopmath.flows.next import next_runs
from loopmath.flows.plan import make_plan, margin_table
from loopmath.flows.records import PLAN_SCHEMA
from loopmath.ocp.canonical import flow_digest
from loopmath.store import Store, fitjob

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "examples" / "flows" / "lit-brief"
TRUTH = json.loads((FOLDER / "truth.json").read_text(encoding="utf-8"))
FACTORS = ("1. extract: model claude-sonnet-5-5 (hypothesis)", "2. critic: drop (hypothesis)",
           "3. synthesize: effort medium", "4. scout: width 2", "5. budget 1")
PRICE_UNKNOWN = "price unknown: no recorded run of this flow has dollars yet"
EXAMPLE_ID = "pln_97f47f3321db"


@pytest.fixture(autouse=True)
def no_fit_job(monkeypatch):
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})


@pytest.fixture
def lb(tmp_path):
    folder = tmp_path / "lit-brief"
    shutil.copytree(FOLDER, folder)
    return folder


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def _flow(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["flow", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _plan(capsys, lb: Path, home: Path, *more: str) -> tuple[int, str, str]:
    return _flow(capsys, "plan", str(lb), "--home", str(home), *more)


def _report(capsys, lb: Path, home: Path, *more: str) -> tuple[int, dict, str]:
    code, out, err = _plan(capsys, lb, home, *more, "--json")
    return code, json.loads(out), err


def _record(lb: Path, home: Path, n: int = 32) -> list[dict]:
    """The first `n` runs `flow next` gives, simulated with the judges their specs name, in the store."""
    folder = load(lb)
    docs = simulate.simulate(folder, arm="base", n=0, seed=5, truth=TRUTH, specs=next_runs(folder, n=n, seed=1))
    store = Store(home)
    for doc in docs:
        store.import_run(doc)
    return docs


def _tree(path: Path) -> dict[str, str]:
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


def _retitle(lb: Path, **flow) -> None:
    """Change the flow document: a digest member (`objective`) or a member the digest leaves out."""
    doc = json.loads((lb / "flow.ocp.json").read_text(encoding="utf-8"))
    for name, value in flow.items():
        if name == "margin":
            doc["flow"]["objective"]["margin"] = value
        else:
            doc["flow"][name] = value
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    write_flow_document(lb / "flow.ocp.json", doc)


def _backups(lb: Path) -> list[str]:
    return sorted(p.name for p in lb.glob("plan.*.json"))


# ------------------------------------------------------------------ the example's plan


def test_the_example_prints_its_screen_checks_and_queue(capsys, lb, home):
    code, out, err = _plan(capsys, lb, home)
    assert code == 0 and err == ""
    lines = out.splitlines()
    assert lines[0] == f"plan {EXAMPLE_ID}: written"
    assert lines[1] == "flow lit-brief, version 1, digest flw_774ab7a14c50" and lines[2] == "margin 0.1"
    at = lines.index("    factors, in order:")
    assert [line.strip() for line in lines[at + 1:at + 6]] == list(FACTORS)
    assert lines[at + 6] == "    16 rows, layout hadamard-16"
    assert lines[at - 1] == ("  screen (screen): 32 arms, against base, on train, pair cap 16, 32 runs, "
                             "price unknown")
    assert "  merge-extract-synthesize (merge): arm merge-extract-synthesize, against base, on holdout, pair cap " \
           "24, 48 runs, price unknown" in lines
    assert "    merge extract+synthesize; removed checks: none" in lines
    assert "  subset (subset): arm after screen, against base, on holdout, pair cap 24, 48 runs, price unknown" \
        in lines
    assert "queued: extract: model claude-haiku-5-5; budget 3" in lines and "excluded: none" in lines


def test_the_example_under_json_is_the_plan_written(capsys, lb, home):
    code, report, err = _report(capsys, lb, home)
    assert code == 0 and err == ""
    assert list(report) == ["schema", "plan", "written", "replaced", "margin_table", "fitted_table", "price_usd",
                            "price_unknown", "wait"]
    assert report["price_unknown"] == {"screen": "no recorded run of this flow has dollars yet",
                                       "merge-extract-synthesize": "no recorded run of this flow has dollars yet",
                                       "subset": "no recorded run of this flow has dollars yet"}
    assert report["schema"] == commands.PLAN_REPORT_SCHEMA == "loopmath.flow-plan-report/1"
    assert report["plan"]["schema"] == PLAN_SCHEMA != report["schema"]
    assert json.loads((lb / "plan.json").read_text(encoding="utf-8"))["schema"] == PLAN_SCHEMA
    assert (report["written"], report["replaced"], report["fitted_table"], report["wait"]) == (True, None, None, None)
    experiments = report["plan"]["experiments"]
    assert [(e["id"], e["kind"], e.get("split")) for e in experiments] == [
        ("screen", "screen", "train"), ("merge-extract-synthesize", "merge", "holdout"),
        ("subset", "subset", "holdout")]
    assert experiments[0]["design"]["factors"] == [
        {"kind": "model", "piece": "extract", "value": {"id": "claude-sonnet-5-5"}},
        {"kind": "drop", "piece": "critic"},
        {"kind": "effort", "piece": "synthesize", "value": "medium"}, {"kind": "width", "piece": "scout", "value": 2},
        {"kind": "budget", "value": 1}]
    assert experiments[2]["arms"] == [] and experiments[2]["design"]["after"] == "screen"
    assert report["plan"]["queued"] == [{"kind": "model", "piece": "extract", "value": {"id": "claude-haiku-5-5"}},
                                        {"kind": "budget", "value": 3}]


def test_the_margin_table_gives_the_interval_and_check_figures(capsys, lb, home):
    code, out, _ = _plan(capsys, lb, home)
    lines = out.splitlines()
    at = lines.index("margin table at the assumed spreads (s_run 0.12, s_judge 0.1, 2 judges averaged):")
    assert lines[at + 1:at + 4] == ["  margin 0.15: interval 7 pairs, check 11 pairs, 22 runs",
                                    "  margin 0.1: interval 16 pairs, check 24 pairs, 48 runs",
                                    "  margin 0.05: interval 63 pairs, check 96 pairs, 192 runs"]
    report = json.loads((lb / "plan.json").read_text(encoding="utf-8"))
    (lb / "plan.json").unlink()
    _, got, _ = _report(capsys, lb, home)
    assert [(r["interval_pairs"], r["check_pairs"]) for r in got["margin_table"]] == [(7, 11), (16, 24), (63, 96)]
    assert got["margin_table"] == margin_table() and got["plan"] == report


def test_a_command_judge_is_one_of_the_judges_the_table_averages(capsys, lb, home):
    """The table's k is the flow's judges that are not people: a command judge in place of a model judge leaves it
    at 2."""
    doc = json.loads((lb / "flow.ocp.json").read_text(encoding="utf-8"))
    assert [j["kind"] for j in doc["flow"]["judges"]] == ["model", "model", "person"]
    doc["flow"]["judges"][1] = {"id": "j-tests", "kind": "command"}
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    write_flow_document(lb / "flow.ocp.json", doc)
    code, got, _ = _report(capsys, lb, home)
    assert code == 0 and got["margin_table"] == margin_table(k=2) != margin_table(k=1)
    assert [r["k"] for r in got["margin_table"]] == [2, 2, 2]
    _, out, _ = _plan(capsys, lb, home)
    lines = out.splitlines()
    at = lines.index("margin table at the assumed spreads (s_run 0.12, s_judge 0.1, 2 judges averaged):")
    assert lines[at + 1:at + 4] == ["  margin 0.15: interval 7 pairs, check 11 pairs, 22 runs",
                                    "  margin 0.1: interval 16 pairs, check 24 pairs, 48 runs",
                                    "  margin 0.05: interval 63 pairs, check 96 pairs, 192 runs"]


def test_each_check_s_pair_cap_is_its_check_figure(capsys, lb, home):
    _, report, _ = _report(capsys, lb, home)
    figure = next(r["check_pairs"] for r in margin_table(margins=(report["plan"]["margin"],)))
    checks = [e for e in report["plan"]["experiments"] if e["kind"] != "screen"]
    assert len(checks) == 2 and all(e["pair_cap"] == figure == 24 for e in checks)


def test_with_no_dollars_recorded_the_price_is_unknown(capsys, lb, home):
    _, out, _ = _plan(capsys, lb, home)
    assert out.splitlines()[-1] == PRICE_UNKNOWN
    (lb / "plan.json").unlink()
    _, report, _ = _report(capsys, lb, home)
    assert report["price_usd"] == {"screen": None, "merge-extract-synthesize": None, "subset": None}


def test_plan_json_is_the_planner_s_plan(capsys, lb, home):
    _plan(capsys, lb, home)
    assert json.loads((lb / "plan.json").read_text(encoding="utf-8")) == make_plan(load(lb)).to_dict()


def test_the_store_is_only_read(capsys, lb, home):
    assert _plan(capsys, lb, home)[0] == 0
    assert not home.exists()
    _record(lb, home)
    before = _tree(home)
    assert _plan(capsys, lb, home)[0] == 0 and _plan(capsys, lb, home, "--margin", "0.05")[0] == 1
    assert _tree(home) == before


# ------------------------------------------------------------------ a plan already there


def test_the_same_id_keeps_the_plan(capsys, lb, home):
    _plan(capsys, lb, home)
    before = (lb / "plan.json").read_bytes()
    code, out, err = _plan(capsys, lb, home)
    assert (code, err) == (0, "") and out.splitlines()[0] == f"plan {EXAMPLE_ID}: kept"
    assert (lb / "plan.json").read_bytes() == before
    code, report, _ = _report(capsys, lb, home)
    assert (code, report["written"], report["replaced"]) == (0, False, None)


def test_the_same_id_after_recording_brings_the_plan_up_to_date(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    before = (lb / "plan.json").read_bytes()
    code, out, _ = _plan(capsys, lb, home)
    assert code == 0 and out.splitlines()[0] == f"plan {EXAMPLE_ID}: brought up to date"
    after = json.loads((lb / "plan.json").read_text(encoding="utf-8"))
    assert after["id"] == EXAMPLE_ID and (lb / "plan.json").read_bytes() != before
    assert after["experiments"][0]["state"] == "done" and _backups(lb) == []


def test_a_new_id_with_nothing_recorded_replaces_the_plan(capsys, lb, home):
    _plan(capsys, lb, home)
    code, out, err = _plan(capsys, lb, home, "--margin", "0.05")
    new = json.loads((lb / "plan.json").read_text(encoding="utf-8"))["id"]
    assert (code, err) == (0, "") and new != EXAMPLE_ID
    assert out.splitlines()[:2] == [f"replaced plan {EXAMPLE_ID} (flow digest flw_774ab7a14c50) with plan {new} "
                                    "(flow digest flw_774ab7a14c50)", f"plan {new}: written"]
    assert _backups(lb) == []


def test_a_new_id_with_runs_recorded_is_refused(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    before = (lb / "plan.json").read_bytes()
    code, out, err = _plan(capsys, lb, home, "--margin", "0.05")
    assert code == 1 and (lb / "plan.json").read_bytes() == before and _backups(lb) == []
    first, unchanged = err.splitlines()
    assert first.startswith(f"error: 32 runs of plan {EXAMPLE_ID} are recorded; a new plan gives new run ids, so "
                            "they would count for nothing in plan pln_")
    assert "what changed: margin 0.1 -> 0.05; pair caps screen 16, merge-extract-synthesize 24, subset 24 -> " in first
    assert first.endswith(f"; loopmath flow plan --replace writes the new plan and keeps the old one as plan."
                          f"{EXAMPLE_ID}.json in the flow folder")
    assert unchanged == f"{lb / 'plan.json'} is unchanged"
    assert out.splitlines()[0].endswith(": not written") and out.splitlines()[0] != f"plan {EXAMPLE_ID}: not written"


def test_a_refused_plan_under_json_is_one_object_not_written(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    code, out, err = _plan(capsys, lb, home, "--margin", "0.05", "--json")
    report = json.loads(out)  # one object, nothing else on stdout
    assert code == 1 and report["schema"] == commands.PLAN_REPORT_SCHEMA
    assert (report["written"], report["replaced"]) == (False, None) and report["plan"]["id"] != EXAMPLE_ID
    assert err.startswith("error: 32 runs of plan ") and "--replace" in err
    assert json.loads((lb / "plan.json").read_text(encoding="utf-8"))["id"] == EXAMPLE_ID


def test_a_candidate_at_fitted_spreads_has_its_fitted_table(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    _, report, _ = _report(capsys, lb, home, "--margin", "0.05")
    spreads = report["plan"]["experiments"][0]["design"]["spreads"]
    assert spreads["label"] == "fitted"
    assert report["fitted_table"] == margin_table(s_run=spreads["s_run"], s_judge=spreads["s_judge"], k=spreads["k"])
    _, out, _ = _plan(capsys, lb, home, "--margin", "0.05")
    assert any(line.startswith("margin table at the plan's fitted spreads (") for line in out.splitlines())


def test_replace_keeps_the_old_plan_byte_for_byte(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    before = (lb / "plan.json").read_bytes()
    code, report, err = _report(capsys, lb, home, "--margin", "0.05", "--replace")
    assert code == 0 and (report["written"], report["replaced"]) == (True, EXAMPLE_ID)
    assert _backups(lb) == [f"plan.{EXAMPLE_ID}.json"] and (lb / f"plan.{EXAMPLE_ID}.json").read_bytes() == before
    assert json.loads((lb / "plan.json").read_text(encoding="utf-8")) == report["plan"]
    assert err.splitlines()[0] == f"kept plan {EXAMPLE_ID} as {lb / f'plan.{EXAMPLE_ID}.json'}"


def test_replace_with_the_same_id_copies_nothing(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    before = (lb / "plan.json").read_bytes()
    code, out, err = _plan(capsys, lb, home, "--replace")
    assert (code, err) == (0, "") and out.splitlines()[0] == f"plan {EXAMPLE_ID}: kept"
    assert (lb / "plan.json").read_bytes() == before and _backups(lb) == []


def test_replace_with_nothing_recorded_copies_nothing(capsys, lb, home):
    _plan(capsys, lb, home)
    code, out, _ = _plan(capsys, lb, home, "--margin", "0.05", "--replace")
    assert code == 0 and out.startswith(f"replaced plan {EXAMPLE_ID} ") and _backups(lb) == []


def test_replace_refuses_a_backup_with_other_bytes(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    before = (lb / "plan.json").read_bytes()
    backup = lb / f"plan.{EXAMPLE_ID}.json"
    backup.write_bytes(b"{}\n")
    code, out, err = _plan(capsys, lb, home, "--margin", "0.05", "--replace")
    assert code == 1 and out.splitlines()[0].endswith(": not written")
    assert err == (f"error: {backup} is already there and is not a copy of {lb / 'plan.json'}; move it away, then "
                   f"run this again\n{lb / 'plan.json'} is unchanged\n")
    code, report, _ = _report(capsys, lb, home, "--margin", "0.05", "--replace")
    assert code == 1 and (report["written"], report["replaced"]) == (False, None)
    assert (lb / "plan.json").read_bytes() == before and backup.read_bytes() == b"{}\n"


def test_replace_goes_on_over_a_backup_with_the_same_bytes(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    before = (lb / "plan.json").read_bytes()
    (lb / f"plan.{EXAMPLE_ID}.json").write_bytes(before)
    code, report, _ = _report(capsys, lb, home, "--margin", "0.05", "--replace")
    assert code == 0 and report["replaced"] == EXAMPLE_ID and (lb / f"plan.{EXAMPLE_ID}.json").read_bytes() == before


@pytest.mark.parametrize(("plan_id", "shown"), [("x/../../outside", "x/../../outside"), ("sub/x", "sub/x"),
                                                ("a" + chr(0) + "b", "a\\u0000b")],
                         ids=["out-of-the-folder", "a-folder-in-it", "nul"])
def test_replace_refuses_a_plan_id_that_is_not_one_file_name(plan_id, shown, capsys, lb, home):
    """The old plan is kept under a name made from plan.json's id, so the id is data: one that gives no single file
    name in the flow folder is refused by name, and nothing is made, copied or replaced, in the folder or out."""
    _plan(capsys, lb, home)
    folder = load(lb)
    docs = simulate.simulate(folder, arm="base", n=0, seed=5, truth=TRUTH, specs=next_runs(folder, n=2, seed=1))
    saved = json.loads((lb / "plan.json").read_text(encoding="utf-8"))
    saved["id"] = plan_id
    (lb / "plan.json").write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    assert load(lb).plan.id == plan_id
    store = Store(home)
    for doc in docs:
        doc["run"]["flow"]["plan"] = plan_id
        store.import_run(doc)
    (lb.parent / "outside.json").write_bytes(b"another file\n")
    files, paths = _tree(lb.parent), sorted(lb.parent.rglob("*"))
    refusal = (f"error: plan {shown} cannot be kept as plan.{shown}.json: that is not one file name in {lb}\n"
               f"{lb / 'plan.json'} is unchanged\n")
    for json_flag in ([], ["--json"]):
        code, out, err = _plan(capsys, lb, home, "--margin", "0.05", "--replace", *json_flag)
        assert (code, err) == (1, refusal)
        if json_flag:
            assert (json.loads(out)["written"], json.loads(out)["replaced"]) == (False, None)
        else:
            assert out.splitlines()[0].endswith(": not written")
        assert _tree(lb.parent) == files and sorted(lb.parent.rglob("*")) == paths


@pytest.mark.parametrize("to", ["plan.json", "a copy outside"], ids=["to-plan-json", "to-a-copy-outside"])
def test_replace_refuses_a_link_where_the_old_plan_is_kept(to, capsys, lb, home):
    """A link is no copy of its own: one to plan.json would read the new plan once plan.json is written, and one to
    a copy elsewhere keeps nothing in the flow folder."""
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    plan_json, backup = lb / "plan.json", lb / f"plan.{EXAMPLE_ID}.json"
    before = plan_json.read_bytes()
    if to == "plan.json":
        backup.symlink_to("plan.json")
    else:
        (lb.parent / "copy.json").write_bytes(before)
        backup.symlink_to(lb.parent / "copy.json")
    assert backup.read_bytes() == before
    files = _tree(lb.parent)
    refusal = (f"error: {backup} is already there and is not a copy of {plan_json}; move it away, then run this "
               f"again\n{plan_json} is unchanged\n")
    for json_flag in ([], ["--json"]):
        code, out, err = _plan(capsys, lb, home, "--margin", "0.05", "--replace", *json_flag)
        assert (code, err) == (1, refusal)
        if json_flag:
            assert (json.loads(out)["written"], json.loads(out)["replaced"]) == (False, None)
        else:
            assert out.splitlines()[0].endswith(": not written")
        assert _tree(lb.parent) == files and backup.is_symlink()


def test_replace_goes_on_over_a_hard_link_to_plan_json(capsys, lb, home):
    """plan.json is replaced, not written over, so a hard link to it is a copy of its own: it keeps the old plan."""
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    plan_json, backup = lb / "plan.json", lb / f"plan.{EXAMPLE_ID}.json"
    before = plan_json.read_bytes()
    os.link(plan_json, backup)
    code, report, err = _report(capsys, lb, home, "--margin", "0.05", "--replace")
    assert code == 0 and (report["written"], report["replaced"]) == (True, EXAMPLE_ID)
    assert backup.read_bytes() == before and json.loads(plan_json.read_text(encoding="utf-8")) == report["plan"]
    assert err.splitlines()[0] == f"kept plan {EXAMPLE_ID} as {backup}"


def test_replace_puts_the_old_plan_back_when_its_copy_no_longer_holds_it(capsys, lb, home, monkeypatch):
    """The kept copy is read again once the new plan is written: a plan.json written over, not replaced, writes a
    hard link to it too, so the old plan is put back in plan.json and the new one is refused."""
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    plan_json, backup = lb / "plan.json", lb / f"plan.{EXAMPLE_ID}.json"
    before = plan_json.read_bytes()
    os.link(plan_json, backup)

    def written_over(directory, plan):
        Path(directory, "plan.json").write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(commands, "save_plan", written_over)
    code, report, err = _report(capsys, lb, home, "--margin", "0.05", "--replace")
    assert code == 1 and (report["written"], report["replaced"]) == (False, None)
    assert err == (f"error: {backup} no longer held plan {EXAMPLE_ID} once the new plan was written, so the old plan "
                   f"is put back in {plan_json}\n{plan_json} is unchanged\n")
    assert plan_json.read_bytes() == before


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a folder its owner cannot write")
def test_replace_refuses_when_the_old_plan_cannot_be_kept(capsys, lb, home):
    """A copy the file system refuses, here in a flow folder that cannot be written, is refused by name."""
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    plan_json = lb / "plan.json"
    before = plan_json.read_bytes()
    lb.chmod(0o555)
    try:
        code, out, err = _plan(capsys, lb, home, "--margin", "0.05", "--replace")
    finally:
        lb.chmod(0o755)
    assert code == 1 and out.splitlines()[0].endswith(": not written")
    assert err == (f"error: plan {EXAMPLE_ID} cannot be kept as {lb / f'plan.{EXAMPLE_ID}.json'}: Permission "
                   f"denied\n{plan_json} is unchanged\n")
    assert plan_json.read_bytes() == before and _backups(lb) == []


# ------------------------------------------------------------------ budgets


def test_a_budget_is_set_kept_and_refused_on_the_same_plan(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)  # the ordinary advance first, so only the budget moves below
    settled = json.loads((lb / "plan.json").read_text(encoding="utf-8"))
    assert "budget_usd" not in settled
    for flag, budget, line in ((["--budget-usd", "5"], 5.0, "budget: none -> $5.00"),
                               (["--budget-usd", "0"], 0.0, "budget: $5.00 -> $0"), ([], 0.0, None)):
        code, out, err = _plan(capsys, lb, home, *flag)
        saved = json.loads((lb / "plan.json").read_text(encoding="utf-8"))
        assert (code, err) == (0, "") and saved == {**settled, "budget_usd": budget}
        lines = out.splitlines()
        if line is None:
            assert lines[0] == f"plan {EXAMPLE_ID}: kept"
        else:
            assert lines[:2] == [line, f"plan {EXAMPLE_ID}: brought up to date"]
        assert f"margin 0.1, budget {'$5.00' if budget else '$0'}" in lines
        code, report, _ = _report(capsys, lb, home, *flag)  # asked again: the same plan, kept
        assert (code, report["written"], report["plan"]) == (0, False, saved)
    before = (lb / "plan.json").read_bytes()
    code, out, err = _plan(capsys, lb, home, "--budget-usd", "-1")
    assert (code, out) == (1, "")
    assert err == "error: budget_usd -1.0: a budget is a number of dollars, 0 or more\nnothing was written\n"
    assert (lb / "plan.json").read_bytes() == before


# ------------------------------------------------------------------ a stale plan


def test_a_stale_plan_with_nothing_recorded_is_replaced(capsys, lb, home):
    _plan(capsys, lb, home)
    _retitle(lb, margin=0.05)
    code, out, err = _plan(capsys, lb, home)
    new = load(lb)
    assert (code, err) == (0, "") and new.plan.flow.digest == new.digest != "flw_774ab7a14c50"
    assert out.splitlines()[0] == (f"replaced plan {EXAMPLE_ID} (flow digest flw_774ab7a14c50) with plan "
                                   f"{new.plan.id} (flow digest {new.digest})")
    assert _backups(lb) == []


def test_a_stale_plan_with_its_runs_recorded_is_refused(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    before = (lb / "plan.json").read_bytes()
    _retitle(lb, margin=0.05)
    now = load(lb)
    code, out, err = _plan(capsys, lb, home)
    assert code == 1 and (lb / "plan.json").read_bytes() == before and _backups(lb) == []
    assert err.startswith(f"error: 32 runs of plan {EXAMPLE_ID} are recorded; they belong to plan {EXAMPLE_ID} of "
                          "the old flow revision and to no experiment of the new one, plan pln_")
    assert (f"; the flow changed: digest flw_774ab7a14c50 -> {now.digest}; loopmath flow plan --replace writes the "
            f"new plan and keeps the old one as plan.{EXAMPLE_ID}.json in the flow folder\n") in err
    assert out.splitlines()[0].endswith(": not written")

    code, report, err = _report(capsys, lb, home, "--replace")
    assert code == 0 and report["replaced"] == EXAMPLE_ID and (lb / f"plan.{EXAMPLE_ID}.json").read_bytes() == before
    assert load(lb).plan.flow.digest == now.digest


RENAMES = pytest.mark.parametrize("change", [{"id": "lit-brief-2"}, {"version": 2}], ids=["renamed", "version"])


@RENAMES
def test_a_renamed_flow_s_same_id_plan_is_refused_over_recorded_runs(change, capsys, lb, home, monkeypatch):
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)  # the same reference and id: advanced, as the ruled case 1
    before, stored = (lb / "plan.json").read_bytes(), _tree(home)
    kept = lb / f"plan.{EXAMPLE_ID}.json"
    kept.write_bytes(b"a copy kept by hand\n")
    _retitle(lb, **change)
    now = load(lb)
    assert make_plan(now).id == EXAMPLE_ID  # the plan id leaves out the flow id and version
    assert now.plan.id == EXAMPLE_ID and not now.plan_stale  # the digest is the same, so C reads it as current
    refusal = (f"error: 32 runs of plan {EXAMPLE_ID} are recorded under flow lit-brief version 1 (digest "
               f"flw_774ab7a14c50); the flow is now flow {now.flow.id} version {now.flow.version} (digest "
               "flw_774ab7a14c50), and the same content gives the same plan id and run ids, so recording them would "
               "replace the recorded runs in the store; keep the flow's id and version while its runs are recorded\n"
               f"{lb / 'plan.json'} is unchanged\n")
    for flag in ([], ["--replace"]):
        code, out, err = _plan(capsys, lb, home, *flag)
        assert (code, err) == (1, refusal) and "--replace" not in err
        assert out.splitlines()[0] == f"plan {EXAMPLE_ID}: not written"
        code, out, err = _plan(capsys, lb, home, *flag, "--json")
        report = json.loads(out)
        assert (code, err) == (1, refusal) and (report["written"], report["replaced"]) == (False, None)
        assert report["plan"]["id"] == EXAMPLE_ID
        assert (report["plan"]["flow"]["id"], report["plan"]["flow"]["version"]) == (now.flow.id, now.flow.version)
        assert (lb / "plan.json").read_bytes() == before and _tree(home) == stored
        assert _backups(lb) == [kept.name] and kept.read_bytes() == b"a copy kept by hand\n"


@RENAMES
def test_flow_next_refuses_a_plan_for_another_flow_id_or_version(change, capsys, lb, home, monkeypatch):
    from loopmath.flows import next as next_mod
    from loopmath.flows import plan as plan_mod

    _plan(capsys, lb, home)
    _record(lb, home)
    before, stored = (lb / "plan.json").read_bytes(), _tree(home)
    _retitle(lb, **change)
    now = load(lb)

    def called(*args, **kwargs):
        raise AssertionError("called on a plan for another flow reference")

    monkeypatch.setattr(plan_mod, "advance", called)
    monkeypatch.setattr(next_mod, "next_runs", called)
    for json_flag in ([], ["--json"]):
        code, out, err = _flow(capsys, "next", str(lb), "--home", str(home), *json_flag)
        reason = (f"{lb / 'plan.json'} was made for flow lit-brief version 1 (digest flw_774ab7a14c50), and the flow "
                  f"is flow {now.flow.id} version {now.flow.version} (digest flw_774ab7a14c50); run loopmath flow plan")
        assert code == 1 and err == f"error: {reason}\n"
        assert (out == "") if not json_flag else (json.loads(out)["error"] == reason)
        assert (lb / "plan.json").read_bytes() == before and _tree(home) == stored


@RENAMES
def test_a_renamed_flow_with_nothing_recorded_gets_its_plan(change, capsys, lb, home):
    _plan(capsys, lb, home)
    _retitle(lb, **change)
    code, out, err = _plan(capsys, lb, home)
    now = load(lb)
    assert (code, err) == (0, "") and now.plan.id == EXAMPLE_ID
    assert (now.plan.flow.id, now.plan.flow.version) == (now.flow.id, now.flow.version)
    assert out.splitlines()[0] == (f"replaced plan {EXAMPLE_ID} (flow digest flw_774ab7a14c50) with plan "
                                   f"{EXAMPLE_ID} (flow digest flw_774ab7a14c50)")
    code, out, _ = _flow(capsys, "next", str(lb), "--home", str(home), "--json")
    runs = json.loads(out)["runs"]
    assert code == 0 and len(runs) == 2
    assert all((r["flow"]["id"], r["flow"]["version"], r["flow"]["plan"]) == (now.flow.id, now.flow.version,
                                                                              EXAMPLE_ID) for r in runs)


@RENAMES
def test_a_renamed_flow_with_a_new_plan_id_keeps_the_replace_path(change, capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    _plan(capsys, lb, home)
    before = (lb / "plan.json").read_bytes()
    _retitle(lb, **change)
    code, _, err = _plan(capsys, lb, home, "--margin", "0.05")
    assert code == 1 and "loopmath flow plan --replace writes the new plan" in err
    assert (lb / "plan.json").read_bytes() == before
    code, report, _ = _report(capsys, lb, home, "--margin", "0.05", "--replace")
    assert code == 0 and report["replaced"] == EXAMPLE_ID and report["plan"]["id"] != EXAMPLE_ID
    assert (lb / f"plan.{EXAMPLE_ID}.json").read_bytes() == before


def test_an_unusual_flow_id_keeps_each_diagnostic_on_one_line(capsys, lb, home):
    odd = "lit" + chr(10) + "brief" + chr(0x2028) + "2"  # a valid id: the schema bounds only its length
    shown = "lit\\u000abrief\\u20282"
    _plan(capsys, lb, home)
    _record(lb, home)
    _retitle(lb, id=odd)
    for flag in ([], ["--replace"]):
        code, out, err = _plan(capsys, lb, home, *flag, "--json")
        assert code == 1 and json.loads(out)["plan"]["flow"]["id"] == odd
        assert len(err.splitlines()) == 2 and f"the flow is now flow {shown} version 1" in err
        code, out, err = _plan(capsys, lb, home, *flag)
        assert code == 1 and len(err.splitlines()) == 2
        assert out.splitlines()[1] == f"flow {shown}, version 1, digest flw_774ab7a14c50"
    code, out, err = _flow(capsys, "next", str(lb), "--home", str(home), "--json")
    assert code == 1 and len(err.splitlines()) == 1 and f"the flow is flow {shown} version 1" in err
    assert json.loads(out)["error"] == err[len("error: "):-1]


def test_a_stale_plan_with_a_conflicting_backup_is_refused(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home)
    before = (lb / "plan.json").read_bytes()
    _retitle(lb, margin=0.05)
    backup = lb / f"plan.{EXAMPLE_ID}.json"
    backup.write_bytes(b"other\n")
    code, out, err = _plan(capsys, lb, home, "--replace")
    assert code == 1 and out.splitlines()[0].endswith(": not written")
    assert err.startswith(f"error: {backup} is already there") and err.endswith(f"{lb / 'plan.json'} is unchanged\n")
    assert (lb / "plan.json").read_bytes() == before and backup.read_bytes() == b"other\n"


# ------------------------------------------------------------------ the price

# `flow plan` over the eight runs `_record(n=8)` stores: the text and `--json` before the price past what a float
# holds was read, kept byte for byte.
NORMAL_TEXT = "".join(line + "\n" for line in (
    f"plan {EXAMPLE_ID}: brought up to date",
    "flow lit-brief, version 1, digest flw_774ab7a14c50",
    "margin 0.1",
    "experiments:",
    "  screen (screen): 32 arms, against base, on train, pair cap 16, 32 runs, price $21.41; running",
    "    factors, in order:",
    *(f"      {factor}" for factor in FACTORS),
    "    16 rows, layout hadamard-16",
    "  merge-extract-synthesize (merge): arm merge-extract-synthesize, against base, on holdout, pair cap 24, "
    "48 runs, price $32.12",
    "    merge extract+synthesize; removed checks: none",
    "  subset (subset): arm after screen, against base, on holdout, pair cap 24, 48 runs, price $32.12",
    "queued: extract: model claude-haiku-5-5; budget 3",
    "excluded: none",
    "margin table at the assumed spreads (s_run 0.12, s_judge 0.1, 2 judges averaged):",
    "  margin 0.15: interval 7 pairs, check 11 pairs, 22 runs",
    "  margin 0.1: interval 16 pairs, check 24 pairs, 48 runs",
    "  margin 0.05: interval 63 pairs, check 96 pairs, 192 runs",
    "price: runs times $0.67, the mean dollars of this flow's recorded runs"))
NORMAL_PRICES = {"screen": 21.414351999999997, "merge-extract-synthesize": 32.121528, "subset": 32.121528}
REPORT_KEYS = ["schema", "plan", "written", "replaced", "margin_table", "fitted_table", "price_usd", "wait"]
MEAN_PAST = "the mean dollars of this flow's recorded runs is past what a float holds"
EXPERIMENTS = ("screen", "merge-extract-synthesize", "subset")


def _strict(text: str) -> dict:
    """`text` read as standard JSON: no Infinity or NaN, and it writes back with `allow_nan=False`."""
    def refuse(constant):
        raise ValueError(f"not standard JSON: {constant}")

    data = json.loads(text, parse_constant=refuse)
    json.dumps(data, allow_nan=False)
    return data


def _priced(lb: Path, home: Path, usd) -> list[dict]:
    """`_record(n=8)`'s runs, the first attempt of run i priced at `usd(i)` dollars (None: as simulated)."""
    folder = load(lb)
    docs = simulate.simulate(folder, arm="base", n=0, seed=5, truth=TRUTH, specs=next_runs(folder, n=8, seed=1))
    for i, doc in enumerate(docs):
        if usd(i) is not None:
            doc["attempts"][0]["cost"]["usd"] = usd(i)
    store = Store(home)
    for doc in docs:
        store.import_run(doc)
    return docs


def _design_prices(lb: Path) -> dict:
    return {e["id"]: e["design"]["price"] for e in _strict((lb / "plan.json").read_text(encoding="utf-8"))
            ["experiments"]}


def test_at_normal_costs_the_report_is_as_it_was(capsys, lb, home):
    _plan(capsys, lb, home)
    _record(lb, home, n=8)
    code, out, err = _plan(capsys, lb, home)
    assert (code, err) == (0, "") and out == NORMAL_TEXT
    code, out, err = _plan(capsys, lb, home, "--json")
    report = _strict(out)
    assert (code, err) == (0, "") and list(report) == REPORT_KEYS and report["price_usd"] == NORMAL_PRICES
    assert out == json.dumps(report, indent=1, ensure_ascii=False) + "\n"
    assert report["plan"] == json.loads((lb / "plan.json").read_text(encoding="utf-8"))


def _product(runs: int) -> str:
    return f"the price of {runs} runs at 1e+307 dollars each is past what a float holds"


# 277's cases: two runs of 1e308 dollars, whose sum overflows; 1e307 on every run, whose price overflows
OVERFLOWS = pytest.mark.parametrize(("usd", "why", "mean_known"), [
    (lambda i: 1e308 if i < 2 else None, dict.fromkeys(EXPERIMENTS, MEAN_PAST), False),
    (lambda i: 1e307, {"screen": _product(32), "merge-extract-synthesize": _product(48), "subset": _product(48)},
     True)], ids=["sum", "each"])


@OVERFLOWS
def test_a_price_past_what_a_float_holds_is_unknown_with_its_reason(usd, why, mean_known, capsys, lb, home):
    _plan(capsys, lb, home)
    _priced(lb, home, usd)
    code, out, err = _plan(capsys, lb, home)
    assert (code, err) == (0, "") and "inf" not in out
    lines = out.splitlines()
    for exp in EXPERIMENTS:
        [line] = [x for x in lines if x.startswith(f"  {exp} (")]
        assert ("price unknown" + ("" if why[exp] == MEAN_PAST else f" ({why[exp]})")) in line
    assert lines[-1].startswith("price: runs times $") if mean_known else lines[-1] == f"price unknown: {MEAN_PAST}"
    code, out, err = _plan(capsys, lb, home, "--json")
    report = _strict(out)
    assert (code, err) == (0, "") and list(report) == [*REPORT_KEYS[:7], "price_unknown", "wait"]
    assert report["price_usd"] == dict.fromkeys(EXPERIMENTS) and report["price_unknown"] == why
    assert report["plan"] == _strict((lb / "plan.json").read_text(encoding="utf-8"))
    (lb / "plan.json").unlink()  # a plan made over these runs: make_plan's own prices, also unknown
    code, out, err = _plan(capsys, lb, home, "--json")
    report = _strict(out)
    assert (code, report["written"], report["price_unknown"]) == (0, True, why)
    made = ("the mean dollars of the 8 recorded runs is past what a float holds" if not mean_known else None)
    for exp, price in _design_prices(lb).items():
        assert price["usd"] is None and price["reason"] == (made or why[exp])


def test_dollars_no_float_holds_stored_open_read_as_a_price_past_a_float(capsys, lb, home, tmp_path):
    """277's third case: an attempt's dollars of 10**400, stored open by `run import`, as it stores them."""
    _plan(capsys, lb, home)
    folder = load(lb)
    doc = simulate.simulate(folder, arm="base", n=0, seed=5, truth=TRUTH, specs=next_runs(folder, n=2, seed=1))[0]
    doc["attempts"][0]["cost"]["usd"] = 10 ** 400
    path = tmp_path / "big.ocp.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert cli.main(["run", "import", str(path), "--home", str(home), "--no-fit", "--allow-simulated"]) == 0
    capsys.readouterr()
    for more in ([], ["--json"]):
        code, out, err = _plan(capsys, lb, home, *more)
        assert (code, err) == (0, "") and "Traceback" not in out
        if more:
            report = _strict(out)
            assert report["price_usd"] == dict.fromkeys(EXPERIMENTS)
            assert report["price_unknown"] == dict.fromkeys(EXPERIMENTS, MEAN_PAST)
        else:
            assert out.splitlines()[-1] == f"price unknown: {MEAN_PAST}" and "inf" not in out
    _strict((lb / "plan.json").read_text(encoding="utf-8"))


def _pool(capsys, lb: Path, home: Path, priced_base: int) -> list[dict]:
    """Four base runs, the first `priced_base` with dollars and the rest with an attempt's dollars dropped, and a
    screen run, of another arm, with an attempt's dollars of 10**400, stored open."""
    _plan(capsys, lb, home)
    folder = load(lb)
    base = simulate.simulate(folder, arm="base", n=4, seed=7, truth=TRUTH)
    for doc in base[priced_base:]:
        del doc["attempts"][0]["cost"]["usd"]
    other = simulate.simulate(folder, arm="base", n=0, seed=5, truth=TRUTH, specs=next_runs(folder, n=2, seed=1))[0]
    assert other["run"]["flow"]["arm"] not in (None, "base")
    other["attempts"][0]["cost"]["usd"] = 10 ** 400
    store = Store(home)
    for doc in base:
        store.import_run(doc)
    store.import_run(other, finished=False)
    return [*base, other]


@pytest.mark.parametrize("priced_base", [4, 2], ids=["priced-base", "partly-priced-base"])
def test_dollars_no_float_holds_on_a_run_outside_the_priced_base_runs_leave_the_price_known(priced_base, capsys, lb,
                                                                                           home):
    """The mean is over the base runs whose dollars are complete, as make_plan's is, so the screen run is not
    read into it."""
    docs = _pool(capsys, lb, home, priced_base)
    (lb / "plan.json").unlink()
    code, out, err = _plan(capsys, lb, home, "--json")
    report = _strict(out)
    assert (code, err) == (0, "") and list(report) == REPORT_KEYS
    made = {e.id: e.design["price"]["usd"] for e in make_plan(load(lb), docs=docs).experiments}
    assert {k: round(v, 4) for k, v in report["price_usd"].items()} == made
    assert all(isinstance(v, float) for v in made.values())


def test_with_no_priced_base_run_the_run_no_float_holds_is_read_and_the_price_is_unknown(capsys, lb, home):
    _pool(capsys, lb, home, 0)
    code, out, err = _plan(capsys, lb, home, "--json")
    report = _strict(out)
    assert (code, err) == (0, "") and report["price_usd"] == dict.fromkeys(EXPERIMENTS)
    assert report["price_unknown"] == dict.fromkeys(EXPERIMENTS, MEAN_PAST)
    code, out, _ = _plan(capsys, lb, home)
    assert code == 0 and out.splitlines()[-1] == f"price unknown: {MEAN_PAST}"


def test_each_price_is_twice_the_pair_cap_runs_at_the_mean(capsys, lb, home):
    """`pricepairs`: the report's price is make_plan's, two runs to a pair."""
    _plan(capsys, lb, home)
    docs = _record(lb, home, n=8)
    _, report, _ = _report(capsys, lb, home)
    made = {e.id: e.design["price"]["usd"] for e in make_plan(load(lb), docs=docs).experiments}
    assert {k: round(v, 4) for k, v in report["price_usd"].items()} == made == {
        "screen": 21.4144, "merge-extract-synthesize": 32.1215, "subset": 32.1215}


@pytest.mark.parametrize("judges", [["person"], []], ids=["person-only", "no-judges"])
def test_with_no_model_judge_the_assumed_spreads_average_one_judge(judges, capsys, lb, home):
    """`kzero`: a flow judged by a person alone, or by no judge, reads k 1, not 0."""
    doc = json.loads((lb / "flow.ocp.json").read_text(encoding="utf-8"))
    _retitle(lb, judges=[j for j in doc["flow"]["judges"] if j["id"] in judges])
    code, out, err = _plan(capsys, lb, home)
    assert (code, err) == (0, "") and "margin table at the assumed spreads (s_run 0.12, s_judge 0.1, 1 judge " \
                                      "averaged):" in out.splitlines()
    (lb / "plan.json").unlink()
    code, report, err = _report(capsys, lb, home)
    assert (code, err) == (0, "") and [r["k"] for r in report["margin_table"]] == [1, 1, 1]


@pytest.mark.parametrize("more", [[], ["--margin", "0.05", "--replace"]], ids=["fresh", "replace"])
def test_a_report_that_fails_comes_before_any_write(more, capsys, lb, home, monkeypatch):
    """The report is made before the old plan is kept and before plan.json is written, so a failure in it leaves
    the flow folder as it was: no plan.json on a fresh flow, no backup and the old plan on --replace."""
    if more:
        _plan(capsys, lb, home)
        _record(lb, home)
        _plan(capsys, lb, home)
    before = _tree(lb)

    def fails(*args, **kwargs):
        raise RuntimeError("the report failed")

    monkeypatch.setattr(commands, "_plan_report", fails)
    with pytest.raises(RuntimeError, match="the report failed"):
        _plan(capsys, lb, home, *more)
    assert _tree(lb) == before and _backups(lb) == []
