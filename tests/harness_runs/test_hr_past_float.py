"""A harness-runs row whose score or dollars are an integer no float holds: `check_row` names the member, "an integer
past what a float holds", at each place it reads a number (a row's `outcome.score`, a flow row's
`outcome.judges[i].score`, `cost.usd` and `cost.partial_usd` of the row and of each stage), of either sign, with or
without a flow. So `flow record --rows` and `adapt harness-runs` refuse the row by name and write nothing. An integer
a float holds, `int(sys.float_info.max) + 1` too (it rounds to the largest float), reads as before."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

from loopmath import cli, flows
from loopmath import harness_runs as HR
from loopmath.harness_runs import rows as HRrows
from loopmath.store import fitjob
from tests.flows.test_flow_record import AT, FLOW_FILE, PLAN
from tests.flows.test_flow_record import _row as _flow_row

PAST = "an integer past what a float holds"
TOP = int(sys.float_info.max) + 1  # past the largest float, yet `float` rounds it to that float
EDGE = 2**1024 - 2**970  # the least integer `float` cannot convert
HELD = {"10**308": 10**308, "1e308": 1e308, "max+1": TOP}
# Each place `check_row` reads a number, and whether a row with no flow reaches it (`_check_judges` reads the judges
# of a row that names a flow only).
SITES = {"outcome.score": True, "outcome.judges.0.score": False, "cost.usd": True, "cost.partial_usd": True,
         "stages.0.cost.usd": True, "stages.0.cost.partial_usd": True}
CASES = [(site, flow) for site, no_flow in SITES.items() for flow in ((True, False) if no_flow else (True,))]
IDS = [f"{site}-{'flow' if flow else 'no-flow'}" for site, flow in CASES]


@pytest.fixture(autouse=True)
def fits(monkeypatch):
    calls: list[Path] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(Path(home)) or {"started": True, "pid": 0})
    return calls


@pytest.fixture
def lb(tmp_path) -> Path:
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    shutil.copyfile(FLOW_FILE, folder / "flow.ocp.json")
    shutil.copyfile(PLAN, folder / "plan.json")
    return folder


def _row(site: str, value, flow: bool) -> dict:
    """The flow's row, or with no flow a row scored by its own `outcome.score`, with `value` at `site`. A row's own
    `cost` goes with no stages (and, with no flow, no workflow)."""
    row = copy.deepcopy(_flow_row())
    if not flow:
        del row["flow"]
        row["outcome"] = {"score": 0.8}
    cost = {"usd": value} if site.endswith(".usd") else {"usd": None, "partial_usd": value, "complete": False}
    if site == "outcome.score":
        row["outcome"] = {"score": value}
    elif site == "outcome.judges.0.score":
        row["outcome"]["judges"][0]["score"] = value
    elif site.startswith("cost."):
        del row["stages"]
        if not flow:
            del row["workflow"]
        row["cost"] = cost
    else:
        row["stages"][0]["cost"] = cost
    return row


def _tree(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "lock"}


def test_the_least_integer_no_float_holds_is_past_it_of_either_sign():
    for sign in (1, -1):
        assert HRrows._past_float(sign * EDGE) and not HRrows._number(sign * EDGE)
        assert not HRrows._past_float(sign * (EDGE - 1)) and HRrows._number(sign * (EDGE - 1))
        assert not HRrows._past_float(sign * TOP) and HRrows._number(sign * TOP)
    for value in (True, 1.5, float("inf"), "1", None):
        assert not HRrows._past_float(value)


@pytest.mark.parametrize("value", [10**400, -10**400], ids=["plus", "minus"])
@pytest.mark.parametrize(("site", "flow"), CASES, ids=IDS)
def test_an_integer_no_float_holds_is_named_at_each_site(site, flow, value):
    held = HRrows.check_row(_row(site, 0.5, flow))
    got = HRrows.check_row(_row(site, value, flow))
    assert f"{site}: {PAST}" in got
    assert [p for p in got if p != f"{site}: {PAST}"] == held


@pytest.mark.parametrize("label", list(HELD))
@pytest.mark.parametrize(("site", "flow"), CASES, ids=IDS)
def test_an_integer_a_float_holds_reads_as_before(site, flow, label, lb):
    """No new problem for these, and a row with no problem converts as before: but for this flow's judges, which
    score a fraction, so the converter refuses these values, as before."""
    row = _row(site, HELD[label], flow)
    problems = HRrows.check_row(row)
    assert problems == HRrows.check_row(_row(site, 0.5, flow))
    if problems:
        return
    if site == "outcome.judges.0.score":
        with pytest.raises(ValueError, match=r"scores a fraction in \[0, 1\]"):
            HR.to_ocp(row, flow=flows.load(lb), observed_default=AT)
        return
    doc = HR.to_ocp(row, flow=flows.load(lb), observed_default=AT) if flow else HR.to_ocp(row)
    assert doc["run"]["id"] and doc["attempts"]


@pytest.mark.parametrize(("site", "flow"), [c for c in CASES if "cost" in c[0]],
                         ids=[i for i in IDS if "cost" in i])
def test_a_negative_cost_a_float_holds_keeps_its_words(site, flow):
    held = HRrows.check_row(_row(site, 0.5, flow))
    got = HRrows.check_row(_row(site, -TOP, flow))
    assert [p for p in got if p not in held] == [f"{site}: a number >= 0 or null"]


@pytest.mark.parametrize("value", [10**400, -10**400], ids=["plus", "minus"])
def test_flow_record_refuses_the_row_by_name_and_writes_nothing(value, capsys, tmp_path, lb, fits):
    rows, home = tmp_path / "rows.jsonl", tmp_path / "home"
    rows.write_text(json.dumps(_row("stages.0.cost.usd", value, True)) + "\n", encoding="utf-8")
    plan = (lb / "plan.json").read_bytes()
    code = cli.main(["flow", "record", str(lb), "--rows", str(rows), "--home", str(home)])
    out, err = capsys.readouterr()
    assert code == 1 and out == "" and err == f"error: {rows}:1: stages.0.cost.usd: {PAST}\n"
    assert _tree(home) == {} and fits == [] and (lb / "plan.json").read_bytes() == plan


@pytest.mark.parametrize("value", [10**400, -10**400], ids=["plus", "minus"])
def test_adapt_refuses_a_row_with_no_flow_by_name_and_writes_nothing(value, capsys, tmp_path):
    rows, out_dir = tmp_path / "rows.jsonl", tmp_path / "ocp"
    rows.write_text(json.dumps(_row("stages.0.cost.usd", value, False)) + "\n", encoding="utf-8")
    code = cli.main(["adapt", "harness-runs", str(rows), "--out", str(out_dir)])
    out, err = capsys.readouterr()
    assert code == 1 and out == ""
    assert err == f"error: adapt harness-runs: {rows}:1: stages.0.cost.usd: {PAST}\n"
    assert _tree(out_dir) == {}
