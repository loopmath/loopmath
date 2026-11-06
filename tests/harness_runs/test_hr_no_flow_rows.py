"""Rows that name no flow read as before rows could name one (spec/harness-runs.md section 10): a row whose `flow`
is absent, null or not an object keeps its `flow`, `slate`, `outcome.judges` and stage `checks` as unknown
members. Each fixture row converts to the document loopmath 0.3.5 wrote for it, byte for byte."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from loopmath import flows
from loopmath import harness_runs as HR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hr_flow import FLOW_FILE, _validator  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "harness_runs"
ROWS = FIXTURES / "no-flow-rows.jsonl"
# One document per row, written by loopmath 0.3.5 (a git archive of its tag, with its rows schema accepting each
# row): to_ocp(row, observed_default=WHEN), compact JSON in key order, producer.version as <version>
WANT = FIXTURES / "no-flow-rows.0.3.5.jsonl"
WHEN = "2026-11-03T12:00:00-08:00"
COUNT = 57
KINDS = {str: "a string", bool: "a boolean", int: "a number", list: "a list"}


def _rows() -> list[dict]:
    return HR.read(ROWS)


def _want() -> list[str]:
    return [line for line in WANT.read_text(encoding="utf-8").splitlines() if line.strip()]


def _bytes(doc: dict) -> str:
    doc = copy.deepcopy(doc)
    doc["producer"]["version"] = "<version>"
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


@pytest.fixture(scope="module")
def flow():
    return flows.load(FLOW_FILE)


def test_the_rows_hold_each_member_its_null_and_each_kind_of_flow():
    rows = _rows()
    assert len(rows) == len(_want()) == COUNT
    assert not any(isinstance(r.get("flow"), dict) for r in rows)
    assert {type(r["flow"]) for r in rows if "flow" in r} == {type(None), str, int, list, bool}
    assert {type(r["slate"]) for r in rows if "slate" in r} == {type(None), str, list, dict}
    judges = [r["outcome"]["judges"] for r in rows if isinstance(r.get("outcome"), dict) and "judges" in r["outcome"]]
    assert {type(j) for j in judges} == {type(None), str, list} and [] in judges
    checks = [s["checks"] for r in rows for s in r.get("stages") or [] if "checks" in s]
    assert {type(c) for c in checks} == {type(None), str, list} and [] in checks


@pytest.mark.parametrize("n", range(COUNT))
def test_a_row_without_flow_gives_the_0_3_5_document(n):
    row, want = _rows()[n], _want()[n]
    assert HR.check_row(row) == [] and not list(_validator().iter_errors(row))
    assert _bytes(HR.to_ocp(row, observed_default=WHEN)) == want


def test_one_conversion_of_the_rows_gives_the_same_documents():
    assert [_bytes(d) for d in HR.to_ocp_rows(_rows(), observed_default=WHEN)] == _want()


def test_a_flow_that_is_not_an_object_gives_one_finding_and_the_same_bytes(flow):
    seen = 0
    for row, want in zip(_rows(), _want()):
        found = HR.flow_row_findings(row, flow)
        if row.get("flow") is None:
            assert found == []
        else:
            seen += 1
            assert found == [f"flow: {KINDS[type(row['flow'])]}, not an object {{id, version, arm, plan, split}}; "
                             "the row was converted as a row without a flow"]
        assert _bytes(HR.to_ocp(row, flow=flow, observed_default=WHEN)) == want  # a flow given changes nothing
    assert seen == 12


@pytest.mark.parametrize("flow_member", ["absent", None, "lit-brief"])
def test_judges_alone_are_not_an_outcome_without_a_flow(flow_member):
    row = {"schema": HR.SCHEMA, "task_id": "q07", "outcome": {"judges": [{"id": "j-claude", "score": 0.7}]}}
    if flow_member != "absent":
        row["flow"] = flow_member
    assert HR.check_row(row) == ["outcome: exactly one of score or accepted"]
    assert list(_validator().iter_errors(row))
