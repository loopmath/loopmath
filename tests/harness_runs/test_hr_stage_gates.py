"""The gates of a row without a flow keep their bytes (spec/harness-runs.md section 10): each gate a stage list
implies gets its default id `g_<piece>`, so the document stays in the OCP 0.3 gates form. The rows hold piece ids
named like the chain's derived ids, ids long enough to be cut, send-backs and a derived run id; each converts to
the document loopmath 0.3.5 wrote for it, byte for byte."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from loopmath import harness_runs as HR

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "harness_runs"
ROWS = FIXTURES / "stage-gate-rows.jsonl"
# One document per row, written by loopmath 0.3.5 (a git archive of its tag, with its rows schema accepting each
# row): to_ocp(row, observed_default=WHEN), compact JSON in key order, producer.version as <version>
WANT = FIXTURES / "stage-gate-rows.base.jsonl"
WHEN = "2026-11-03T12:00:00-08:00"
COUNT = 18


def _rows() -> list[dict]:
    return HR.read(ROWS)


def _want() -> list[str]:
    return [line for line in WANT.read_text(encoding="utf-8").splitlines() if line.strip()]


def _bytes(doc: dict) -> str:
    doc = copy.deepcopy(doc)
    doc["producer"]["version"] = "<version>"
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


def test_the_rows_name_no_flow_and_reach_clashing_and_long_gate_ids():
    rows, want = _rows(), [json.loads(line) for line in _want()]
    assert len(rows) == len(want) == COUNT and not any("flow" in r for r in rows)
    workflows = [d["run"]["configuration"]["workflow"] for d in want]
    assert {d["ocp"] for d in want} == {"0.3"} and not any("checks" in w["control"] for w in workflows)
    clash = [w for w in workflows if any(f"g_{g}" in {p["id"] for p in w["pieces"]} for g in w["control"]["gates"])]
    long = [w for w in workflows if any(len(f"g_{g}") > 200 for g in w["control"]["gates"])]
    assert len(clash) == 5 and len(long) == 1  # a default id a piece holds; one past OCP's 200 characters


@pytest.mark.parametrize("n", range(COUNT))
def test_a_row_gives_the_0_3_5_document(n):
    assert _bytes(HR.to_ocp(_rows()[n], observed_default=WHEN)) == _want()[n]


def test_one_conversion_of_the_rows_gives_the_same_documents():
    assert [_bytes(d) for d in HR.to_ocp_rows(_rows(), observed_default=WHEN)] == _want()
