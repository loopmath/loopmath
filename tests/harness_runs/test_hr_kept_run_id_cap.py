"""The longest run id a row naming a flow keeps (spec/harness-runs.md section 10): every id made from a run id of
MAX_KEPT_RUN_ID characters fits OCP's 200, with up to MAX_ITEMS judge scores and check verdicts. One more
character, score or verdict, or a score name that would push its signal id past 200, is refused by name before
anything is converted or written."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

from loopmath import flows
from loopmath import harness_runs as HR
from loopmath.harness_runs import rows
from loopmath.ocp import conformance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hr_flow import AT, FLOW_FILE, SCHEMA_FILE, _validator, example_rows  # noqa: E402

CAP = HR.MAX_KEPT_RUN_ID
AT_CAP = "run_" + "q" * (CAP - 4)
OVER = "run_" + "q" * (CAP - 3)
CHECKS = [{"id": "extract", "verdict": "pass"}, {"id": "critic", "verdict": "pass"}]


@pytest.fixture(scope="module")
def flow():
    return flows.load(FLOW_FILE)


def _row(rid=AT_CAP, judges=3, checks=0, name="quality", **outcome) -> dict:
    """lit-brief's first example row with `judges` judge scores and `checks` check verdicts, two per extra critic
    stage (a stage reports each check once)."""
    row = example_rows()[0]
    stages, left = list(row["stages"]), checks
    while left:
        take = CHECKS[:min(2, left)]
        stages.append({"role": "reviewer", "piece": "critic", "checks": copy.deepcopy(take)})
        left -= len(take)
    row.update(id=rid, stages=stages,
               outcome={"name": name, "score": 0.5, "judge": "j-claude", **outcome,
                        "judges": [{"id": "j-claude", "score": 0.5}] * judges})
    return row


def test_the_cap_is_200_less_the_longest_addition():
    assert rows.DERIVED_ADDS == (5, 13, 15, 15)
    assert CAP == rows.MAX_ID - max(rows.DERIVED_ADDS) == 185 and HR.MAX_ITEMS == 9999
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    assert schema["allOf"][-1]["then"]["properties"]["id"]["pattern"] == f"^run_[A-Za-z0-9._:-]{{1,{CAP - 4}}}(?![\\s\\S])"
    assert schema["$defs"]["judges"]["maxItems"] == HR.MAX_ITEMS


def test_every_id_from_a_run_id_at_the_cap_is_whole_and_valid(flow):
    row = _row(judges=HR.MAX_ITEMS, checks=HR.MAX_ITEMS, name="q" * 10)  # the longest name that fits at the cap
    assert HR.check_row(row) == [] and not list(_validator().iter_errors(row))
    doc = HR.to_ocp(row, flow=flow, observed_default=AT)
    attempts, signals = [a["id"] for a in doc["attempts"]], [s["id"] for s in doc["run"]["signals"]]
    assert doc["run"]["id"] == AT_CAP and len(attempts) == 4 + HR.MAX_ITEMS // 2 + 1
    assert attempts == [f"att_{AT_CAP[4:]}_{i}" for i in range(1, len(attempts) + 1)]
    assert signals == ([f"sig_{AT_CAP}_{'q' * 10}"] + [f"sig_{AT_CAP}_judge_{i}" for i in range(HR.MAX_ITEMS)]
                       + [f"sig_{AT_CAP}_check_{k}" for k in range(HR.MAX_ITEMS)])
    assert {len(signals[0]), len(signals[HR.MAX_ITEMS]), len(signals[-1])} == {200}  # the score, the last judge
    assert max(map(len, attempts)) == 5 + CAP <= 200                                     # and check
    assert [[f.code for f in found] for found in conformance.validate_many([flow.doc, doc])] == [[], []]
    twin = _row(judges=1, accepted=True, by="j-claude")
    del twin["outcome"]["score"], twin["outcome"]["name"], twin["outcome"]["judge"]
    ids = [s["id"] for s in HR.to_ocp(twin, flow=flow, observed_default=AT)["run"]["signals"]]
    assert ids == [f"sig_{AT_CAP}_accepted", f"sig_{AT_CAP}_judge_0"] and len(ids[0]) == 198


def test_a_score_named_like_a_check_at_the_cap_still_gets_its_own_ids(flow):
    doc = HR.to_ocp(_row(judges=1, checks=2, name="check_0"), flow=flow, observed_default=AT)
    ids = [s["id"] for s in doc["run"]["signals"]]
    assert ids[0] == f"sig_{AT_CAP}_check_0" and ids[2].startswith(f"sig_{AT_CAP}_check_0"[:187] + "-")
    assert len(set(ids)) == len(ids) == 4 and all(len(i) <= 200 for i in ids) and ids[3] == f"sig_{AT_CAP}_check_1"


OVER_CASES = [  # (row, the problem check_row gives, whether the schema refuses it too)
    (_row(OVER), f"id: run id {OVER!r} is {CAP + 1} characters; a row naming a flow keeps a run id of at most "
                 f"{CAP} (MAX_KEPT_RUN_ID), so every id made from it fits 200", True),
    (_row(judges=HR.MAX_ITEMS + 1), "outcome.judges: 10000 scores; a row naming a flow carries at most 9999 "
                                    "(MAX_ITEMS)", True),
    (_row(checks=HR.MAX_ITEMS + 1), f"stages: run {AT_CAP!r} reports 10000 check verdicts; a row naming a flow "
                                    "reports at most 9999 (MAX_ITEMS)", False),
    (_row(name="q" * 11), f"outcome.name: the score's signal id sig_<run id>_<name> is 201 characters for run "
                          f"{AT_CAP!r}; ids take at most 200, so give a shorter name", False),
]


@pytest.mark.parametrize(("row", "problem", "schema"), OVER_CASES, ids=["id", "judges", "checks", "name"])
def test_one_over_is_refused_by_name(row, problem, schema, flow):
    assert HR.check_row(row) == [problem]
    assert bool(list(_validator().iter_errors(row))) is schema
    with pytest.raises(ValueError) as caught:
        HR.to_ocp(row, flow=flow, observed_default=AT)
    assert str(caught.value) == problem
    with pytest.raises(ValueError, match=r"^row 2: "):
        HR.to_ocp_rows([_row("run_q07-ok"), row], flow=flow, observed_default=AT)


@pytest.mark.parametrize(("row", "problem", "schema"), OVER_CASES, ids=["id", "judges", "checks", "name"])
def test_a_refused_row_is_never_written(row, problem, schema, tmp_path):
    log, missing = tmp_path / "rows.jsonl", tmp_path / "new.jsonl"
    HR.append(log, _row())
    before = log.read_bytes()
    for path in (log, missing):
        with pytest.raises(ValueError, match="^" + re.escape(problem)):
            HR.append(path, row)
    assert log.read_bytes() == before and not missing.exists()


def test_bad_neighbours_give_named_problems_not_errors():
    row = _row(checks=HR.MAX_ITEMS + 1, name="q" * 11)
    n = len(row["stages"])
    row["stages"] += ["not a stage", {"role": "reviewer", "piece": "critic", "checks": "extract"}]
    row["outcome"]["judges"] = "j-claude"
    problems = HR.check_row(row)
    assert f"stages: run {AT_CAP!r} reports 10000 check verdicts; a row naming a flow reports at most 9999 " \
           "(MAX_ITEMS)" in problems
    assert any(p.startswith("outcome.name: ") for p in problems)
    assert {p.split(":")[0] for p in problems} >= {f"stages.{n}", f"stages.{n + 1}.checks", "outcome.judges"}
    over = _row(OVER, name=["q"])
    assert {p.split(":")[0] for p in HR.check_row(over)} == {"id", "outcome.name"}  # the name, not its length
