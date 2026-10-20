"""0.3.1: `check-change` counts take the singular for 1 ("1 pair", "1 candidate run"), through one helper.

The reasons name only the arms that have runs ("1 candidate run has no partner"), and a $0 cost is described
without the model's internal amount. No verdict or JSON field changes.
"""

from __future__ import annotations

import json

from loopmath.compare import Observation, paired
from tests.compare.test_check_change_cli import SOLO, home, lm, tasks_file  # noqa: F401  (home is a fixture)


def test_counted_and_arm_runs():
    from loopmath.compare import arm_runs
    from loopmath.output import counted

    assert (counted(1, "pair"), counted(0, "run"), counted(2, "candidate run")) == ("1 pair", "0 runs",
                                                                                     "2 candidate runs")
    assert counted(1, "match", "matches") == "1 match" and counted(3, "match", "matches") == "3 matches"
    assert arm_runs(0, 1) == "0 base and 1 candidate run" and arm_runs(3, 3) == "3 base and 3 candidate runs"
    assert arm_runs(0, 1, nonzero=True) == "1 candidate run"
    assert arm_runs(2, 1, nonzero=True) == "2 base runs and 1 candidate run"


def test_start_and_status_with_one_pair(home, tmp_path, capsys):
    code, out, err = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                        str(tasks_file(tmp_path, 1)))  # one task, one pair
    assert code == 0, err
    head = out.splitlines()[0]
    assert ", 1 pair, " in head and "1 pairs" not in out
    code, out, err = lm(capsys, "check-change", "status")
    assert "0 base and 0 candidate runs of 1 pair" in out.splitlines()[0] and "1 pairs" not in out


def test_from_runs_headline_with_one_run_each(home, tmp_path, capsys):
    task = tmp_path / "t.json"
    task.write_text(json.dumps({"id": "tsk_hand00", "type": "bug_fix", "repo": "acme/app"}))
    for label in ("old", "new"):
        code, _, err = lm(capsys, "run", "record", "--task-file", str(task), *SOLO, "--label", f"config={label}",
                          "--cwd", f"implement={tmp_path / 'wt'}", "--since", "1h", "--verified", "tests=pass",
                          "--no-commits", "--no-fit", "--json")
        assert code == 0, err
    code, out, err = lm(capsys, "check-change", "from-runs", "--base-label", "config=old", "--cand-label", "config=new")
    assert out.splitlines()[0].startswith("config=new (1 run) against config=old (1 run), "), out


def _obs(arm: str, tasks: list[str], usd: float = 0.5, failed: tuple[str, ...] = ()) -> list[Observation]:
    return [Observation(t, arm, t not in failed, usd, run=f"{arm}-{t}", failed=t in failed) for t in tasks]


def test_reasons_name_one_run_in_the_singular():
    base = _obs("base", ["t1", "t2", "t3"])
    cand = _obs("cand", ["t1", "t2", "t3", "t4"], failed=("t2",))
    cand[0] = Observation("t1", "cand", True, 0.0, run="cand-t1")
    c = paired(base, cand, rule="tests", seed=1)
    said = "\n".join(c.reasons)
    assert "1 candidate run has no partner" in said
    assert "1 run costs $0; a $0 cost is counted as a tiny positive amount" in said and "1e-06" not in said
    assert "1 candidate run failed; a failed run counts as not accepted" in said
    assert "0 base" not in said


def test_reasons_with_runs_on_both_arms_keep_the_plural():
    base = _obs("base", ["t1", "t2", "t3", "t5"], failed=("t1",))
    cand = _obs("cand", ["t1", "t2", "t3", "t4"], failed=("t1", "t2"))
    said = "\n".join(paired(base, cand, rule="tests", seed=1).reasons)
    assert "1 base run and 1 candidate run have no partner" in said
    assert "1 base run and 2 candidate runs failed" in said
