import builtins
import dis
from types import CodeType

import numpy as np

from loopmath.e0 import estimate, estimate_rework
from loopmath.e0.arms import build_attempt_table, build_session_table
from loopmath.e0.estimate import (
    attempts_per_done_task,
    bootstrap_S,
    naive_arm_estimates,
    shrunken_arm_estimates,
    spread_S,
)
from loopmath.e0.io import segment_sessions


def _df(fake_sessions):
    return build_session_table(segment_sessions(fake_sessions)["main"])


def test_attempts_per_done_task_facade_has_moved_function_globals():
    def loaded_globals(code):
        names = {
            instruction.argval
            for instruction in dis.get_instructions(code)
            if instruction.opname == "LOAD_GLOBAL"
        }
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                names.update(loaded_globals(constant))
        return names

    dependencies = loaded_globals(attempts_per_done_task.__code__) - set(dir(builtins))
    missing = dependencies - vars(estimate).keys()
    assert not missing, (
        "attempts_per_done_task is bound to loopmath.e0.estimate globals, but these "
        f"non-builtin LOAD_GLOBAL dependencies exist only in estimate_rework: {sorted(missing)}"
    )

    mismatched = {
        name
        for name in dependencies
        if name in vars(estimate_rework)
        and vars(estimate)[name] is not vars(estimate_rework)[name]
    }
    assert not mismatched, (
        "attempts_per_done_task dependencies must resolve to identical objects in "
        f"estimate and estimate_rework; mismatches: {sorted(mismatched)}"
    )


def test_naive_estimates(fake_sessions):
    df = _df(fake_sessions)
    est = naive_arm_estimates(df, min_n=2)
    est = est.set_index("arm")
    assert est.loc["opus5·medium", "n"] == 4
    assert est.loc["opus5·medium", "accepted"] == 4
    # 1000+1001+1002+1003 tokens over 4 accepts, but min 5 accepts for the
    # per-accepted number, so it is NaN on this tiny fixture
    assert np.isnan(est.loc["opus5·medium", "naive_tokens_per_accepted"])
    assert est.loc["fable5·xhigh", "accepted"] == 2


def test_shrunken_estimates_and_spread(fake_sessions):
    df = _df(fake_sessions)
    est = shrunken_arm_estimates(df, min_n=2, min_overlap_cells=1)
    assert set(est["arm"]) == {"opus5·medium", "fable5·xhigh"}
    # the expensive arm must stay more expensive after shrinkage
    est = est.set_index("arm")
    assert est.loc["fable5·xhigh", "R"] > est.loc["opus5·medium", "R"]
    # shrinkage factor is in [0, 1]
    assert ((est["shrink"] >= 0) & (est["shrink"] <= 1)).all()
    s = spread_S(est.reset_index())
    assert s > 1


def test_spread_excludes_trivial_low_arms(fake_sessions):
    import pandas as pd

    est = pd.DataFrame(
        {"arm": ["a·low", "b·max", "c·medium"], "R": [1.0, 50.0, 100.0]}
    )
    assert spread_S(est) == 2.0  # a·low excluded, else it would be 100x


def test_bootstrap_runs(fake_sessions):
    df = _df(fake_sessions)
    boot = bootstrap_S(df, n_boot=20, seed=1, min_n=2, min_overlap_cells=1)
    assert boot["n_boot_valid"] > 0
    assert boot["S_point"] > 1
    assert 0 <= boot["P_S_gt_10"] <= 1


def test_attempts_per_done_task(fake_attempts):
    df = build_attempt_table(fake_attempts)
    table, pair = attempts_per_done_task(df, min_attempts=2, n_boot=50)
    apdt = table.set_index("arm")
    # opus5·medium: 4 attempts over 3 tasks, all 3 terminal-done
    assert apdt.loc["opus5·medium", "attempts"] == 4
    assert apdt.loc["opus5·medium", "done_tasks"] == 3
    assert np.isclose(apdt.loc["opus5·medium", "attempts_per_done_task"], 4 / 3)
    # fable5·xhigh: 3 attempts (canonicalized from 3 spellings), 2 done tasks
    assert apdt.loc["fable5·xhigh", "attempts"] == 3
    assert apdt.loc["fable5·xhigh", "done_tasks"] == 2
    # rework decomposition: opus5·medium has 1 sent_back of 4 attempts
    assert np.isclose(apdt.loc["opus5·medium", "failure_rework_share"], 0.25)
    assert np.isclose(apdt.loc["opus5·medium", "followup_share"], 0.0)
    # null/dash attempts share no task with labeled attempts in this fixture
    assert pair["n_unlabeled_on_labeled_tasks"] == 0


def test_attempts_per_done_task_empty(fake_attempts):
    df = build_attempt_table(fake_attempts)
    table, pair = attempts_per_done_task(df, min_attempts=99, n_boot=10)
    assert table.empty
    assert np.isnan(pair["ratio"])


def test_naive_estimates_empty(fake_sessions):
    df = _df(fake_sessions)
    est = naive_arm_estimates(df, min_n=99)
    assert est.empty
    assert "arm" in est.columns
