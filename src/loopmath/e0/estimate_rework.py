"""Attempt rework estimates for the E0 analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Prespecified pairwise guard (design section 5, named before the run).
DAG_PAIR = ("opus5·max", "opus5·xhigh")

# Attempt causes that mark failure-driven rework, as opposed to followup
# attempts (scope extensions and new instructions on an already-accepted
# task). Both shares are reported; conflating them overstates failure churn.
FAILURE_CAUSES = ("sent_back", "gate_failed")

_APDT_COLS = [
    "arm", "attempts", "tasks", "done_tasks", "followup_share",
    "failure_rework_share", "any_rework_share", "attempts_per_done_task",
    "apdt_lo90", "apdt_hi90",
]


def attempts_per_done_task(
    attempt_df: pd.DataFrame,
    min_attempts: int = 15,
    n_boot: int = 2000,
    seed: int = 20260830,
    pair: tuple[str, str] = DAG_PAIR,
) -> tuple[pd.DataFrame, dict]:
    """Gate-grade rework metric on the dag slice (join-free, primary).

    A task is credited to an arm through that arm's attempts; it counts as
    done when the arm's terminal (highest-n) attempt on the task is accepted
    (outcome_result == done, evidence verified/reported). This is an arm-level
    attribution rule: on tasks with attempts from several arms it can differ
    from the design 1c task-global terminal rule (8 of 926 task-arm pairs on
    this corpus), and each arm counts only its own attempts as cost, so
    unlabeled attempts and other arms' attempts on shared tasks are NOT in an
    arm's cost (on this corpus all 42 unlabeled attempts sit on tasks with no
    labeled attempts, so nothing is dropped; the diagnostic is returned).
    Open work (null outcome or ended_at) is excluded from done counts but its
    attempts still count as cost.

    Rework decomposition: followup_share counts cause == followup (scope
    extensions on accepted work); failure_rework_share counts sent_back and
    gate_failed. any_rework_share is their union plus other non-initial
    causes. Conflating followups with failures inverts the arm ranking, so
    both shares print.

    Intervals: the design-frozen cluster bootstrap over runs, implemented as
    a two-stage Bayesian bootstrap (Dirichlet weights over runs, then over
    tasks within run, shared across arms per draw), which keeps run
    clustering without degenerate 10-cluster resamples. The prespecified pair
    ratio gets its own interval from the shared draws. Seed fixed.
    """
    rng = np.random.default_rng(seed)
    df = attempt_df.copy()
    labeled = df[df["arm"].notna()]

    # diagnostic: unlabeled attempts sharing a task with labeled attempts
    labeled_tasks = set(map(tuple, labeled[["run_id", "task_id"]].drop_duplicates().itertuples(index=False)))
    unlabeled = df[df["arm"].isna()]
    n_unlabeled_on_labeled_tasks = int(
        sum(1 for t in unlabeled[["run_id", "task_id"]].itertuples(index=False) if tuple(t) in labeled_tasks)
    )

    # per-arm task table: (run_id, task_key, n_attempts, done)
    arm_tasks: dict[str, list[tuple[str, tuple, int, bool]]] = {}
    arm_meta = []
    for arm, g in labeled.groupby("arm"):
        if len(g) < min_attempts:
            continue
        tasks = []
        for (run_id, task_id), t in g.groupby(["run_id", "task_id"]):
            terminal = t.loc[t["n"].idxmax()]
            tasks.append((run_id, (run_id, task_id), len(t), bool(terminal["accepted"])))
        arm_tasks[arm] = tasks
        n_tasks = len(tasks)
        done = sum(1 for *_, d in tasks if d)
        causes = g["cause"]
        arm_meta.append(
            {
                "arm": arm,
                "attempts": len(g),
                "tasks": n_tasks,
                "done_tasks": done,
                "followup_share": float((causes == "followup").mean()),
                "failure_rework_share": float(causes.isin(FAILURE_CAUSES).mean()),
                "any_rework_share": float((~causes.isin([None, "initial"])).mean()),
                "attempts_per_done_task": len(g) / done if done else np.nan,
            }
        )
    if not arm_meta:
        return pd.DataFrame(columns=_APDT_COLS), {"ratio": float("nan")}

    # two-stage Bayesian bootstrap, weights shared across arms per draw
    runs = sorted({r for tasks in arm_tasks.values() for r, *_ in tasks})
    all_task_keys = sorted({k for tasks in arm_tasks.values() for _, k, *_ in tasks})
    run_idx = {r: i for i, r in enumerate(runs)}
    task_idx = {k: i for i, k in enumerate(all_task_keys)}
    arm_arrays = {
        arm: (
            np.array([run_idx[r] for r, *_ in tasks]),
            np.array([task_idx[k] for _, k, *_ in tasks]),
            np.array([na for *_, na, _ in tasks], dtype=float),
            np.array([d for *_, d in tasks], dtype=float),
        )
        for arm, tasks in arm_tasks.items()
    }
    boot_vals: dict[str, list[float]] = {arm: [] for arm in arm_arrays}
    pair_vals: list[float] = []
    for _ in range(n_boot):
        g_run = rng.exponential(1.0, size=len(runs))
        g_task = rng.exponential(1.0, size=len(all_task_keys))
        draw = {}
        for arm, (ri, ti, natt, done) in arm_arrays.items():
            w = g_run[ri] * g_task[ti]
            denom = float(np.sum(w * done))
            draw[arm] = float(np.sum(w * natt)) / denom if denom > 0 else np.nan
            if np.isfinite(draw[arm]):
                boot_vals[arm].append(draw[arm])
        if (
            pair[0] in draw
            and pair[1] in draw
            and np.isfinite(draw[pair[0]])
            and np.isfinite(draw[pair[1]])
            and draw[pair[0]] > 0
        ):
            pair_vals.append(draw[pair[1]] / draw[pair[0]])

    for row in arm_meta:
        vals = boot_vals.get(row["arm"], [])
        row["apdt_lo90"] = float(np.percentile(vals, 5)) if vals else np.nan
        row["apdt_hi90"] = float(np.percentile(vals, 95)) if vals else np.nan

    table = (
        pd.DataFrame(arm_meta)
        .sort_values("attempts", ascending=False)
        .reset_index(drop=True)
    )
    a = table.set_index("arm")["attempts_per_done_task"]
    pair_info = {
        "label": f"{pair[1]} / {pair[0]} attempts-per-done-task",
        "ratio": (
            float(a[pair[1]] / a[pair[0]])
            if pair[0] in a.index and pair[1] in a.index and a[pair[0]] > 0
            else float("nan")
        ),
        "lo90": float(np.percentile(pair_vals, 5)) if pair_vals else float("nan"),
        "hi90": float(np.percentile(pair_vals, 95)) if pair_vals else float("nan"),
        "n_unlabeled_on_labeled_tasks": n_unlabeled_on_labeled_tasks,
    }
    return table, pair_info
