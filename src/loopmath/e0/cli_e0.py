"""E0 corpus walkdown, preserved verbatim as `loopmath analyze-e0`.

The E0 estimator is the ancestor of surface.py; this verb keeps the original
walkdown reproducible against the E0 corpus (research.e0_corpus). Behavior is
frozen: do not change the numbers it prints.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .arms import attach_run_topology, build_attempt_table, build_session_table
from .estimate import (
    arm_overlap_pairs,
    attempts_per_done_task,
    bootstrap_S,
    naive_arm_estimates,
    shrunken_arm_estimates,
    spread_S,
    unknown_brackets,
    within_cell_anchor,
)
from .io import (
    fleet_ledger,
    load_corpus,
    segment_sessions,
    subagent_token_fold,
)
from .figures import (
    bootstrap_figure,
    forest_figure,
    markdown_table,
    pareto_figure,
    rework_figure,
    write_report,
)

# SPEC section 0 bans the experiment jargon from user-facing output. The E0
# code is preserved verbatim and its internal column names stay as they are,
# so the rename happens once, here, at the moment the data becomes something
# a person reads.
_DISPLAY_COLUMNS = {
    "arm": "configuration",
    "arm_a": "configuration a",
    "arm_b": "configuration b",
    "n_arms": "n_configurations",
    "arm_set": "configuration_set",
}



def analyze(args: argparse.Namespace) -> int:
    from ..research_paths import ResearchPathError

    try:
        corpus = load_corpus(args.corpus)
    except ResearchPathError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    segments = segment_sessions(corpus["sessions"])
    fold, fold_diag = subagent_token_fold(segments)
    ledger = fleet_ledger(segments["fleet_stubs"])
    print(
        f"sessions: {len(corpus['sessions'])} total, "
        f"{len(segments['main'])} real main, "
        f"{len(segments['fleet_stubs'])} fleet stubs, "
        f"{len(segments['subagents'])} subagent transcripts"
    )
    print(
        f"subagent fold: {fold_diag['folded_tokens']:,} output tokens into "
        f"{len(fold)} parents; {fold_diag['unmatched_files']} transcripts "
        f"({fold_diag['unmatched_tokens']:,} tokens) have no real-main parent"
    )
    print(
        f"fleet ledger: {ledger['n_stubs']} stubs, "
        f"{ledger['wallclock_min']:.1f} min recorded wall-clock "
        f"(median {ledger['median_s']:.1f} s, {ledger['n_over_60s']} over 60 s), "
        f"dates {ledger['dates']}"
    )

    session_df = build_session_table(segments["main"], extra_output_tokens=fold)
    multi = int(session_df["multi_effort"].sum()) if not session_df.empty else 0
    print(f"multi-effort sessions (dominant label blurs them): {multi}")

    naive = naive_arm_estimates(session_df, min_n=args.min_n)
    anchor = within_cell_anchor(session_df, min_n=args.min_n)
    shrunk = shrunken_arm_estimates(session_df, min_n=args.min_n)
    overlap_pairs = arm_overlap_pairs(session_df, min_n=args.min_n)
    boot = bootstrap_S(session_df, n_boot=args.boot, seed=args.seed, min_n=args.min_n)
    brackets = unknown_brackets(session_df, min_n=args.min_n)

    # sensitivity: original frozen cell definition including week
    week_est = shrunken_arm_estimates(session_df, min_n=args.min_n, cell_col="cell_week")
    week_sensitivity = {
        "S_point": spread_S(week_est),
        "n_arms": 0 if week_est.empty else int(week_est["arm"].nunique()),
        "note": (
            "week is aliased with arm inside the video family, so this variant "
            "loses the opus5 medium vs xhigh contrast"
        ),
    }

    attempt_df = build_attempt_table(corpus["dag_attempts"])
    attempt_df = attach_run_topology(attempt_df, corpus["dag_runs"])
    excluded = Counter(
        str(m) for m in attempt_df.loc[attempt_df["arm"].isna(), "raw_model"]
    )
    print(
        f"dag attempts: {len(attempt_df)}, excluded from the configuration tables "
        f"(no canonical label): {sum(excluded.values())} = "
        + ", ".join(f"{k}: {v}" for k, v in sorted(excluded.items()))
    )
    apdt, dag_pair = attempts_per_done_task(attempt_df, n_boot=args.boot, seed=args.seed)

    # merge per-arm bootstrap intervals into the normalized table
    if not shrunk.empty and not boot["arm_intervals"].empty:
        shrunk = shrunk.merge(boot["arm_intervals"], on="arm", how="left")

    print("\n== Within-cell matched anchor (design step 1, model-free) ==")
    print(markdown_table(anchor.rename(columns=_DISPLAY_COLUMNS)))

    print(
        "\n== Headline: joint workflow-configuration and cell fit, shrunken, "
        "mix-standardized tokens-per-accepted (proxy grade) =="
    )
    if not shrunk.empty:
        cols = [c for c in ["arm", "n", "n_overlap_cells", "acc_rate", "shrink",
                            "R_unshrunken", "R", "R_lo90", "R_hi90"] if c in shrunk.columns]
        # `arm` is the internal column name and stays that way in the data;
        # what a reader sees is renamed, because a printed header is
        # user-facing output and the vocabulary rule applies to it.
        print(markdown_table(shrunk[cols].rename(columns=_DISPLAY_COLUMNS)))
    else:
        print("(no eligible workflow configurations at this --min-n)")
    print(
        f"\nS = {boot['S_point']:.2f}  90% CI [{boot['S_lo90']:.2f}, {boot['S_hi90']:.2f}]  "
        f"P(S>10) = {boot['P_S_gt_10']:.4f} ({boot.get('n_gt_10', 0)}/{boot['n_boot_valid']} draws "
        f"above 10, max draw {boot.get('S_draw_max', float('nan')):.2f}, seed {boot['seed']})"
    )
    if "pair_ratio_point" in boot:
        print(
            f"prespecified pair {boot['pair'][1]} / {boot['pair'][0]}: "
            f"{boot['pair_ratio_point']:.2f} "
            f"[{boot.get('pair_ratio_lo90', np.nan):.2f}, {boot.get('pair_ratio_hi90', np.nan):.2f}]"
        )
    print(
        f"unknown-proxy brackets: S = {brackets['S_unknown_accepted']:.2f} (all accepted) / "
        f"{brackets['S_unknown_rejected']:.2f} (all not accepted)"
    )
    print(
        f"week-in-cell sensitivity: S = {week_sensitivity['S_point']:.2f} "
        f"on {week_sensitivity['n_arms']} workflow configurations"
    )
    print(
        "\n== Overlap diagnostic (shared overlap cells per eligible "
        "pair of workflow configurations) =="
    )
    print(markdown_table(overlap_pairs.rename(columns=_DISPLAY_COLUMNS)))
    zero = overlap_pairs[overlap_pairs["shared_cells"] == 0] if not overlap_pairs.empty else overlap_pairs
    for _, row in zero.iterrows():
        print(
            f"model-only comparison: {row['arm_a']} vs {row['arm_b']} "
            "share no overlap cell"
        )
    print("\n== Bootstrap configuration-set diagnostic ==")
    print(markdown_table(boot["arm_set_diag"].rename(columns=_DISPLAY_COLUMNS)))

    print("\n== Attempts per done task (gate grade) ==")
    print(markdown_table(apdt.rename(columns=_DISPLAY_COLUMNS)))
    print(
        f"prespecified dag pair {dag_pair['label']}: {dag_pair['ratio']:.2f} "
        f"[{dag_pair.get('lo90', float('nan')):.2f}, {dag_pair.get('hi90', float('nan')):.2f}]"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    forest_figure(shrunk, out_dir, "forest",
                  "Tokens per accepted session, normalized, per arm (proxy grade)")
    pareto_figure(shrunk, naive, out_dir, "pareto",
                  "Cost vs acceptance per arm (proxy grade)")
    rework_figure(apdt, out_dir, "rework",
                  "Attempts per done task, per arm (gate grade, dag slice)")
    bootstrap_figure(boot, out_dir, "bootstrap_s", "Bootstrap distribution of S")

    # csv side-tables for downstream use
    naive.to_csv(out_dir / "naive_arms.csv", index=False)
    shrunk.to_csv(out_dir / "normalized_arms.csv", index=False)
    apdt.to_csv(out_dir / "attempts_per_done_task.csv", index=False)
    anchor.to_csv(out_dir / "within_cell_anchor.csv", index=False)

    report_path = write_report(
        out_dir, segments, fold_diag, naive, shrunk, boot, brackets,
        week_sensitivity, apdt, dag_pair, args.min_n,
        anchor=anchor, overlap_pairs=overlap_pairs, ledger=ledger,
        excluded_labels=dict(excluded),
    )
    print(f"\nwrote {report_path} and figures (png+svg) to {out_dir}")
    return 0


from .verb import register  # noqa: E402,F401  (the verb registers without importing this module)
