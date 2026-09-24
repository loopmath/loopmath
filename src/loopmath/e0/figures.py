"""Markdown table and figure generation for `loopmath analyze`.

Paper-bound prose rules apply to generated text: no em-dashes, plain
declarative sentences, abstract first, honest numbers.

Figures commit to a single light look (paper-bound static images). Colors
come from the validated dataviz reference palette: categorical slot 1 blue
#2a78d6 for the primary series, neutral inks for text and the deliberately
recessive ghost layer (identity carried by marker shape and labels, never
color alone). Every figure saves as both PNG and SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"
GHOST = "#8a8987"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _save(fig, out_dir: Path, stem: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    """Plain markdown table without extra dependencies."""
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "-"
            return floatfmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def forest_figure(est: pd.DataFrame, out_dir: Path, stem: str, title: str) -> None:
    """Normalized cost per arm with 90% intervals; unadjusted as ghosts (F2)."""
    e = est.dropna(subset=["R"]).sort_values("R")
    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.7 * len(e) + 1.2)))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    y = np.arange(len(e))
    has_ci = {"R_lo90", "R_hi90"}.issubset(e.columns)
    if has_ci:
        for yi, (_, row) in zip(y, e.iterrows()):
            if np.isfinite(row.get("R_lo90", np.nan)):
                ax.plot([row["R_lo90"], row["R_hi90"]], [yi, yi], color=BLUE, lw=2, alpha=0.5,
                        solid_capstyle="round", zorder=2)
    ax.scatter(e["R_unshrunken"], y, marker="o", facecolors="none", edgecolors=GHOST,
               s=42, label="joint-fit, unshrunken", zorder=3)
    ax.scatter(e["R"], y, marker="D", color=BLUE, s=46,
               label="shrunken, mix-standardized (90% interval)", zorder=4)
    for yi, (_, row) in zip(y, e.iterrows()):
        ax.annotate(f"{row['R']:,.0f}", (row["R"], yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(e["arm"], color=INK, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("tokens per accepted session (log scale)", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    leg = ax.legend(fontsize=8, loc="lower right", frameon=False)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def pareto_figure(
    est: pd.DataFrame, naive: pd.DataFrame, out_dir: Path, stem: str, title: str
) -> None:
    """Cost vs acceptance Pareto scatter (F7 flavor).

    Eligible arms: shrunken, mix-standardized R (blue, sized by n).
    Trivial low arms: naive values, ghost markers, labeled as unadjusted.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    e = est.dropna(subset=["R"])
    sizes = 30 + 200 * (e["n"] / max(e["n"].max(), 1))
    ax.scatter(e["acc_rate"], e["R"], s=sizes, color=BLUE, alpha=0.85, zorder=4,
               label="eligible arms, normalized (size = n)")
    low = naive[naive["arm"].str.endswith("low") & naive["naive_tokens_per_accepted"].notna()]
    if not low.empty:
        ax.scatter(low["acc_rate"], low["naive_tokens_per_accepted"], s=40,
                   facecolors="none", edgecolors=GHOST, zorder=3,
                   label="trivial-task low arms, unadjusted")
    ax.set_yscale("log")
    # direct labels with greedy top-down collision avoidance in axes-fraction
    # space, so near-coincident points (opus5 xhigh vs max vs sol, the two low
    # arms) get vertically separated labels
    pts = [{"x": r["acc_rate"], "y": r["R"], "label": r["arm"], "ink": INK}
           for _, r in e.iterrows()]
    pts += [{"x": r["acc_rate"], "y": r["naive_tokens_per_accepted"],
             "label": r["arm"], "ink": INK2} for _, r in low.iterrows()]
    x0, x1 = ax.get_xlim()
    ly0, ly1 = np.log10(ax.get_ylim())
    placed: list[tuple[float, float]] = []
    for p in sorted(pts, key=lambda q: -q["y"]):
        xf = (p["x"] - x0) / (x1 - x0)
        yf = (np.log10(max(p["y"], 1)) - ly0) / (ly1 - ly0)
        target = yf
        for qx, qy in placed:
            if abs(qx - xf) < 0.14 and target > qy - 0.042:
                target = qy - 0.042
        placed.append((xf, target))
        ax.annotate(p["label"], (p["x"], p["y"]), textcoords="offset points",
                    xytext=(9, 4 + (target - yf) * 290), fontsize=8.5, color=p["ink"])
    ax.set_xlabel("acceptance rate (structural proxy, known outcomes)", color=INK2, fontsize=9)
    ax.set_ylabel("tokens per accepted session (log scale)", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    leg = ax.legend(fontsize=8, loc="center left", frameon=False)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def rework_figure(apdt: pd.DataFrame, out_dir: Path, stem: str, title: str) -> None:
    """DAG-slice rework chart: attempts per done task with 90% intervals."""
    e = apdt.dropna(subset=["attempts_per_done_task"]).sort_values("attempts_per_done_task")
    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.7 * len(e) + 1.2)))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    y = np.arange(len(e))
    ax.barh(y, e["attempts_per_done_task"], height=0.55, color=BLUE, alpha=0.9, zorder=3)
    if {"apdt_lo90", "apdt_hi90"}.issubset(e.columns):
        ax.errorbar(e["attempts_per_done_task"], y,
                    xerr=[e["attempts_per_done_task"] - e["apdt_lo90"],
                          e["apdt_hi90"] - e["attempts_per_done_task"]],
                    fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4)
    for yi, (_, row) in zip(y, e.iterrows()):
        ax.annotate(
            f"{row['attempts_per_done_task']:.2f}  ({row['attempts']} attempts, "
            f"{row['failure_rework_share']:.0%} failure rework, "
            f"{row['followup_share']:.0%} followup)",
            (max(row.get("apdt_hi90", row["attempts_per_done_task"]),
                 row["attempts_per_done_task"]), yi),
            textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK2)
    ax.axvline(1.0, color=INK2, lw=0.9, ls=":", zorder=2)
    ax.annotate("1.0 = no rework", (1.0, len(e) - 0.3), fontsize=7.5, color=INK2,
                textcoords="offset points", xytext=(4, 0))
    ax.set_yticks(y)
    ax.set_yticklabels(e["arm"], color=INK, fontsize=10)
    ax.set_xlabel("attempts per done task (gate grade, 90% task-bootstrap interval)",
                  color=INK2, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    _save(fig, out_dir, stem)


def bootstrap_figure(boot: dict, out_dir: Path, stem: str, title: str) -> None:
    """Histogram of bootstrap S draws with the 10x line (F4)."""
    draws = boot.get("draws")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if draws is not None and len(draws):
        ax.hist(draws, bins=40, color=BLUE, alpha=0.85)
        ax.axvline(10, color=INK, ls="--", lw=1.1, label="S = 10 (thesis line)")
        p = boot.get("P_S_gt_10")
        ax.axvline(boot["S_point"], color=INK2, lw=1.4,
                   label=f"point S = {boot['S_point']:.2f}")
        if p is not None:
            ax.set_title(f"{title}   P(S > 10) = {p:.2f}", color=INK, fontsize=11, loc="left")
        else:
            ax.set_title(title, color=INK, fontsize=11, loc="left")
        leg = ax.legend(fontsize=8, frameon=False)
        for t in leg.get_texts():
            t.set_color(INK2)
    else:
        ax.text(0.5, 0.5, "no valid bootstrap draws", ha="center", color=INK2)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("S (max/min shrunken, mix-standardized tokens per accepted)",
                  color=INK2, fontsize=9)
    ax.set_ylabel("draws", color=INK2, fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def write_report(
    out_dir: Path,
    segments: dict,
    fold_diag: dict,
    naive: pd.DataFrame,
    shrunk: pd.DataFrame,
    boot: dict,
    brackets: dict,
    week_sensitivity: dict,
    apdt: pd.DataFrame,
    dag_pair: dict,
    min_n: int,
    anchor: pd.DataFrame | None = None,
    overlap_pairs: pd.DataFrame | None = None,
    ledger: dict | None = None,
    excluded_labels: dict | None = None,
) -> Path:
    """Assemble report.md. Returns the report path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s = boot
    decision = "insufficient data"
    if not np.isnan(s.get("P_S_gt_10", np.nan)):
        if s["P_S_gt_10"] >= 0.9:
            decision = "claim 'more than tenfold' CONFIRMED under the frozen decision rule"
        elif s.get("S_point", 0) > 10:
            decision = "downgrade to 'consistent with an order of magnitude'"
        else:
            decision = "report the honest smaller number; tenfold not supported on this slice"

    pair = s.get("pair", ("?", "?"))
    pair_line = "not computable (pair not jointly eligible)"
    if "pair_ratio_point" in s:
        pair_line = (
            f"{pair[1]} / {pair[0]} = {s['pair_ratio_point']:.2f} "
            f"[{s.get('pair_ratio_lo90', float('nan')):.2f}, "
            f"{s.get('pair_ratio_hi90', float('nan')):.2f}] (90% interval)"
        )

    # model-only comparisons: eligible pairs sharing zero overlap cells
    model_only_lines = []
    if overlap_pairs is not None and not overlap_pairs.empty:
        zero = overlap_pairs[overlap_pairs["shared_cells"] == 0]
        for _, row in zero.iterrows():
            model_only_lines.append(
                f"- {row['arm_a']} and {row['arm_b']} share NO overlap cell; their "
                "contrast is bridged entirely through the additive model "
                "(design section 2 requires saying so)."
            )

    lines = [
        "# loopmath analyze report",
        "",
        "## Abstract",
        "",
        (
            f"This run estimates tokens-per-accepted across model x effort arms on "
            f"{len(segments['main'])} real main sessions "
            f"({len(segments['fleet_stubs'])} fleet stubs and "
            f"{len(segments['subagents'])} subagent transcripts segmented out; "
            f"{fold_diag['folded_tokens']:,} subagent output tokens folded into "
            f"parents from {fold_diag['n_folded']} transcripts). "
            f"The shrunken, cell-adjusted, mix-standardized spread is "
            f"S = {s.get('S_point', float('nan')):.2f} "
            f"with 90% interval [{s.get('S_lo90', float('nan')):.2f}, "
            f"{s.get('S_hi90', float('nan')):.2f}] and P(S > 10) = "
            f"{s.get('P_S_gt_10', float('nan')):.2f} "
            f"({s.get('n_boot_valid', 0)} valid draws, seed {s.get('seed')}). "
            f"Decision rule outcome: {decision}. "
            "Proxy-grade acceptance on this slice. Proxy coverage on real "
            "conversations is about 91% (computed from this corpus); the about "
            "91% reliability figure is the corpus builder's reported validation, "
            "carried from the E0 briefing, with no provenance inside the corpus "
            "itself."
        ),
        "",
        "## Segmentation",
        "",
        f"- real main sessions: {len(segments['main'])}",
        (
            f"- harvest fleet stubs, Aug 23-24 (primary_model null AND "
            f"error_events >= 1): {len(segments['fleet_stubs'])}"
        ),
        (
            f"- subagent transcripts: {fold_diag['n_subagent_transcripts']}, "
            f"{fold_diag['n_folded']} folded into parents "
            f"({fold_diag['folded_tokens']:,} output tokens) via "
            f"parent_session_id or parent_thread_id; "
            f"{fold_diag['unmatched_files']} without a real-main parent "
            f"({fold_diag['unmatched_tokens']:,} tokens, reported, not attributed)"
        ),
        "",
        f"## Naive per-arm estimates (unadjusted, full population, arms with n >= {min_n})",
        "",
        markdown_table(naive),
        "",
        "The per-accepted ratio uses known-proxy sessions in both numerator and",
        "denominator (design section 3); total_output_tokens covers all arm",
        "sessions and is a ledger column, not the ratio numerator.",
        "",
        "## Within-cell matched anchor (design step 1, model-free)",
        "",
        markdown_table(anchor) if anchor is not None else "(not computed)",
        "",
        "Exhaustive over eligible arms and cells with at least 5 sessions per",
        "side. Ratios are unadjusted within-cell median-cost ratios; acc_diff is",
        "the known-proxy acceptance difference (hi minus lo arm).",
        "",
        "## Normalized estimates (joint arm+cell fit, shrunken, mix-standardized, proxy grade)",
        "",
        markdown_table(shrunk),
        "",
        "Cells are project family x user-turn bucket, restricted to overlap cells.",
        "The fit is the design step-2 joint OLS on arm + cell effects; arm",
        "effects are then James-Stein shrunk (step 3) and mix-standardized.",
        "Trivial low-effort arms are excluded from the headline S.",
        f"Prespecified pairwise guard: {pair_line}.",
        (
            f"Unknown-proxy brackets: S = {brackets.get('S_unknown_accepted', float('nan')):.2f} "
            f"with unknowns counted accepted, {brackets.get('S_unknown_rejected', float('nan')):.2f} "
            "with unknowns counted not accepted."
        ),
        (
            f"Week-in-cell sensitivity (original frozen cell definition): "
            f"S = {week_sensitivity.get('S_point', float('nan')):.2f} on "
            f"{week_sensitivity.get('n_arms', 0)} arms "
            f"({week_sensitivity.get('note', '')})"
        ),
        "",
        "### Overlap diagnostic (shared overlap cells per eligible arm pair)",
        "",
        markdown_table(overlap_pairs) if overlap_pairs is not None else "(not computed)",
        "",
        *(model_only_lines or ["- every eligible arm pair shares at least one overlap cell."]),
        "",
        "### Bootstrap arm-set diagnostic",
        "",
        (
            markdown_table(s["arm_set_diag"])
            if isinstance(s.get("arm_set_diag"), pd.DataFrame) and not s["arm_set_diag"].empty
            else "(no draws)"
        ),
        "",
        "Eligibility reruns inside every draw, so the arm set behind S varies in",
        "both directions: arms can lose eligibility in a draw, and arms outside",
        "the point-estimate set can enter. in_S_draws counts draws in which the",
        "arm entered the S computation; as_max / as_min count draws in which it",
        "was the S endpoint.",
        "",
        "## Attempts per done task (gate grade, dag slice, join-free)",
        "",
        markdown_table(apdt),
        "",
        (
            f"Prespecified dag pair: {dag_pair.get('label', '?')} = "
            f"{dag_pair.get('ratio', float('nan')):.2f} "
            f"[{dag_pair.get('lo90', float('nan')):.2f}, "
            f"{dag_pair.get('hi90', float('nan')):.2f}] (90% interval)."
        ),
        "",
        "followup_share counts followup attempts (scope extensions on accepted",
        "work); failure_rework_share counts sent_back and gate_failed attempts.",
        "The two invert the arm ranking, so neither is quoted alone as 'rework'.",
        "Tasks are credited per arm via the arm's own terminal attempt; on",
        "multi-arm tasks this is an attribution choice, documented in the",
        "design's Changes log.",
        (
            f"Unlabeled attempts sharing a task with labeled attempts: "
            f"{dag_pair.get('n_unlabeled_on_labeled_tasks', 0)} "
            "(when 0, excluding unlabeled attempts from arm tables drops no"
            " labeled-arm task cost)."
        ),
        "",
        "## Fleet ledger (harvest stubs, fleet-run unit)",
        "",
    ]
    if ledger:
        dates = ", ".join(f"{d}: {n}" for d, n in ledger["dates"].items())
        lines += [
            f"- stubs: {ledger['n_stubs']} ({dates})",
            (
                f"- recorded wall-clock: {ledger['wallclock_min']:.1f} minutes total "
                f"(median {ledger['median_s']:.1f} s, mean {ledger['mean_s']:.1f} s, "
                f"max {ledger['max_s']:.1f} s; {ledger['n_over_60s']} stubs over 60 s)"
            ),
            (
                f"- tokens: {ledger['output_tokens']:,} output, "
                f"{ledger['input_tokens']:,} input"
            ),
            "- the distribution is heavy-tailed: the median stub is a ~2.3 s",
            "  one-shot but the wall-clock total is dominated by the tail.",
        ]
    else:
        lines += ["(not computed)"]
    if excluded_labels:
        comp = ", ".join(f"{k}: {v}" for k, v in sorted(excluded_labels.items()))
        lines += [
            "",
            "## Excluded attempt labels (no canonical arm)",
            "",
            f"- composition: {comp}",
        ]
    lines += [
        "",
        "## Figures (each as PNG and SVG)",
        "",
        "- `forest`: normalized cost per arm with 90% intervals (F2)",
        "- `pareto`: cost vs acceptance scatter (F7)",
        "- `rework`: dag-slice attempts per done task with intervals",
        "- `bootstrap_s`: bootstrap distribution of S (F4)",
        "",
        "## Method deviations and open items",
        "",
        "- Cell = family x turn bucket; week dropped from the cell key because",
        "  week is perfectly aliased with arm inside the video family. Logged in",
        "  docs/e0-design.md Changes; week-in-cell runs as sensitivity above.",
        "- Cost adjustment is the design step-2 joint OLS (arm + cell effects);",
        "  the x beta covariates (log user turns, tool, week) are not included.",
        "  James-Stein moments shrinkage (step 3 fallback), not MixedLM.",
        "- Acceptance is a raw rate, not the logit model; acceptance is near",
        "  ceiling on this slice so the model would move little.",
        "- DAG-slice intervals: two-stage Bayesian bootstrap over run weights",
        "  then task weights (the design's frozen run-cluster bootstrap; the",
        "  earlier iid task bootstrap ignored run clustering).",
        "- Gate-grade tokens-per-done-task via the session join not run",
        "  (attribution tiers too loose; attempts-per-done-task is primary).",
        "- Price-weighted billed tokens not run (secondary in the design).",
        "- Step-1 inverse-variance-weighted combination and figures F1, F3, F5,",
        "  F6 not produced; the waterfall and anchor exist as tables, the fleet",
        "  ledger as a section above.",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
