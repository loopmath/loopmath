"""Arm construction: model x effort at minimum (design doc section 1).

Session slice: primary_model x dominant effort signal.
DAG slice: canonicalized attempt model labels, which already carry effort as a
suffix. Topology and parallelism covariates come from the run table where
present (dag_depth, max_parallel_width).
"""

from __future__ import annotations

import re

import pandas as pd

from .io import dominant_effort, parse_effort_signals

# Canonicalization for attempt model labels (design doc 1b). Observed
# vocabulary includes claude-fable-5·xhigh, fable5·xhigh, fable·xhigh, the
# typo opus5.xhigh, plus null and "-" labels. This table ships with the
# analysis and is reported verbatim.
_MODEL_ALIASES = {
    "claude-fable-5": "fable5",
    "fable": "fable5",
    "fable5": "fable5",
    "claude-opus-5": "opus5",
    "opus": "opus5",
    "opus5": "opus5",
    "claude-sonnet-5": "sonnet5",
    "sonnet5": "sonnet5",
    "sol5.6": "sol5.6",
    "gpt-5.6-sol": "sol5.6",
}

_EFFORTS = {"low", "medium", "high", "xhigh", "max", "1m"}

# Session-slice primary_model -> short arm model name, to line up with the
# dag-slice vocabulary.
_SESSION_MODEL_ALIASES = {
    "claude-opus-5": "opus5",
    "claude-fable-5": "fable5",
    "claude-sonnet-5": "sonnet5",
    "gpt-5.6-sol": "sol5.6",
    "gpt-5.4": "gpt5.4",
    "gpt-5.6-terra": "terra5.6",
    "gpt-5.6-luna": "luna5.6",
    "claude-opus-4-8": "opus4.8",
    "claude-sonnet-4-6": "sonnet4.6",
}


def canonicalize_attempt_model(label: str | None) -> str | None:
    """Normalize an attempt model label to 'model·effort', or None.

    Returns None for null, empty, and '-' labels, and also for labels without a
    recognizable effort suffix (bare 'opus5', 'local·script'); the corpus
    decomposition is 34 null + 4 '-' + 3 'opus5' + 1 'local·script' = 42. The
    caller excludes these from arm tables; see attempts_per_done_task for how
    their cost is handled (design doc risk 7).
    """
    if not label or label.strip() in {"-", "?"}:
        return None
    label = label.strip()
    if "·" in label:  # the middle-dot separator
        model, _, effort = label.rpartition("·")
    elif "." in label and label.rsplit(".", 1)[-1] in _EFFORTS:
        # typo form like opus5.xhigh; guard so sol5.6 does not split
        model, _, effort = label.rpartition(".")
    else:
        return None
    model = _MODEL_ALIASES.get(model, model)
    effort = effort.lower()
    if effort not in _EFFORTS:
        return None
    return f"{model}·{effort}"


def session_arm(session: dict) -> str | None:
    """'model·effort' for a main session, or None when no model label."""
    model = session.get("primary_model")
    if not model:
        return None
    model = _SESSION_MODEL_ALIASES.get(model, model)
    effort = dominant_effort(session) or "?"
    return f"{model}·{effort}"


# Project-family collapse: worktree lanes fold into families (design doc
# section 2). Verified against the corpus inventory on 2026-08-30:
# - trailing digits are lane numbers (bots-c1..c6, project-fable2..3)
# - -remediation / -vX.Y-recovery are herdr-dagr recovery lanes
# - pitauri-t1..t7 ("pitauri-teams" sibling set per census section 7) and
#   pitauri-ft1..ft5 ("pit-followups") carry task-name suffixes, e.g.
#   pitauri-t1-settings, so they collapse by prefix, not by trailing digit
# - -fable / -sol suffixes are paired model lanes of the same remediation
#   project (interactions / interactions-fable3 / interactions-sol3 and five
#   more base names under herdr-dagr-remediation); they collapse to the base
#   so the deliberate fable-vs-sol A/B lanes land in shared cells
_LANE_RE = re.compile(r"(-remediation.*|-v[\d.]+-recovery.*|\d+$)")
_MODEL_LANE_RE = re.compile(r"-(fable|sol)$")
_PITAURI_TEAM_RE = re.compile(r"^pitauri-t\d")
_PITAURI_FT_RE = re.compile(r"^pitauri-ft\d")


def project_family(project: str | None) -> str:
    if not project:
        return "unknown"
    base = project.rstrip("/").split("/")[-1]
    if _PITAURI_FT_RE.match(base):
        return "pitauri-ft"
    if _PITAURI_TEAM_RE.match(base):
        return "pitauri-teams"
    base = _LANE_RE.sub("", base)
    base = _MODEL_LANE_RE.sub("", base)
    if base.startswith("video-"):  # video-analyzer <-> video-orchestrator
        base = "video"
    return base or "unknown"


def _turn_bucket(n: int | None) -> str:
    n = n or 0
    if n <= 1:
        return "1"
    if n <= 4:
        return "2-4"
    return "5+"


def build_session_table(
    main_sessions: list[dict],
    extra_output_tokens: dict[str, int] | None = None,
) -> pd.DataFrame:
    """One row per real main session with arm, cost, acceptance, and cell.

    Primary cell = project family x user-turn bucket. The design froze
    family x bucket x week, but week is perfectly aliased with arm inside the
    video family (opus5 xhigh ran only in W34, opus5 medium only in W35), so a
    week term makes the design's own named key contrast unobservable. Week is
    kept in `cell_week` for the sensitivity variant and the deviation is
    logged in docs/e0-design.md Changes.

    extra_output_tokens folds subagent output tokens into their parent
    session's cost (io.subagent_token_fold).
    """
    extra = extra_output_tokens or {}
    rows = []
    for s in main_sessions:
        tokens = s.get("tokens") or {}
        outcome = s.get("outcome") or {}
        started = s.get("started_at") or ""
        week = pd.Timestamp(started).strftime("%G-W%V") if started else "unknown"
        efforts = parse_effort_signals(s.get("reasoning_effort_signals"))
        proxy = outcome.get("acceptance_proxy")
        sid = s.get("session_id")
        rows.append(
            {
                "session_id": sid,
                "arm": session_arm(s),
                "tool": s.get("tool"),
                "project": s.get("project"),
                "family": project_family(s.get("project")),
                "week": week,
                "turn_bucket": _turn_bucket(s.get("n_turns_user")),
                "n_turns_user": s.get("n_turns_user") or 0,
                "output_tokens": (tokens.get("output") or 0) + extra.get(sid, 0),
                "input_tokens": tokens.get("input") or 0,
                "accepted": proxy == "accepted",
                "proxy": proxy,
                "proxy_known": proxy in ("accepted", "mixed", "rejected"),
                "multi_effort": len(efforts) > 1,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["cell"] = df["family"] + "|" + df["turn_bucket"]
        df["cell_week"] = df["family"] + "|" + df["turn_bucket"] + "|" + df["week"]
    return df


def build_attempt_table(attempts: list[dict]) -> pd.DataFrame:
    """One row per DAG attempt with canonical arm and cell = run_id x task_kind."""
    rows = []
    for a in attempts:
        accepted = a.get("outcome_result") == "done" and a.get("outcome_evidence") in (
            "verified",
            "reported",
        )
        rows.append(
            {
                "run_id": a.get("run_id"),
                "task_id": a.get("task_id"),
                "task_kind": a.get("task_kind"),
                "attempt_id": a.get("attempt_id"),
                "n": a.get("n"),
                "arm": canonicalize_attempt_model(a.get("model")),
                "raw_model": a.get("model"),
                "cause": a.get("cause"),
                "outcome_result": a.get("outcome_result"),
                "outcome_evidence": a.get("outcome_evidence"),
                "accepted": accepted,
                "open_work": a.get("outcome_result") is None or a.get("ended_at") is None,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["cell"] = df["run_id"].astype(str) + "|" + df["task_kind"].astype(str)
    return df


def attach_run_topology(attempt_df: pd.DataFrame, runs: list[dict]) -> pd.DataFrame:
    """Join run-level topology/parallelism covariates onto attempts."""
    topo = pd.DataFrame(
        [
            {
                "run_id": r.get("run_id"),
                "dag_depth": r.get("dag_depth"),
                "max_parallel_width": r.get("max_parallel_width"),
                "contract_version": r.get("contract_version"),
            }
            for r in runs
        ]
    )
    return attempt_df.merge(topo, on="run_id", how="left")
