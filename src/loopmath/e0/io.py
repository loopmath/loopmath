"""Loaders for the E0 corpus (metadata-only, read-only).

Files: sessions.jsonl, dag-runs.jsonl, dag-attempts.jsonl, session-dag-join.jsonl.
Field definitions live in the corpus's parser-spec.md; segmentation rules come
from docs/e0-design.md sections 1 and 4.

The corpus contains no transcript text and none may appear in any output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .. import research_paths

# The corpus folder is a config value, not a path in the package: None means
# "resolve per call" through research_paths.e0_corpus() (the --corpus flag, then
# LOOPMATH_E0_CORPUS, then research.e0_corpus in config.toml). Tests may set it.
DEFAULT_CORPUS: Path | None = None

# reasoning_effort_signals entries look like "assistant.effort=xhigh(695)" or
# "turn_context.effort=max(12)"; non-effort entries like "thinking_blocks(208)"
# are ignored. Same regex as the first-cut script.
_EFFORT_RE = re.compile(r".*effort=(\w+)\((\d+)\)")


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _corpus_dir(corpus_dir: str | Path | None) -> Path:
    return research_paths.e0_corpus(corpus_dir if corpus_dir else DEFAULT_CORPUS)


def load_sessions(path: str | Path | None = None) -> list[dict]:
    return _load_jsonl(Path(path) if path else _corpus_dir(None) / "sessions.jsonl")


def load_dag_runs(path: str | Path | None = None) -> list[dict]:
    return _load_jsonl(Path(path) if path else _corpus_dir(None) / "dag-runs.jsonl")


def load_dag_attempts(path: str | Path | None = None) -> list[dict]:
    return _load_jsonl(Path(path) if path else _corpus_dir(None) / "dag-attempts.jsonl")


def load_join(path: str | Path | None = None) -> list[dict]:
    return _load_jsonl(Path(path) if path else _corpus_dir(None) / "session-dag-join.jsonl")


def load_corpus(corpus_dir: str | Path | None = None) -> dict[str, list[dict]]:
    d = _corpus_dir(corpus_dir)
    return {
        "sessions": load_sessions(d / "sessions.jsonl"),
        "dag_runs": load_dag_runs(d / "dag-runs.jsonl"),
        "dag_attempts": load_dag_attempts(d / "dag-attempts.jsonl"),
        "join": load_join(d / "session-dag-join.jsonl"),
    }


def parse_effort_signals(signals: list[str] | None) -> dict[str, int]:
    """Parse reasoning_effort_signals into {effort_level: count}."""
    out: dict[str, int] = {}
    for sig in signals or []:
        m = _EFFORT_RE.match(sig)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + int(m.group(2))
    return out


def dominant_effort(session: dict) -> str | None:
    """Highest-count effort signal wins (design doc 1a).

    Sessions with several effort levels get blurred to the dominant one; the
    caller should report the multi-effort count as a diagnostic.
    """
    counts = parse_effort_signals(session.get("reasoning_effort_signals"))
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def is_subagent(session: dict) -> bool:
    """Subagent transcripts are never their own unit (design doc 1a).

    Claude Code marks subagents via parallelism_signals / extras.is_subagent /
    extras.parent_session_id; codex marks them via extras.parent_thread_id.
    """
    if "subagent_transcript" in (session.get("parallelism_signals") or []):
        return True
    extras = session.get("extras") or {}
    if extras.get("is_subagent"):
        return True
    if extras.get("parent_session_id"):
        return True
    if extras.get("parent_thread_id"):
        return True
    return False


def is_fleet_stub(session: dict) -> bool:
    """Aug-23 fleet-churn rule, exact (design doc section 4):
    primary_model == null AND outcome.error_events >= 1, on main sessions.
    """
    outcome = session.get("outcome") or {}
    return session.get("primary_model") is None and (outcome.get("error_events") or 0) >= 1


def segment_sessions(sessions: list[dict]) -> dict[str, list[dict]]:
    """Split into main (real), fleet stubs, and subagent transcripts.

    Fleet stubs are counted as fleet-arm cost, never as model-arm evidence.
    Subagent tokens fold into the parent session via subagent_token_fold
    (design doc 1a).
    """
    main, stubs, subagents = [], [], []
    for s in sessions:
        if is_subagent(s):
            subagents.append(s)
        elif is_fleet_stub(s):
            stubs.append(s)
        else:
            main.append(s)
    return {"main": main, "fleet_stubs": stubs, "subagents": subagents}


def subagent_token_fold(segments: dict[str, list[dict]]) -> tuple[dict[str, int], dict]:
    """Fold subagent output tokens into their parent main session (design 1a).

    Returns ({parent_session_id: extra_output_tokens}, diagnostics). The parent
    link is extras.parent_session_id (Claude Code) or, failing that,
    extras.parent_thread_id (codex; thread ids are session ids there). Subagent
    transcripts whose parent is not a real main session (parent missing, parent
    itself a subagent, or a fleet stub) cannot be folded; their token total is
    reported as a diagnostic, never silently dropped.
    """
    main_ids = {s.get("session_id") for s in segments["main"]}
    fold: dict[str, int] = {}
    unmatched_tokens = 0
    unmatched_files = 0
    for s in segments["subagents"]:
        tokens = (s.get("tokens") or {}).get("output") or 0
        extras = s.get("extras") or {}
        parent = extras.get("parent_session_id") or extras.get("parent_thread_id")
        if parent in main_ids:
            fold[parent] = fold.get(parent, 0) + tokens
        else:
            unmatched_tokens += tokens
            unmatched_files += 1
    diag = {
        "n_subagent_transcripts": len(segments["subagents"]),
        "n_folded": len(segments["subagents"]) - unmatched_files,
        "folded_tokens": sum(fold.values()),
        "unmatched_files": unmatched_files,
        "unmatched_tokens": unmatched_tokens,
    }
    return fold, diag


def fleet_ledger(stubs: list[dict]) -> dict:
    """Fleet-arm churn accounting for the harvest stubs (design 1d, section 4).

    Computed from the corpus, never copied forward: counts, recorded wall-clock
    (sum of duration_s, not count times median), date composition, and token
    totals. The stubs are heavy-tailed: most are ~2.3 s one-shots but a minority
    run minutes, so the sum is the honest wall-clock number.
    """
    durs = sorted((s.get("duration_s") or 0.0) for s in stubs)
    dates: dict[str, int] = {}
    for s in stubs:
        d = (s.get("started_at") or "")[:10] or "unknown"
        dates[d] = dates.get(d, 0) + 1
    n = len(stubs)
    total_s = float(sum(durs))
    return {
        "n_stubs": n,
        "wallclock_s": total_s,
        "wallclock_min": total_s / 60.0,
        "median_s": float(durs[n // 2]) if n else float("nan"),
        "mean_s": total_s / n if n else float("nan"),
        "max_s": float(durs[-1]) if n else float("nan"),
        "n_over_60s": sum(1 for d in durs if d > 60),
        "dates": dict(sorted(dates.items())),
        "output_tokens": sum(((s.get("tokens") or {}).get("output") or 0) for s in stubs),
        "input_tokens": sum(((s.get("tokens") or {}).get("input") or 0) for s in stubs),
    }
