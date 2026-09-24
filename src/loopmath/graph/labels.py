"""Role and phase heuristics layered on the structural graph, labeled as such.

Moved unchanged from extract.py; no task owns this module today. The lead is the
top-level Claude session with the most spawn/launch out-edges in its workspace
(ties: most output tokens, then longest wall clock); a lead with no out-edges is
`solo`. Everything that starts more than `POST_BUILD_GAP_S` after the lead's last
activity is phase `post` (the swarms each had one post-build review session per
harness). Subagent roles come from the declared `subagent_type` when it names a
role (tier `reported`), else from the spawn description (tier `heuristic`), else
`dev`. Codex roles come from the launching command text or the session's first
user prompt; without either the role stays None.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .scan import codex_first_prompt, epoch
from .schema import GraphEdge, GraphNode

_PLANNER_TYPE_RE = re.compile(r"plan|architect", re.I)
_REVIEWER_TYPE_RE = re.compile(r"review|referee|audit|critic", re.I)
_PLANNER_DESC_RE = re.compile(r"^\W*(plan|design|architect)\b", re.I)
# Anchored: "Fix grade.py review blockers" is a dev task, "Review grade.py" is not.
_REVIEWER_DESC_RE = re.compile(r"^\W*(review|referee|audit|critique)\b", re.I)
_REVIEWER_TEXT_RE = re.compile(r"\b(review|referee|audit|critique|verdict)\b", re.I)

POST_BUILD_GAP_S = 60.0


def tokens_out(n: GraphNode) -> int:
    for key in ("out", "output", "output_tokens"):
        v = (n.tokens or {}).get(key)
        if isinstance(v, (int, float)):
            return int(v)
    return 0


def label_roles_and_phases(nodes: dict[str, GraphNode], edges: list[GraphEdge], launch_cmd: dict[str, str]) -> None:
    out_deg: dict[str, int] = defaultdict(int)
    for e in edges:
        if e.kind in ("spawn", "launch"):
            out_deg[e.src] += 1
    by_ws: dict[str | None, list[GraphNode]] = defaultdict(list)
    for n in nodes.values():
        by_ws[n.workspace].append(n)

    for ws, group in by_ws.items():
        group = [n for n in group if n.source != "external"]
        tops = [n for n in group if n.source == "top" and n.phase != "external"]
        lead = None
        if tops:
            lead = max(tops, key=lambda n: (out_deg[n.id], tokens_out(n), n.wall_s or 0.0))
            lead.role = "lead" if out_deg[lead.id] else "solo"
            lead.role_tier = "heuristic"
            lead.role_evidence = f"top-level session with the most spawn/launch out-edges in workspace ({out_deg[lead.id]})"
        lead_end = None
        if lead is not None:
            le = epoch(lead.ts)
            if le is not None:
                lead_end = le + (lead.wall_s or 0.0)
        for n in group:
            e = epoch(n.ts)
            if n.phase == "external":
                pass
            elif lead_end is None or e is None:
                n.phase = None
                n.phase_tier = None
            else:
                n.phase = "post" if e > lead_end + POST_BUILD_GAP_S else "build"
                n.phase_tier = "heuristic"
            if n is lead:
                continue
            if n.source == "top":
                n.role = None
                if n.phase == "external":
                    n.role_evidence = "top-level session launched from another workspace"
                else:
                    n.role_evidence = "top-level session that is not the lead" + ("; starts after the lead ended" if n.phase == "post" else "")
            elif n.source == "subagent":
                _label_subagent(n)
            else:
                _label_codex(n, launch_cmd.get(n.id))


def _label_subagent(n: GraphNode) -> None:
    sp = n.spawn or {}
    declared = " ".join(str(sp.get(k) or "") for k in ("subagent_type", "agent_type"))
    desc = str(sp.get("description") or sp.get("meta_description") or "")
    if declared.strip() and _PLANNER_TYPE_RE.search(declared):
        n.role, n.role_tier, n.role_evidence = "planner", "reported", f"declared type {declared.strip()!r}"
    elif declared.strip() and _REVIEWER_TYPE_RE.search(declared):
        n.role, n.role_tier, n.role_evidence = "reviewer", "reported", f"declared type {declared.strip()!r}"
    elif _PLANNER_DESC_RE.search(desc):
        n.role, n.role_tier, n.role_evidence = "planner", "heuristic", f"spawn description {desc[:80]!r}"
    elif _REVIEWER_DESC_RE.search(desc):
        n.role, n.role_tier, n.role_evidence = "reviewer", "heuristic", f"spawn description {desc[:80]!r}"
    else:
        n.role, n.role_tier, n.role_evidence = "dev", "heuristic", "subagent with no planner/reviewer signal"


def _label_codex(n: GraphNode, command: str | None) -> None:
    if command and _REVIEWER_TEXT_RE.search(command):
        n.role, n.role_tier, n.role_evidence = "reviewer", "heuristic", "launch command mentions review"
        return
    prompt = codex_first_prompt(Path(n.session_path))
    launched = "launched by a session" if command else "not launched by any session in scope"
    if prompt and _REVIEWER_TEXT_RE.search(prompt):
        n.role, n.role_tier, n.role_evidence = "reviewer", "heuristic", f"{launched}; first user prompt mentions review"
    elif command:
        n.role, n.role_tier, n.role_evidence = "cli", "heuristic", f"{launched}; neither command nor prompt has role words"
    else:
        n.role, n.role_tier, n.role_evidence = None, None, f"{launched}; " + ("prompt has no role words" if prompt else "no prompt found")
