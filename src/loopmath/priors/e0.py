"""The E0 corpus (lane 11): our own Claude Code and Codex sessions as logged habit runs.

Input: the read-only corpus copy (D18), `sessions.jsonl` and
`session-dag-join.jsonl`, through an explicit path; `e0/io.py`'s default corpus
is never used. The corpus is metadata only (parser-spec.md); no text reaches
the bundle.

- One run per real main session: it has a model and tokens. Left out and
  counted: fleet stubs (no model and an API error, the Aug-23 harvest churn,
  which is launch failure rather than model behaviour), other sessions with no
  model or no tokens, and sessions in temporary folders (the bb integration
  test harness).
- A subagent transcript joins its parent's run as one more attempt on the same
  piece, with its own model, effort and tokens (`dev.loopmath.subagent`);
  subagents whose parent is not a bundled run are counted, not bundled.
- The configuration is the catalog `solo` shape: one implementer, the session's
  harness, primary model and dominant effort; `configuration.source: habit`,
  `provenance {kind: logged, chooser: habit}`.
- No verdicts (D7): no signals and no acceptance rule; attempts are
  `settled_unverified`. These runs feed the cost and tokens heads only.
- The task has no type (the corpus has no text to label) and no features; the
  repo is the workspace folder name, which the reduction replaces with a salted
  hash.
- Tokens: Codex `input_tokens` include the cached tokens, so the uncached input
  is `input - cache_read` (as `ingest.codex` does); Claude Code streams are
  already separate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from ..ingest.base import canonical_effort
from . import ocpdoc
from .registry import E0

CONVERTER_VERSION = "e0/1"
_EFFORT_RE = re.compile(r".*effort=(\w+)\((\d+)\)")
_TMP = re.compile(r"^/(private/)?var/|^/tmp/")
_WORKSPACE = re.compile(r"/Workspace/([^/]+)")


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _efforts(session: dict) -> Counter:
    out: Counter = Counter()
    for sig in session.get("reasoning_effort_signals") or []:
        m = _EFFORT_RE.match(str(sig))
        if m:
            out[m.group(1)] += int(m.group(2))
    return out


def _is_subagent(s: dict) -> bool:
    extras = s.get("extras") or {}
    return ("subagent_transcript" in (s.get("parallelism_signals") or []) or bool(extras.get("is_subagent"))
            or bool(extras.get("parent_session_id")) or bool(extras.get("parent_thread_id")))


def _parent(s: dict) -> str | None:
    extras = s.get("extras") or {}
    return extras.get("parent_session_id") or extras.get("parent_thread_id")


def _is_fleet_stub(s: dict) -> bool:
    return s.get("primary_model") is None and int((s.get("outcome") or {}).get("error_events") or 0) >= 1


def repo_of(project: str | None) -> str | None:
    """The workspace folder a session ran in, None for temporary folders."""
    if not project or _TMP.search(project):
        return None
    m = _WORKSPACE.search(project)
    return m.group(1) if m else "other"


def project_kind(project_class: str | None) -> str | None:
    """The corpus's project class without the project or tool name in it (D70): orchestrated, agent or other."""
    if not project_class:
        return None
    for kind in ("orchestrated", "agent"):
        if project_class.endswith("-" + kind):
            return kind
    return "other"


def _tokens(s: dict) -> dict:
    t = s.get("tokens") or {}
    streams = {k: int(t.get(k) or 0) for k in ("input", "output", "cache_read", "cache_write")}
    if s.get("tool") == "codex":
        streams["input"] = max(0, streams["input"] - streams["cache_read"])
    return streams


def _harness(tool: str | None) -> str:
    return "codex" if tool == "codex" else "claude-code"


def _run_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def _attempt(s: dict, n: int, *, subagent: bool, prices) -> dict:
    model = s.get("primary_model")
    efforts = _efforts(s)
    eff = canonical_effort(efforts.most_common(1)[0][0]) if efforts else None
    rec: dict = {"id": f"implement.a{n}", "node": "implement", "n": n, "vertex": "implement", "round": 1,
                 "actor": "subagent" if subagent else "session", "harness": _harness(s.get("tool")),
                 "model": ocpdoc.model_ref(model), "effort": ocpdoc.effort(eff),
                 "role": {"value": "implementer", "tier": "heuristic", "evidence": None},
                 "status": "settled_unverified"}
    for k in ("started_at", "ended_at"):
        if s.get(k):
            rec[k] = s[k]
    rec["outcome"] = {"result": "settled_unverified", "evidence": "reported"}
    rec["cost"] = ocpdoc.cost_block(model, _tokens(s), prices=prices)
    ext: dict = {}
    if subagent:
        ext["dev.loopmath.subagent"] = True
    if len(efforts) > 1:
        ext["dev.loopmath.multi_effort"] = dict(sorted(efforts.items()))
    if ext:
        rec["ext"] = ext
    return rec


def convert_session(s: dict, children: list[dict], *, prices=None, producer_version: str = "",
                    joined: bool = False) -> dict:
    """One OCP v0.3 run for a main session and its subagent transcripts."""
    model = s.get("primary_model")
    efforts = _efforts(s)
    eff = canonical_effort(efforts.most_common(1)[0][0]) if efforts else None
    workflow = ocpdoc.workflow_solo()
    settings = {"implement": ocpdoc.setting(_harness(s.get("tool")), model, eff)}
    attempts = [_attempt(s, 1, subagent=False, prices=prices)]
    for i, child in enumerate(sorted(children, key=lambda c: str(c.get("started_at") or "")), start=2):
        attempts.append(_attempt(child, i, subagent=True, prices=prices))
    key = _run_key(str(s["session_id"]))
    outcome = s.get("outcome") or {}
    run = {
        "id": f"e0/{key}",
        "started_at": s.get("started_at"),
        "ended_at": s.get("ended_at"),
        "task": {"id": f"e0/{key}", "repo": repo_of(s.get("project")), "org": ocpdoc.ORG,
                 "source": {"kind": E0, "ref": "e0-2026-08-28"}},
        "configuration": ocpdoc.configuration(workflow, settings, source="habit"),
        "provenance": {"kind": "logged", "chooser": "habit"},
        "ext": {"dev.loopmath.prior": {
            "source": E0, "converter": CONVERTER_VERSION, "tool": s.get("tool"),
            "project_kind": project_kind(s.get("project_class")), "dagr_joined": joined, "subagents": len(children),
            "window_flag": s.get("window_flag"), "ended_by": outcome.get("ended_by"),
            "error_events": int(outcome.get("error_events") or 0),
            "duration_s": s.get("duration_s"), "tool_calls": s.get("n_tool_calls"),
        }},
    }
    run = {k: v for k, v in run.items() if v is not None}
    nodes = [{"id": "implement", "kind": "impl", "vertex": "implement", "state": "settled_unverified"}]
    return ocpdoc.run_doc(run=run, nodes=nodes, attempts=attempts, producer_version=producer_version,
                          emitted_at=str(s.get("ended_at") or ""), source_contract="e0-corpus/parser-spec-v1")


def iter_e0(corpus_dir: Path, *, prices=None, producer_version: str = "", counts: dict | None = None
            ) -> Iterator[dict]:
    """Bundled runs from the corpus, in session order; `counts` gets what was left out and why."""
    corpus_dir = Path(corpus_dir)
    sessions = _load(corpus_dir / "sessions.jsonl")
    join_path = corpus_dir / "session-dag-join.jsonl"
    joined = {str(r.get("session_id")) for r in _load(join_path)} if join_path.is_file() else set()
    tally = counts if counts is not None else {}
    tally.update({"sessions": len(sessions), "runs": 0, "subagent_attempts": 0, "skipped": Counter()})

    main, subs = [], []
    for s in sessions:
        (subs if _is_subagent(s) else main).append(s)
    kept: dict[str, dict] = {}
    for s in main:
        if _is_fleet_stub(s):
            tally["skipped"]["fleet_stub"] += 1
        elif not s.get("primary_model"):
            tally["skipped"]["no_model"] += 1
        elif not s.get("tokens") or not any(_tokens(s).values()):
            tally["skipped"]["no_tokens"] += 1
        elif repo_of(s.get("project")) is None:
            tally["skipped"]["temporary_folder"] += 1
        else:
            kept[str(s["session_id"])] = s
    children: dict[str, list[dict]] = defaultdict(list)
    for s in subs:
        parent = _parent(s)
        if parent in kept and s.get("primary_model") and s.get("tokens"):
            children[parent].append(s)
        else:
            tally["skipped"]["subagent_without_bundled_parent"] += 1
    for sid, s in sorted(kept.items(), key=lambda kv: (str(kv[1].get("started_at") or ""), kv[0])):
        tally["runs"] += 1
        tally["subagent_attempts"] += len(children[sid])
        yield convert_session(s, children[sid], prices=prices, producer_version=producer_version,
                              joined=sid in joined)
    tally["skipped"] = dict(sorted(tally["skipped"].items()))
