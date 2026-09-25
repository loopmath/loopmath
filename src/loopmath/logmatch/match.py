"""Session matching: exact by session id (verified), else working folder, time window and model (heuristic, one candidate only).

Exact. An attempt that names its session is matched by that id only:
Claude Code keeps `<projects>/<cwd slug>/<id>.jsonl`, Codex keeps
`rollout-<local time>-<id>.jsonl` under `YYYY/MM/DD` (or flat in
`archived_sessions`), confirmed against `session_meta.payload.id`; every
file with the id counts, since a resumed Codex thread continues in a new
file. Tier `verified`. An id that is not found is no match, never a
fallback guess.

Heuristic. An attempt without a session id is matched by the same harness,
the same working folder (resolved), a session start inside the attempt's
window (`started_at` minus 120 s to `ended_at` plus 120 s) and, when the
attempt names one, the same canonical model. Sub-agent sessions are never
candidates, nor sessions already matched in the run (`exclude`). Exactly one
candidate gives tier `heuristic`; more than one is no match.

Children. Sub-agent sessions under the matched session count toward the same
attempt: Claude Code `<id>/subagents/agent-*.jsonl` (every depth sits in that
folder), and Codex rollouts whose `session_meta` names the session as
`parent_thread_id`, transitively, or as their root `session_id`. Each part
keeps its own model, so a child on a cheaper model is priced as such.

Only structure is read: ids, timestamps, working folders, models and token
counts, through the existing ingest parsers. No message text is kept.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..ingest import claude_code, codex
from ..ingest.base import RunRecord, Tokens, canonical_model, iter_jsonl, parse_ts, ts_epoch
from ..ingest.codex_support import parent_session_id

HARNESSES = ("claude-code", "codex")
WINDOW_S = 120
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_ROLLOUT_RE = re.compile(r"^rollout-(\d{4})-(\d{2})-(\d{2})T[\d-]+-(.+)\.jsonl$")


class SessionError(ValueError):
    """A session value that cannot name a session (see `resolve_session`)."""


@dataclass
class LogRoots:
    claude: list[Path] = field(default_factory=list)
    codex: list[Path] = field(default_factory=list)


@dataclass
class Match:
    harness: str
    session_path: Path
    session_id: str
    tier: str  # "verified" | "heuristic"
    tokens: Tokens
    model: str
    effort: str | None
    started_at: str
    ended_at: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    # Additive to spec 03: one entry per (file, model) with keys session,
    # role ("session" or "child"), harness, model, tokens, path, parent,
    # started_at, ended_at; every model seen; and why this session matched.
    parts: list[dict[str, Any]] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    reason: str = ""
    # Set by `clip.clip_match` for a `--session self` attempt:
    # `{from, to}` of the window the parts were cut to.
    clip: dict[str, Any] | None = None


def default_roots(env: Mapping[str, str] | None = None) -> LogRoots:
    """This machine's log folders, honoring `CLAUDE_CONFIG_DIR` and `CODEX_HOME`."""
    env = os.environ if env is None else env
    claude_home = Path(env["CLAUDE_CONFIG_DIR"]) if env.get("CLAUDE_CONFIG_DIR") else Path.home() / ".claude"
    codex_home = Path(env["CODEX_HOME"]) if env.get("CODEX_HOME") else Path.home() / ".codex"
    return LogRoots(
        claude=[claude_home.expanduser() / "projects"],
        codex=[codex_home.expanduser() / "sessions", codex_home.expanduser() / "archived_sessions"],
    )


def resolve_session(value: str, harness: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """The session id `value` names; `self` is the calling Claude Code session.

    `self` is exact under Claude Code, which exports `CLAUDE_CODE_SESSION_ID`
    to its tools. Codex exports no documented equivalent, so `self` there is
    an error asking for the id, never a guessed session.
    """
    env = os.environ if env is None else env
    text = str(value or "").strip()
    if text == "self":
        if harness == "codex":
            raise SessionError(
                "--session self works under Claude Code only; under Codex pass the thread id "
                "(the thread_id in the first event of `codex exec --json`, or the id that ends "
                "the rollout file name)"
            )
        sid = (env.get("CLAUDE_CODE_SESSION_ID") or "").strip()
        if not sid:
            raise SessionError(
                "--session self needs CLAUDE_CODE_SESSION_ID, which Claude Code sets for its "
                "tools; outside Claude Code pass the session id"
            )
        text = sid
    if not _ID_RE.match(text):
        raise SessionError(f"not a session id: {value!r}")
    return text


# --- reading attempts -------------------------------------------------------


def _attempt_model(attempt: Mapping[str, Any]) -> str | None:
    model = attempt.get("model")
    if isinstance(model, Mapping):
        model = model.get("id") or model.get("raw")
    return canonical_model(model) if isinstance(model, str) else None


def _epoch(value: Any) -> float | None:
    norm = parse_ts(value)
    return ts_epoch(norm) if norm is not None else None


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _same_folder(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return os.path.realpath(os.path.expanduser(a)) == os.path.realpath(os.path.expanduser(b))


# --- parts ------------------------------------------------------------------


def _parse(harness: str, path: Path | list[Path]) -> RunRecord:
    """The ingest record for a session; an empty one when it logged no usage.

    A session that made no model request is still the session the attempt
    ran in, so it matches with zero tokens and no model; its cost carries no
    dollars (see `costs.cost_record`).
    """
    record = (claude_code if harness == "claude-code" else codex).parse_session(path)
    if record is not None:
        return record
    paths = path if isinstance(path, list) else [path]
    stamps = sorted(e for p in paths for e in (_epoch(line.get("timestamp")) for line in iter_jsonl(p)) if e is not None)
    return RunRecord(
        run_id="",
        harness=harness,
        model=None,
        effort=None,
        tokens=Tokens(),
        wall_s=(stamps[-1] - stamps[0]) if stamps else 0.0,
        ts=_utc(stamps[0]) if stamps else None,
        workspace=None,
        session_path=str(paths[0]),
    )


def _record_span(record: RunRecord) -> tuple[float | None, float | None]:
    start = _epoch(record.ts)
    if start is None:
        return None, None
    return start, start + float(record.wall_s or 0.0)


def _parts_of(
    record: RunRecord, *, session: str, role: str, harness: str, path: Path | list[Path], parent: str | None
) -> list[dict[str, Any]]:
    # `path` is the session's file, or every file of a resumed Codex thread.
    files = [str(f) for f in (path if isinstance(path, list) else [path])]
    unsplit = None
    if record.tokens_by_model:
        by_model = record.tokens_by_model
    elif record.tokens.total:
        # The parser could not split a thread that switched models: its
        # tokens are kept whole under no model, never given to one of them.
        by_model = {None: record.tokens}
        unsplit = "the thread switched models and its tokens could not be split by model"
    else:
        by_model = {record.model: record.tokens}
    start, end = _record_span(record)
    parts = []
    for model, tokens in by_model.items():
        part = {
            "session": session,
            "role": role,
            "harness": harness,
            "model": model,
            "tokens": tokens,
            "path": files[0],
            "files": files,
            "parent": parent,
            "started_at": _utc(start) if start is not None else None,
            "ended_at": _utc(end) if end is not None else None,
        }
        if unsplit:
            part["unpriced"] = unsplit
            part["unsplit"] = True
        parts.append(part)
    return parts


def _build_match(
    harness: str,
    path: Path | list[Path],
    sid: str,
    tier: str,
    record: RunRecord,
    children: list[tuple[str, str, Path | list[Path], RunRecord]],
    reason: str,
) -> Match:
    parts = _parts_of(record, session=sid, role="session", harness=harness, path=path, parent=None)
    if isinstance(path, list):
        path = path[0]
    for child_id, parent_id, child_path, child_record in children:
        parts.extend(
            _parts_of(child_record, session=child_id, role="child", harness=harness, path=child_path, parent=parent_id)
        )
    total = Tokens()
    models: list[str] = []
    for part in parts:
        t = part["tokens"]
        total.in_ += t.in_
        total.cache_read += t.cache_read
        total.cache_write += t.cache_write
        total.out += t.out
        if part["model"] and part["model"] not in models:
            models.append(part["model"])
    starts = [s for s in (_record_span(r)[0] for r in [record, *(c[3] for c in children)]) if s is not None]
    ends = [e for e in (_record_span(r)[1] for r in [record, *(c[3] for c in children)]) if e is not None]
    return Match(
        harness=harness,
        session_path=path,
        session_id=sid,
        tier=tier,
        tokens=total,
        model=record.model,
        effort=record.effort,
        started_at=_utc(min(starts)) if starts else "",
        ended_at=_utc(max(ends)) if ends else "",
        children=[c[0] for c in children],
        parts=parts,
        models=models,
        reason=reason,
    )


# --- Claude Code --------------------------------------------------------------


def _claude_files(sid: str, roots: LogRoots) -> list[Path]:
    hits: list[Path] = []
    for root in roots.claude:
        if root.is_dir():
            hits.extend(p for p in sorted(root.glob(f"*/{sid}.jsonl")) if p.is_file())
    return hits


def _embedded_agents(path: Path) -> set[str]:
    """Sub-agent ids whose turns the parent file already embeds as sidechains."""
    ids: set[str] = set()
    for line in iter_jsonl(path):
        if line.get("isSidechain") is True and isinstance(line.get("agentId"), str):
            ids.add(line["agentId"])
    return ids


def _claude_children(path: Path, sid: str, record: RunRecord) -> list[tuple[str, str, Path, RunRecord]]:
    folder = path.parent / sid / "subagents"
    if not folder.is_dir():
        return []
    # A parent that embeds a sub-agent's turns has counted them already.
    embedded = _embedded_agents(path) if record.subagents else set()
    children = []
    for child_path in sorted(folder.rglob("agent-*.jsonl")):
        agent_id = child_path.stem[len("agent-"):]
        if agent_id not in embedded:
            children.append((agent_id, sid, child_path, _parse("claude-code", child_path)))
    return children


def _claude_exact(sid: str, roots: LogRoots, cwd: str | None) -> tuple[Match | None, str]:
    paths = _claude_files(sid, roots)
    if len(paths) > 1 and cwd:
        paths = [p for p in paths if _first_value(p, "cwd") and _same_folder(_first_value(p, "cwd"), cwd)] or paths
    if not paths:
        return None, f"claude-code session {sid} not found under {_names(roots.claude)}"
    if len(paths) > 1:
        return None, f"claude-code session {sid} has {len(paths)} files in different projects"
    record = _parse("claude-code", paths[0])
    children = _claude_children(paths[0], sid, record)
    return _build_match("claude-code", paths[0], sid, "verified", record, children, "session id given and found"), ""


def _first_value(path: Path, key: str, limit: int = 200) -> Any:
    for n, line in enumerate(iter_jsonl(path)):
        value = line.get(key)
        if value:
            return value
        if n >= limit:
            break
    return None


def _claude_slug(cwd: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(os.path.expanduser(cwd)))


def _claude_candidates(cwd: str, lo: float, hi: float, roots: LogRoots) -> list[tuple[str, Path]]:
    slugs = {_claude_slug(cwd), re.sub(r"[^A-Za-z0-9]", "-", cwd)}
    found = []
    for root in roots.claude:
        for slug in slugs:
            folder = root / slug
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.jsonl")):
                try:
                    if path.stat().st_mtime < lo:
                        continue  # last written before the window opened
                except OSError:
                    continue
                start = _epoch(_first_value(path, "timestamp"))
                if start is None or not lo <= start <= hi:
                    continue
                if not _same_folder(_first_value(path, "cwd"), cwd):
                    continue
                found.append((path.stem, path))
    return found


# --- Codex ------------------------------------------------------------------


def _codex_meta(path: Path) -> dict[str, Any] | None:
    """`session_meta.payload` from the rollout's first line, or None."""
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
        line = json.loads(first)
    except (OSError, ValueError):
        return None
    if not isinstance(line, dict) or line.get("type") != "session_meta":
        return None
    payload = line.get("payload")
    return payload if isinstance(payload, dict) else None


def _codex_files(roots: LogRoots, days: Iterable[date] | None = None, sid: str = "*") -> list[Path]:
    """Rollout files, in dated folders for `days` (every day when None), plus flat roots."""
    pattern = f"rollout-*-{sid}.jsonl"
    out: list[Path] = []
    for root in roots.codex:
        if not root.is_dir():
            continue
        if days is None:
            out.extend(root.rglob(pattern))
            continue
        wanted = sorted(set(days))
        for d in wanted:
            out.extend((root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}").glob(pattern))
        for path in root.glob(pattern):  # flat layout (archived_sessions)
            m = _ROLLOUT_RE.match(path.name)
            if m and date(int(m.group(1)), int(m.group(2)), int(m.group(3))) in wanted:
                out.append(path)
    return sorted(set(out), key=lambda p: p.name)


def _days(lo: float, hi: float) -> list[date]:
    first = datetime.fromtimestamp(lo, timezone.utc).date() - timedelta(days=1)
    last = datetime.fromtimestamp(hi, timezone.utc).date() + timedelta(days=1)
    return [first + timedelta(days=n) for n in range((last - first).days + 1)]


def _is_codex_child(meta: Mapping[str, Any]) -> bool:
    return parent_session_id(dict(meta)) is not None


def _codex_children(sid: str, lo: float, hi: float, roots: LogRoots) -> list[tuple[str, str, list[Path], RunRecord]]:
    metas = []
    for path in _codex_files(roots, _days(lo, hi)):
        meta = _codex_meta(path)
        if meta and meta.get("id") and _is_codex_child(meta):
            metas.append((str(meta["id"]), parent_session_id(meta), str(meta.get("session_id") or ""), path))
    family = {sid}
    changed = True
    while changed:  # transitive: a grandchild names its direct parent
        changed = False
        for own, parent, root, _path in metas:
            if own not in family and (parent in family or root == sid):
                family.add(own)
                changed = True
    # One child per thread, with every file of it (a child resumed later).
    seen: dict[str, str] = {}
    for own, parent, _root, _path in metas:
        if own in family and own != sid:
            seen.setdefault(own, parent or sid)
    children = []
    for own, parent in seen.items():
        paths = _codex_thread_files(own, roots) or [m[3] for m in metas if m[0] == own]
        children.append((own, parent, paths, _codex_record(own, paths)))
    return children


def _codex_thread_files(sid: str, roots: LogRoots) -> list[Path]:
    """Every rollout file of thread `sid`, first file first.

    A resumed thread continues in a new file with the same id, possibly days
    later, so the whole tree is searched (by file name, then `session_meta`):
    a named session is the whole session whatever the attempt's times.
    """
    return [p for p in _codex_files(roots, None, sid) if (_codex_meta(p) or {}).get("id") == sid]


def _codex_record(sid: str, paths: list[Path]) -> RunRecord:
    # More than one file with one id is a thread resumed into a new file.
    return _parse("codex", paths if len(paths) > 1 else paths[0])


def _codex_exact(sid: str, roots: LogRoots) -> tuple[Match | None, str]:
    paths = _codex_thread_files(sid, roots)
    if not paths:
        return None, f"codex session {sid} not found under {_names(roots.codex)}"
    record = _codex_record(sid, paths)
    start, end = _record_span(record)
    children = _codex_children(sid, start, end + WINDOW_S, roots) if start is not None else []
    return _build_match("codex", paths, sid, "verified", record, children, "session id given and found"), ""


def _codex_candidates(cwd: str, lo: float, hi: float, roots: LogRoots) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for path in _codex_files(roots, _days(lo, hi)):
        meta = _codex_meta(path)
        if not meta or not meta.get("id") or _is_codex_child(meta):
            continue
        start = _epoch(meta.get("timestamp"))
        if start is None or not lo <= start <= hi:
            continue
        if _same_folder(meta.get("cwd"), cwd):
            found.setdefault(str(meta["id"]), path)  # one candidate per thread
    return list(found.items())


def session_cwd(harness: str, path: Path) -> str | None:
    """The working folder a session file records, or None."""
    if harness == "codex":
        value = (_codex_meta(path) or {}).get("cwd")
    else:
        value = _first_value(path, "cwd")
    return value if isinstance(value, str) and value else None


# --- entry points -------------------------------------------------------------


def _names(paths: list[Path]) -> str:
    return ", ".join(str(p) for p in paths) or "no log folder"


def explain_match(
    attempt: Mapping[str, Any], *, roots: LogRoots, exclude: Iterable[str] = ()
) -> tuple[Match | None, str]:
    """`match_attempt` plus the reason, so an unmatched attempt can say why."""
    harness = attempt.get("harness")
    harnesses = (harness,) if harness in HARNESSES else HARNESSES
    if harness and harness not in HARNESSES:
        return None, f"harness {harness!r} writes no session logs loopmath reads"
    start = _epoch(attempt.get("started_at"))

    session = attempt.get("session")
    if session:
        try:
            sid = resolve_session(str(session), harness if isinstance(harness, str) else None, env={})
        except SessionError as exc:
            return None, str(exc)
        reasons = []
        for h in harnesses:
            if h == "claude-code":
                match, why = _claude_exact(sid, roots, attempt.get("cwd"))
            else:
                match, why = _codex_exact(sid, roots)
            if match is not None:
                return match, match.reason
            reasons.append(why)
        return None, "; ".join(reasons)

    if start is None:
        return None, "no session id and no started_at, so no window to search"
    cwd = attempt.get("cwd")
    if not cwd:
        return None, "no session id and no cwd, so no working folder to search"
    end = _epoch(attempt.get("ended_at"))
    lo = start - WINDOW_S
    hi = (end if end is not None and end >= start else start) + WINDOW_S
    want_model = _attempt_model(attempt)
    skip = set(exclude)

    candidates: list[tuple[str, str, Path | list[Path], RunRecord]] = []
    for h in harnesses:
        found = _claude_candidates(cwd, lo, hi, roots) if h == "claude-code" else _codex_candidates(cwd, lo, hi, roots)
        for sid, path in found:
            if sid in skip:
                continue
            if h == "codex":
                paths = _codex_thread_files(sid, roots) or [path]
                path, record = paths, _codex_record(sid, paths)
            else:
                record = _parse(h, path)
            if want_model and record.model != want_model:
                continue
            candidates.append((h, sid, path, record))
    if not candidates:
        return None, (
            f"no session id; no {'/'.join(harnesses)} session started in {cwd} between "
            f"{_utc(lo)} and {_utc(hi)}" + (f" on {want_model}" if want_model else "")
        )
    if len(candidates) > 1:
        return None, (
            f"no session id; {len(candidates)} candidate sessions in {cwd} between "
            f"{_utc(lo)} and {_utc(hi)}, so no match"
        )
    h, sid, path, record = candidates[0]
    if h == "claude-code":
        children = _claude_children(path, sid, record)
    else:
        s, e = _record_span(record)
        children = _codex_children(sid, s, e + WINDOW_S, roots) if s is not None else []
    reason = "one session in the working folder and window" + (f" on {want_model}" if want_model else "")
    return _build_match(h, path, sid, "heuristic", record, children, reason), reason


def match_attempt(attempt: dict[str, Any], *, roots: LogRoots) -> Match | None:
    """The session `attempt` ran in (spec 03 section 6) with its artifacts, or None."""
    from .artifacts import artifacts_for

    match = explain_match(attempt, roots=roots)[0]
    if match is not None:
        match.artifacts = artifacts_for(match)
    return match
