"""Launch lineage helpers and joins. Origin corroboration lives in ``launch_origin``."""
from __future__ import annotations

import os
import re
from pathlib import Path

from .schema import GraphEdge, GraphNode

# `codex exec`, `codex e`, or `codex -q` at a command position. The lookbehind
# rejects paths (`~/.codex/sessions`) and names (`herdr-codex`), which is what
# made a plain "codex" substring match 158 commands in swarm A.
_CODEX_LAUNCH_RE = re.compile(
    r"(?<![\w./-])codex\s+(?:(?:-[\w-]+(?:\s+\S+)?\s+)*)(?:exec|e|-q|--quiet)\b"
)
# `claude -p ...` / `claude --print ...`: a non-interactive top-level session.
_CLAUDE_LAUNCH_RE = re.compile(r"(?<![\w./-])claude\s+(?:(?:-[\w-]+(?:\s+\S+)?\s+)*)(?:-p|--print)\b")
_LAUNCH_RES = {"codex": _CODEX_LAUNCH_RE, "claude-code": _CLAUDE_LAUNCH_RE}

LAUNCH_WINDOW_S = (0.0, 120.0)
BACKGROUND_WINDOW_S = (0.0, 600.0)
SCOPE_PAD_S = 3600.0
CONTAIN_SLACK_S = 2.0
LONG_CALL_S = 20.0

SCRIPT_EXTS = (".sh", ".bash", ".zsh", ".py")

_INTERPRETERS = {"bash", "zsh", "sh", "python", "python3", "source", "."}
_PREFIX_WORDS = {"nohup", "time", "exec", "env", "setsid", "caffeinate", "do", "then", "else", "{", "!", "\\"}
_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")
_SEGMENT_SPLIT_RE = re.compile(r"\n|;|&&|\|\||\||(?<![&>])&(?!&)|\(|\)")
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
_BACKGROUND_RE = re.compile(r"(?<![&>|\d])&(?![&>])|(?<![\w./-])(?:nohup|setsid)\s")
_UNRESOLVABLE_RE = re.compile(r"[*?\[\]{}$`]")
_COMMENT_START = frozenset(" \t\n;(&|")
REJECT_REASONS = ("unresolved", "no_root", "missing", "outside_root")


def detect_launch(command: str) -> frozenset:
    """Harness names (`codex`, `claude-code`) whose CLI `command` invokes directly."""
    return frozenset(h for h, rx in _LAUNCH_RES.items() if rx.search(command))


def _strip_heredocs(command: str) -> str:
    """Drop heredoc bodies: a `cat > tools/x.sh <<'EOF'` body is file content, not
    commands the session ran."""
    lines = command.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = _HEREDOC_RE.search(line)
        i += 1
        if m:
            term = m.group(2)
            while i < len(lines) and lines[i].strip() != term:
                i += 1
            i += 1
    return "\n".join(out)


def _mask_quotes_and_comments(command: str) -> str:
    """`command` with the text inside single quotes, double quotes and `#` comments
    replaced by spaces (same length), so shell operators are only seen where the
    shell sees them: `echo 'nohup x &'` and `x # run with &` background nothing."""
    out = list(command)
    n = len(command)
    i = 0
    quote: str | None = None
    while i < n:
        ch = command[i]
        if quote is None:
            if ch == "\\" and i + 1 < n:
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if ch in "'\"":
                quote = ch
                out[i] = " "
            elif ch == "#" and (i == 0 or command[i - 1] in _COMMENT_START):
                while i < n and command[i] != "\n":
                    out[i] = " "
                    i += 1
                continue
        elif quote == "'":
            if ch == "'":
                quote = None
            out[i] = " "
        else:
            if ch == "\\" and i + 1 < n:
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if ch == '"':
                quote = None
            out[i] = " "
        i += 1
    return "".join(out)


def is_backgrounded(command: str) -> bool:
    """True when the command backgrounds work (`cmd &`, `nohup cmd`, `setsid cmd`),
    so its Bash interval says nothing about when the work finished. Operators inside
    quoted strings or comments do not count."""
    return bool(_BACKGROUND_RE.search(_mask_quotes_and_comments(_strip_heredocs(command))))


def _script_tokens(command: str) -> list[str]:
    """Script paths the command runs at a command position: `tools/x.sh`, `./x.sh`,
    `bash x.sh`, `zsh x.sh`, `python x.py`, with `nohup`/`timeout N`/`VAR=v`
    prefixes skipped. `grep x tools/x.sh` or `python3 - tools/x.sh` are not runs."""
    found: list[str] = []
    for seg in _SEGMENT_SPLIT_RE.split(_strip_heredocs(command)):
        toks = [t.strip("'\"") for t in seg.split()]
        toks = [t for t in toks if t]
        i = 0
        while i < len(toks) and (toks[i] in _PREFIX_WORDS or _ASSIGN_RE.match(toks[i]) or toks[i] == "timeout"):
            if toks[i] == "timeout":
                i += 1
                while i < len(toks) and toks[i].startswith("-"):
                    i += 1
            i += 1
        if i >= len(toks) or toks[i].startswith("#"):
            continue
        head = toks[i]
        if head.endswith(SCRIPT_EXTS):
            found.append(head)
            continue
        if os.path.basename(head) in _INTERPRETERS or re.fullmatch(r"python3\.\d+", os.path.basename(head)):
            j = i + 1
            while j < len(toks) and toks[j].startswith("-") and toks[j] != "-":
                j += 1
            if j < len(toks) and toks[j].endswith(SCRIPT_EXTS):
                found.append(toks[j])
    return found


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _workspace_root() -> Path:
    """Default parent directory used to resolve workspace names."""
    return Path.home() / "Workspace"


def script_root(workspace: str | None, worktree: str | None) -> str | None:
    """Real path of the one directory a session's scripts may live under: the
    record's `worktree` when it exists on disk, else the workspace name resolved
    under `_workspace_root()` when that exists, never both (spec section 4, clarified
    13:40). None when neither is available."""
    if worktree and os.path.isdir(worktree):
        return os.path.realpath(worktree)
    if workspace:
        cand = _workspace_root() / workspace
        if os.path.isdir(cand):
            return os.path.realpath(cand)
    return None


def _resolve_script(token: str, cwd: str | None, root: str | None) -> tuple[Path | None, str | None]:
    """`(path, None)` for the script file `token` names, or `(None, reason)` with
    `reason` one of `REJECT_REASONS`: `unresolved` (globs or variables in the
    token), `no_root` (no script root for the session), `missing` (no such file),
    `outside_root` (a file whose real path lies outside `root`, such as
    `../other-ws/tools/review.sh` or `tools/review.sh` from a cwd outside it). A
    relative path is taken from the entry's cwd when that directory still exists,
    else from the root (a deleted worktree whose workspace is checked out elsewhere)."""
    if _UNRESOLVABLE_RE.search(token):
        return None, "unresolved"
    if root is None:
        return None, "no_root"
    tok = os.path.expanduser(token)
    if os.path.isabs(tok):
        p = os.path.normpath(tok)
    elif cwd and os.path.isdir(cwd):
        p = os.path.normpath(os.path.join(cwd, tok))
    else:
        p = os.path.normpath(os.path.join(root, tok))
    if not os.path.isfile(p):
        return None, "missing"
    if not _under(os.path.realpath(p), root):
        return None, "outside_root"
    return Path(p), None


_script_text_cache: dict[tuple, frozenset] = {}


def _script_harnesses(path: Path) -> frozenset | None:
    """Harnesses the whole text of `path` launches, or None when the file cannot be
    read. Hits are memoised by `(path, mtime, size)`; failures are not."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _script_text_cache.get(key)
    if hit is None:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            return None
        hit = detect_launch(text)
        _script_text_cache[key] = hit
    return hit


def find_script_launches(
    command: str,
    cwd: str | None,
    workspace: str | None = None,
    worktree: str | None = None,
    *,
    meta: dict | None = None,
) -> dict[str, frozenset]:
    """Scripts the command runs whose text launches a harness: resolved path to the
    harness set, only for scripts with a hit, in command order. When `meta` is
    given, a script that resolves but cannot be read is counted in
    `meta["scripts_unreadable"]` and listed in `meta["scripts_unreadable_paths"]`,
    and every rejected candidate in `meta["scripts_rejected_<reason>"]`."""
    return _scan_scripts(command, cwd, workspace, worktree, meta)[0]


def _scan_scripts(command, cwd, workspace, worktree, meta) -> tuple[dict[str, frozenset], int]:
    """`find_script_launches` plus the number of times the command runs a launching
    script (`tools/review.sh A ...; tools/review.sh B ...` is two)."""
    out: dict[str, frozenset] = {}
    root = script_root(workspace, worktree)
    seen: dict[str, frozenset | None] = {}
    n_runs = 0
    for tok in _script_tokens(command):
        p, reason = _resolve_script(tok, cwd, root)
        if p is None:
            if meta is not None:
                key = f"scripts_rejected_{reason}"
                meta[key] = meta.get(key, 0) + 1
            continue
        if str(p) in seen:
            if seen[str(p)]:
                n_runs += 1
            continue
        h = _script_harnesses(p)
        seen[str(p)] = h
        if h is None:
            if meta is not None:
                meta["scripts_unreadable"] = meta.get("scripts_unreadable", 0) + 1
                paths = meta.setdefault("scripts_unreadable_paths", [])
                if str(p) not in paths:
                    paths.append(str(p))
                    paths.sort()
        elif h:
            out[str(p)] = h
            n_runs += 1
    return out, n_runs


def script_launches(command: str, cwd: str | None, workspace: str | None = None, worktree: str | None = None) -> frozenset:
    """Harnesses launched through a script the command runs (`tools/x.sh`, `bash x.sh`,
    `python x.py`) that lies under the session's script root (`worktree` when it
    exists, else `_workspace_root()/<workspace>`), resolved from `cwd` (the Bash
    entry's)."""
    out: frozenset = frozenset()
    for h in find_script_launches(command, cwd, workspace, worktree).values():
        out = out | h
    return out


def _epoch(ts):
    from .scan import epoch

    return epoch(ts)


class _Call:
    """One Bash entry that may have launched sessions."""

    __slots__ = ("st", "en", "cmd", "src", "src_ws", "direct", "scripts", "launch", "background", "capacity", "launched")

    def __init__(self, st, en, cmd, src, src_ws, direct, scripts, script_runs=0):
        self.st = st
        self.en = en
        self.cmd = cmd
        self.src = src
        self.src_ws = src_ws
        self.direct = direct
        self.scripts = scripts
        launch = frozenset(direct)
        for h in scripts.values():
            launch = launch | h
        self.launch = launch
        self.background = is_backgrounded(cmd)
        # Launches the command text names: direct CLI invocations plus runs of a
        # launching script. The most sessions the call claims through the window.
        text = _strip_heredocs(cmd)
        self.capacity = max(1, sum(len(rx.findall(text)) for rx in _LAUNCH_RES.values()) + script_runs)
        self.launched: list[tuple[str, bool]] = []  # (session id, contained)

    def contains(self, ce: float) -> bool:
        """The interval holds `ce`; never for a backgrounded call, whose end is when
        the shell returned, not when the launched work did, and never for a call
        that started after the session."""
        if self.background or self.en is None:
            return False
        return self.st <= ce <= self.en + CONTAIN_SLACK_S

    def window(self) -> tuple[float, float]:
        return BACKGROUND_WINDOW_S if self.background else LAUNCH_WINDOW_S

    def via_script(self, harness: str) -> str | None:
        if harness in self.direct:
            return None
        for path, h in self.scripts.items():
            if harness in h:
                return path
        return None


def _make_call(b: dict, src: str, src_ws: str | None, worktree: str | None, meta: dict) -> _Call | None:
    st = _epoch(b["ts"])
    if st is None:
        return None
    scripts, script_runs = _scan_scripts(b["command"], b.get("cwd"), src_ws, worktree, meta)
    return _Call(st, _epoch(b["end_ts"]), b["command"], src, src_ws, b["launch"], scripts, script_runs)


def join_launches(
    nodes: dict[str, GraphNode],
    scans: dict[str, dict],
    records: list[dict],
    wanted: set[str] | None,
    *,
    meta: dict | None = None,
) -> tuple[dict[str, str], dict[str, dict], list[GraphEdge]]:
    """Join launch commands to launched sessions. Mutates `nodes`: sets `parent`,
    `launched_by` and `phase` on launched nodes and adds one `external` node per
    launcher outside the requested workspaces. Returns `(launch_cmd, external,
    edges)`: launched node id to its launch command, external launcher id to its
    record info, and the launch edges. Counters go into `meta` when given."""
    from .scan import scan_claude_session  # lazy: scan.py imports detect_launch from here

    meta = meta if meta is not None else {}
    meta.setdefault("scripts_unreadable", 0)
    meta.setdefault("scripts_unreadable_paths", [])
    for reason in REJECT_REASONS:
        meta.setdefault(f"scripts_rejected_{reason}", 0)
    worktrees = {str(r.get("run_id")): r.get("worktree") for r in records if r.get("run_id")}
    calls: list[_Call] = []
    for nid in nodes:
        if nid not in scans:
            continue
        for b in scans[nid]["bash"]:
            c = _make_call(b, nid, nodes[nid].workspace, worktrees.get(nid), meta)
            if c is not None:
                calls.append(c)
    external: dict[str, dict] = {}
    if wanted is not None:
        starts = [_epoch(n.ts) for n in nodes.values()]
        starts = [s for s in starts if s is not None]
        if starts:
            lo = min(starts) - SCOPE_PAD_S
            hi = max(_epoch(n.ts) + (n.wall_s or 0.0) for n in nodes.values() if _epoch(n.ts) is not None) + SCOPE_PAD_S
            for r in records:
                if r.get("workspace") in wanted or str(r.get("harness")) == "codex" or not r.get("session_path"):
                    continue
                rs = _epoch(r.get("ts"))
                if rs is None or rs > hi or rs + float(r.get("wall_s") or 0.0) < lo:
                    continue
                rid = str(r["run_id"])
                for b in scan_claude_session(Path(r["session_path"]))["bash"]:
                    c = _make_call(b, rid, r.get("workspace"), r.get("worktree"), meta)
                    if c is None or not c.launch or c.st < lo or c.st > hi:
                        continue
                    external[rid] = {"workspace": r.get("workspace"), "session_path": r["session_path"]}
                    calls.append(c)
    calls.sort(key=lambda c: (c.st, c.src))

    sessions = [(ce, cnid, c) for cnid, c in nodes.items() if c.source != "subagent" for ce in [_epoch(c.ts)] if ce is not None]
    sessions.sort(key=lambda t: (t[0], t[1]))
    chosen: dict[str, tuple[_Call, bool]] = {}  # session id -> (call, contained)

    # Rule 1: every (session, call) pair where the call names the harness and
    # started at or before the session start, within the window (a call starting
    # after the session never qualifies), ranked by start-time distance, closest
    # first; containment is only a tiebreak between equally close calls. Settled
    # greedily: a session is claimed once; a call keeps taking sessions it contains
    # (a loop) but takes at most `capacity` sessions through the window, and never
    # mixes the two.
    pairs: list[tuple[float, bool, float, str, str, _Call, bool]] = []
    for ce, cnid, c in sessions:
        for t in calls:
            if t.src == cnid or c.harness not in t.launch:
                continue
            lag = ce - t.st
            if lag < 0:
                continue
            contained = t.contains(ce)
            if t.src_ws == c.workspace:
                wlo, whi = t.window()
                if not contained and not (wlo <= lag <= whi):
                    continue
            elif not (contained and c.workspace and c.workspace in t.cmd):
                continue
            pairs.append((lag, not contained, t.st, t.src, cnid, t, contained))
    pairs.sort(key=lambda p: p[:5])
    for _dist, _win, _st, _src, cnid, t, contained in pairs:
        if cnid in chosen:
            continue
        if t.launched and (contained != t.launched[0][1] or (not contained and len(t.launched) >= t.capacity)):
            continue
        chosen[cnid] = (t, contained)
        t.launched.append((cnid, contained))
    # Rule 2, last resort: codex sessions inside a long synchronous same-workspace
    # call naming nothing (`contains` is already False for backgrounded calls and
    # for calls that started after the session).
    for ce, cnid, c in sessions:
        if cnid in chosen or c.source != "codex":
            continue
        longs = [
            t for t in calls
            if t.src != cnid and t.src_ws == c.workspace and not t.launch and t.contains(ce) and t.en - t.st >= LONG_CALL_S
        ]
        if longs:
            t = max(longs, key=lambda t: (t.en - t.st, -t.st, t.src))
            chosen[cnid] = (t, True)
            t.launched.append((cnid, True))

    edges: list[GraphEdge] = []
    launch_cmd: dict[str, str] = {}
    via_script_n = 0
    for ce, cnid, c in sessions:
        if cnid not in chosen:
            continue
        t, contained = chosen[cnid]
        lag = round(ce - t.st, 1)
        script = t.via_script(c.harness) if c.harness in t.launch else None
        if c.harness not in t.launch:
            how = f"inside a running Bash call of {round(t.en - t.st)}s naming no CLI (last resort)"
        elif contained and script:
            how = "inside a running Bash call running a launch script"
        elif contained:
            how = "inside a running Bash call naming the CLI"
        elif script:
            how = f"launch script started {round(lag)}s before session start"
        else:
            how = f"launch command issued {round(lag)}s before session start"
        detail = {"lag_s": lag, "how": how, "command": t.cmd[:240]}
        if script:
            detail["via_script"] = script
            via_script_n += 1
        if len(t.launched) > 1:
            detail["shared_launcher"] = True
        if t.capacity > 1:
            detail["launches_in_call"] = t.capacity
        launch_cmd[cnid] = t.cmd
        c.parent = t.src
        c.launched_by = {"id": t.src, "workspace": t.src_ws, **detail}
        if t.src in external:
            c.phase = "external"
        c.phase_tier = "heuristic"
        edges.append(GraphEdge(t.src, cnid, "launch", "heuristic", detail))
    meta["unmatched_launches"] = sum(1 for t in calls if t.launch and t.src not in external and not t.launched)
    meta["unlaunched_codex"] = sum(1 for nid, n in nodes.items() if n.source == "codex" and nid not in launch_cmd)
    meta["external_launchers"] = sorted(external)
    meta["script_launchers"] = sum(1 for t in calls if t.scripts and not t.direct and t.src not in external)
    meta["launches_via_script"] = via_script_n
    meta["shared_launchers"] = sum(1 for t in calls if len(t.launched) > 1 and t.src not in external)
    for rid, info in sorted(external.items()):
        if any(e.src == rid for e in edges):
            nodes[rid] = GraphNode(id=rid, harness="claude-code", source="external", session_path=info["session_path"], workspace=info["workspace"], role="external", role_tier="heuristic", role_evidence="session outside the requested workspaces that launched sessions inside them", phase="external", phase_tier="heuristic")

    corroborate_origins(nodes, scans, records, launch_cmd, edges, meta)
    return launch_cmd, external, edges

# Compatibility re-exports for L2 callers at ``loopmath.graph.launch``.
from .launch_origin import (
    CWD_ELSEWHERE, CWD_IN_WORKTREE, CWD_OUTSIDE_WORKTREE, CWD_UNKNOWN,
    CWD_WORKSPACE_NAME, INTERACTIVE_ORIGINS, LAUNCHED_BUCKETS,
    ORIGIN_EXEC_CWD_UNKNOWN, ORIGIN_EXEC_ELSEWHERE, ORIGIN_EXEC_OUTSIDE_WORKTREE,
    ORIGIN_MISSING, ORIGIN_NOT_LAUNCHED, ORIGIN_UNREADABLE, ORIGIN_UNSCANNED,
    ORIGIN_UNSUPPORTED, UNLAUNCHED_BUCKETS, _origin_cache, codex_origin,
    codex_origin_status, corroborate_origins, cwd_in_workspace, cwd_placement,
    origin_pair, origin_pair_label, origin_says_exec, origin_says_interactive,
    read_session_meta, scan_origin_is_complete,
)
