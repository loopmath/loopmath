"""Claude Code and Codex history to OCP v0.3 runs (configuration.source: habit).

Owner: lane 03. Spec: design/0.1/02-commands.md section 4, 08-lanes.md section 3.

The pipeline is today's `loopmath graph` pipeline over every workspace: discover,
parse (cached), grade, price, extract. The graph is then cut into session groups:
a root session plus everything it started (Claude Code sub-agents, Codex child
threads, and `codex exec` or `claude -p` sessions launched from a Bash call).
Artifact edges do not join groups: two sessions that touched the same file are
not one task.

Prompt text is read only through the existing head readers
(`graph.dataset_nodes.claude_head` and `codex_head`), kept in memory on the
group for the labeller, and never written to a run document or printed.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .. import gitwalk
from ..graph.schema import Artifact, Graph, GraphEdge, GraphNode

# Edges that mean "this session started that one". Artifact edges are left out.
LINK_KINDS = ("spawn", "launch")
PROMPT_LIMIT = 400
MODEL_TOKENS_KEY = "dev.loopmath.model_tokens"  # D67: an attempt's per-model token split, in cost.ext
_OCP_TOKEN_FIELDS = (("in", "input_tokens"), ("cache_read", "cached_input_tokens"),
                     ("cache_write", "cache_creation_tokens"), ("out", "output_tokens"))
UNLABELLED_MODEL = "unknown"  # tokens under no model label; no price table has this row
_FAR_FUTURE = _dt.datetime.max.replace(tzinfo=_dt.timezone.utc)


def parse_ts(value: Any) -> _dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=_dt.timezone.utc)


def local_iso(ts: _dt.datetime | None) -> str | None:
    return ts.astimezone().isoformat(timespec="seconds") if ts else None


@dataclass
class History:
    since_days: float
    records: list[dict]
    graph: Graph
    files: dict[str, int]  # session files discovered in the window, per harness
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_id = {str(r.get("run_id")): r for r in self.records if r.get("run_id")}

    def record(self, node_id: str) -> dict | None:
        return self._by_id.get(node_id)

    @property
    def by_id(self) -> dict[str, dict]:
        return self._by_id


def load_history(since_days: float, *, logs: str | Path | None = None,
                 progress: Callable[[str, int, int], None] | None = None,
                 stage: Callable[[str], None] | None = None) -> History:
    """discover -> parse (cached) -> grade -> price -> extract, over every workspace.
    `stage` hears one line before the graph step, the slow one on a long history."""
    from .. import grade as grade_mod
    from .. import ingest
    from .. import price as price_mod
    from ..graph import extract

    found = ingest._discovery_manifest(ingest.discover(logs, since_days=since_days))
    files = dict(Counter(h for h, *_ in found))
    records, diag = ingest.parse_all(logs, progress=progress, since_days=since_days, discovered=found)
    records, _coverage = grade_mod.grade_all(records)
    records, _warnings = price_mod.price_all(records, price_mod.load_prices())
    if stage is not None:
        stage(f"linking {len(records):,} sessions and the files they touched; "
              "with thousands of sessions this takes several minutes")
    graph = extract(records)
    return History(since_days=since_days, records=records, graph=graph, files=files, diagnostics=diag)


@dataclass
class SessionGroup:
    id: str                      # grp_ + 12 hex of SHA-256 over the root node id
    root: str                    # graph node id of the root session
    nodes: list[GraphNode]       # root first, then by start time
    edges: list[GraphEdge]
    artifacts: list[Artifact]
    started_at: _dt.datetime | None
    ended_at: _dt.datetime | None
    cwd: str | None = None
    repo: str = "unknown"
    prompt: str | None = field(default=None, repr=False)  # in memory only, for the labeller

    @property
    def harnesses(self) -> list[str]:
        return sorted({n.harness for n in self.nodes if n.harness})

    @property
    def models(self) -> list[str]:
        return sorted({n.model for n in self.nodes if n.model})

    @property
    def tokens(self) -> int:
        return sum(sum(v for v in (n.tokens or {}).values() if isinstance(v, int) and not isinstance(v, bool))
                   for n in self.nodes)

    @property
    def usd(self) -> float | None:
        """Dollars in the logs; None when any session with tokens is unpriced."""
        total = 0.0
        for n in self.nodes:
            if n.usd is None:
                if n.tokens:
                    return None
                continue
            total += float(n.usd)
        return total

    @property
    def wall_s(self) -> float | None:
        if self.started_at and self.ended_at:
            return max(0.0, (self.ended_at - self.started_at).total_seconds())
        return None

    def subgraph(self) -> Graph:
        return Graph(nodes=list(self.nodes), edges=list(self.edges), artifacts=list(self.artifacts),
                     meta={"onboard_group": self.id})


def group_id(root: str) -> str:
    return "grp_" + hashlib.sha256(root.encode("utf-8")).hexdigest()[:12]


def _record_parent(record: dict | None) -> str | None:
    """The parent session id a record declares itself (D28: lane 2's `parent_session`, a bare
    Codex thread id or Claude Code session id)."""
    value = record.get("parent_session") if isinstance(record, dict) else None
    return value if isinstance(value, str) and value else None


def group_artifact(art: Artifact, node_ids: set[str]) -> Artifact | None:
    """The artifact as one group saw it, or None when no node of the group wrote it. A file that
    sessions of several groups touched keeps only this group's writers, consumers and writes, so
    the run names no attempt outside itself (OCP E118). The producer and first write are the
    group's first writer's; a read count that included other groups becomes unknown."""
    writers = [w for w in art.writers if w in node_ids]
    if not writers:
        return None
    consumers = [c for c in art.consumers if c in node_ids]
    if writers == list(art.writers) and consumers == list(art.consumers):
        return art
    writes = [w for w in art.writes if isinstance(w, dict) and w.get("node") in node_ids]
    first = next((w.get("ts") for w in writes if w.get("node") == writers[0] and w.get("ts")), None)
    return replace(art, producer=writers[0], writers=writers, consumers=consumers, writes=writes,
                   first_write_ts=art.first_write_ts if writers[0] == art.producer else first,
                   n_writes=len(writes) if art.writes else 0,
                   n_reads=art.n_reads if consumers == list(art.consumers) else None)


def group_sessions(graph: Graph, records: dict[str, dict] | None = None) -> list[SessionGroup]:
    """Cut the graph into session groups (root plus descendants), oldest first."""
    ids = {n.id for n in graph.nodes}
    parent: dict[str, str] = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    has_parent: set[str] = set()
    for e in graph.edges:
        if e.kind in LINK_KINDS and e.src in ids and e.dst in ids and e.src != e.dst:
            union(e.src, e.dst)
            has_parent.add(e.dst)
    for n in graph.nodes:
        for p in (n.parent, _record_parent((records or {}).get(n.id))):
            if not p:
                continue
            pid = next((c for c in (p, f"cx_{p}", f"cc_{p}") if c in ids), None)
            if pid and pid != n.id:
                union(pid, n.id)
                has_parent.add(n.id)

    members: dict[str, list[GraphNode]] = {}
    for n in graph.nodes:
        members.setdefault(find(n.id), []).append(n)
    # Each edge and artifact is looked at once, not once per group: an edge belongs to the
    # group holding both its ends, an artifact to every group holding one of its writers.
    group_of = {i: find(i) for i in ids}
    edges_of: dict[str, list[GraphEdge]] = {}
    for e in graph.edges:
        g = group_of.get(e.src)
        if g is not None and group_of.get(e.dst) == g:
            edges_of.setdefault(g, []).append(e)
    arts_of: dict[str, list[Artifact]] = {}
    for x in graph.artifacts:
        for g in dict.fromkeys(group_of[w] for w in x.writers if w in group_of):
            arts_of.setdefault(g, []).append(x)

    def start(n: GraphNode) -> _dt.datetime:
        return parse_ts(n.ts) or _FAR_FUTURE

    groups: list[SessionGroup] = []
    for key, nodes in members.items():
        roots = [n for n in nodes if n.id not in has_parent] or nodes
        root = min(roots, key=lambda n: (start(n), n.id))
        rest = sorted((n for n in nodes if n.id != root.id), key=lambda n: (start(n), n.id))
        ordered = [root, *rest]
        node_ids = {n.id for n in ordered}
        edges = edges_of.get(key, [])
        arts = [a for a in (group_artifact(x, node_ids) for x in arts_of.get(key, ())) if a is not None]
        starts = [t for t in (parse_ts(n.ts) for n in ordered) if t]
        ends = [t + _dt.timedelta(seconds=float(n.wall_s)) for n in ordered
                if (t := parse_ts(n.ts)) and isinstance(n.wall_s, (int, float))]
        groups.append(SessionGroup(
            id=group_id(root.id), root=root.id, nodes=ordered, edges=edges, artifacts=arts,
            started_at=min(starts) if starts else None,
            ended_at=max(ends) if ends else (max(starts) if starts else None),
        ))
    groups.sort(key=lambda g: (g.started_at or _FAR_FUTURE, g.id))
    return groups


def in_window(groups: list[SessionGroup], since_days: float, *,
              now: _dt.datetime | None = None) -> tuple[list[SessionGroup], int]:
    """(groups that started inside the window, how many started before it). Discovery
    keeps any session file touched in the window, so a session begun months ago and
    resumed last week is read; its group is left out here, so `--since` means when the
    work began. A group with no start time is kept."""
    cutoff = (now or _dt.datetime.now(_dt.timezone.utc)) - _dt.timedelta(days=since_days)
    kept = [g for g in groups if g.started_at is None or g.started_at >= cutoff]
    return kept, len(groups) - len(kept)


def read_heads(groups: list[SessionGroup], *, head: Callable[[GraphNode], dict] | None = None) -> None:
    """Fill each group's cwd, repo and (in memory) first prompt from its root session."""
    head = head or _head
    for g in groups:
        info = head(g.nodes[0]) if g.nodes else {}
        g.cwd = info.get("cwd")
        prompt = info.get("prompt")
        g.prompt = prompt.strip() if isinstance(prompt, str) and prompt.strip() else None
        g.repo = repo_for(g.cwd)


def _head(node: GraphNode) -> dict:
    from ..graph.dataset_nodes import claude_head, codex_head

    path = Path(node.session_path) if node.session_path else None
    if path is None or not path.is_file():
        return {}
    try:
        return codex_head(path) if node.harness == "codex" else claude_head(path)
    except (OSError, ValueError):
        return {}


_REMOTE_RE = re.compile(r"(?:[:/])([^/:]+)/([^/]+?)(?:\.git)?/?$")
_repo_cache: dict[str, str] = {}
_top_cache: dict[str, str | None] = {}     # git's top level, by worktree (or by folder when unsure)
_origin_cache: dict[str, str | None] = {}  # the origin remote, by top level


def remote_name(url: str) -> str | None:
    """`owner/name` from a git remote URL (ssh, https or scp form), lowercased."""
    m = _REMOTE_RE.search(url.strip())
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}".lower()


def _git(args: list[str], cwd: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def repo_for(cwd: str | None) -> str:
    """The repo a session worked in: the git `origin` remote as `owner/name` (local git,
    no network), else the git top-level folder name, else the workspace name, else
    `unknown`. A folder that no longer exists is looked up through its nearest existing
    parent."""
    from ..ingest.base import workspace_name

    if not cwd:
        return "unknown"
    if cwd in _repo_cache:
        return _repo_cache[cwd]
    probe = Path(cwd)
    while not probe.is_dir() and probe != probe.parent:
        probe = probe.parent
    name = None
    if probe.is_dir() and probe != probe.parent and probe != Path.home():
        # Thousands of session folders sit in about a hundred worktrees: git is asked
        # once per worktree and once per top level, and not at all outside any worktree.
        kind, worktree = gitwalk.where(str(probe))
        if kind != "none":
            key = worktree if kind == "repo" else str(probe)
            if key not in _top_cache:
                _top_cache[key] = _git(["rev-parse", "--show-toplevel"], str(probe))
            top = _top_cache[key]
            if top and Path(top) != Path.home():
                if top not in _origin_cache:
                    _origin_cache[top] = _git(["remote", "get-url", "origin"], top)
                url = _origin_cache[top]
                name = (remote_name(url) if url else None) or Path(top).name
    name = name or workspace_name(cwd) or "unknown"
    _repo_cache[cwd] = name
    return name


def group_summary(group: SessionGroup, history: History | None = None, *, prompt_limit: int = PROMPT_LIMIT) -> dict:
    """What the labeller sees for one group: metadata plus the first prompt, cut."""
    kinds: Counter = Counter()
    dirs: Counter = Counter()
    if history is not None:
        for n in group.nodes:
            rec = history.record(n.id) or {}
            kinds.update({str(k): int(v) for k, v in (rec.get("written_file_kinds") or {}).items() if isinstance(v, int)})
            dirs.update(str(d) for d in (rec.get("touched_dirs") or []))
    roles = Counter(n.role or "unlabeled" for n in group.nodes)
    wall = group.wall_s
    return {
        "id": group.id,
        "harness": "+".join(group.harnesses) or "unknown",
        "repo": group.repo,
        "date": group.started_at.date().isoformat() if group.started_at else None,
        "minutes": round(wall / 60.0, 1) if wall is not None else None,
        "sessions": len(group.nodes),
        "models": group.models,
        "roles": dict(roles.most_common()),
        "file_kinds": dict(kinds.most_common(6)),
        "dirs": [d for d, _ in dirs.most_common(5)],
        "prompt": (group.prompt or "")[:prompt_limit] or None,
    }


# ---------------------------------------------------------------- run documents

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _b32(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def run_id_for(group: SessionGroup) -> str:
    """`run_` + a ULID whose time part is the group's start (ms) and whose random part is
    the first 80 bits of SHA-256 over the group id: the same group always gets the same
    run id, so re-running onboard replaces its runs instead of adding copies."""
    ms = int(group.started_at.timestamp() * 1000) if group.started_at else 0
    rand = int.from_bytes(hashlib.sha256(group.id.encode("utf-8")).digest()[:10], "big")
    return "run_" + _b32(ms & ((1 << 48) - 1), 10) + _b32(rand, 16)


def task_id_for(group: SessionGroup) -> str:
    return "tsk_" + hashlib.sha256(("task:" + group.id).encode("utf-8")).hexdigest()[:16]


def fallback_vertex(node: GraphNode, pieces: list[Any]) -> str | None:
    """Node to piece by role, used only when infer gives no `node_vertex` (D25)."""
    role = (node.role or "").lower()
    wanted = {
        "planner": ("plan", "planner"),
        "reviewer": ("review", "reviewer", "referee", "select"),
        "smoke": ("test", "tester"),
    }.get(role, ("implement", "implementer", "worker", "dev"))
    for piece in pieces:
        if str(getattr(piece, "role", "")).lower() in wanted:
            return piece.id
    return pieces[0].id if pieces else None


def model_tokens(record: dict | None) -> dict | None:
    """D67: a record's per-model split (lane 2's `tokens_by_model`) as {model_id: {OCP token fields}},
    or None when the record has no split (one model, the record's own). An empty object says the
    session switched models and its tokens could not be split. Model ids and counts only."""
    parts = record.get("tokens_by_model") if isinstance(record, dict) else None
    if not isinstance(parts, list):
        return None
    out: dict[str, dict] = {}
    for part in parts:
        tokens = part.get("tokens") if isinstance(part, dict) else None
        if not isinstance(tokens, dict):
            return {}  # a malformed part: the split is not known
        into = out.setdefault(str(part.get("model") or UNLABELLED_MODEL), {})
        for src, dst in _OCP_TOKEN_FIELDS:
            v = tokens.get(src)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                into[dst] = into.get(dst, 0) + int(v)
    return out


def run_doc(group: SessionGroup, *, label: dict, labeled_by: dict, config: Any, confidence: float,
            org: str | None = None, records: dict[str, dict] | None = None) -> dict:
    """One OCP v0.3 run document for a labelled group.

    `config` is the `types.Configuration` from `workflows.infer.infer`. The document is
    today's graph emitter output for the group (nodes, attempts, costs, edges,
    artifacts), raised to v0.3 with the task, the configuration (`source: habit`), the
    provenance and each node's piece. The configuration is written in OCP form by
    lane 1's emit helpers (D2). Costs are the emitter's as they are: lane 2's
    shared pricing (`price.price_all` in `load_history`) is the one pricing path (D62),
    so a priced aggregate keeps its tariff evidence and an unpriced one stays unpriced.
    `records` (run id to record) adds each mixed-model session's split to its cost (D67).
    """
    from .. import __version__
    from ..graph import to_ocp
    from ..ocp.emit import settings_to_ocp, workflow_to_ocp

    capabilities = {"task": True, "configuration": True, "signals": False, "slate": False, "receipt": False}
    doc = to_ocp(group.subgraph(), producer={"name": "loopmath", "version": __version__, "capabilities": capabilities})
    doc["ocp"] = "0.3"
    run = doc.setdefault("run", {})
    run["id"] = run_id_for(group)
    title = label.get("title") or f"{label['type'].replace('_', ' ')} in {group.repo}"
    run["title"] = title
    run.pop("workspace", None)
    task: dict = {
        "id": task_id_for(group),
        "title": title,
        "type": label["type"],
        "repo": group.repo,
        "source": {"kind": "history", "ref": group.id},
        "labeled_by": dict(labeled_by),
    }
    if label.get("subtype"):
        task["subtype"] = label["subtype"]
    if org:
        task["org"] = org
    if label.get("features"):
        task["features"] = dict(label["features"])
    run["task"] = task

    node_vertex = dict((getattr(config, "extra", None) or {}).get("node_vertex") or {})
    pieces = list(config.workflow.pieces)
    for n in group.nodes:
        if not node_vertex.get(n.id):
            node_vertex[n.id] = fallback_vertex(n, pieces)
    run["configuration"] = {"id": config.id, "workflow": workflow_to_ocp(config.workflow),
                            "settings": settings_to_ocp(config.settings), "source": "habit"}
    run["provenance"] = {"kind": "logged", "chooser": "habit"}

    for node in doc.get("nodes", []):
        vertex = node_vertex.get(node.get("id"))
        if vertex:
            node["vertex"] = vertex
    for attempt in doc.get("attempts", []):
        vertex = node_vertex.get(attempt.get("node"))
        if vertex:
            attempt["vertex"] = vertex
        attempt["round"] = 1
        split = model_tokens((records or {}).get(attempt.get("node")))
        if split is not None and isinstance(attempt.get("cost"), dict):
            attempt["cost"].setdefault("ext", {})[MODEL_TOKENS_KEY] = split

    ext = doc.setdefault("ext", {})
    ext["dev.loopmath.onboard"] = {
        "group": group.id,
        "sessions": len(group.nodes),
        "infer_confidence": round(float(confidence), 3),
    }
    return doc
