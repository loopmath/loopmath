"""The workflow graph: what the extractor emits and what everything downstream reads.

Three object kinds, deliberately few:

- `GraphNode`: one attempt, i.e. one parsed session (a `RunRecord`, by
  `run_id`). Nodes are never invented; every node is a session file on disk.
- `GraphEdge`: a directed relation between two nodes. Five kinds. The
  extractor discovers three of them: `spawn` (a Claude Code session started a
  subagent through the Task tool), `launch` (a session ran another harness's
  CLI, e.g. `codex exec`, and a session of that harness started right after in
  the same workspace), and `artifact` (a node wrote a file that another node
  later read). The other two are the OCP core scheduling kinds, read from a
  document and never inferred from a session: `dep` (`dst` cannot start before
  `src` settles) and `fan_in` (`src` is a member of gate `dst`'s acceptance
  input set).
- `Artifact`: a file path with its writers and readers. Artifacts are the
  handoff objects the valuation work needs to price; the graph records who
  produced them and who consumed them, never what they are worth.

Every edge and every role label carries an evidence tier reused from
`grade.py` so consumers know what to trust: `verified` means both ends of the
relation are structurally present in the logs (a Task tool_use id matched a
subagent's meta file; a write and a later read of the same path), `heuristic`
means an inference from timing or text (a codex session that began 3 s after
a `codex exec` command in the same workspace; a role guessed from a spawn
description), `reported` means declared in the log by the caller or the
harness and taken at face value (a Task call with `subagent_type: "Plan"`, a
codex session's model field). Anything the extractor cannot establish is left
None and counted in `Graph.meta`, never filled in.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict

# The one edge vocabulary. Order is the OCP recommended order, and it is the
# order the viewer lays edges out in, so consumers that need a kind list or a
# per-kind style map derive it from here instead of keeping a second copy.
EDGE_KINDS = ("dep", "fan_in", "spawn", "launch", "artifact")
TIERS = ("verified", "heuristic", "reported")
SOURCES = ("top", "subagent", "codex", "external")
PHASES = ("build", "external", "post")


@dataclass
class GraphNode:
    id: str
    harness: str
    source: str
    session_path: str
    model: str | None = None
    model_tier: str | None = None
    effort: str | None = None
    workspace: str | None = None
    ts: str | None = None
    wall_s: float | None = None
    tokens: dict | None = None
    usd: float | None = None
    parent: str | None = None
    spawn: dict | None = None
    launched_by: dict | None = None
    role: str | None = None
    role_tier: str | None = None
    role_evidence: str | None = None
    phase: str | None = None
    phase_tier: str | None = None


@dataclass
class GraphEdge:
    src: str
    dst: str
    kind: str
    tier: str
    detail: dict = field(default_factory=dict)


@dataclass
class Artifact:
    id: str
    producer: str
    writers: list[str] = field(default_factory=list)
    consumers: list[str] = field(default_factory=list)
    first_write_ts: str | None = None
    n_writes: int = 0
    n_reads: int = 0
    kind: str | None = None
    kind_tier: str | None = None
    # A3: the heredoc first line that refined the kind (None when no heredoc wrote the
    # path), and the full write list: every write event of the path with its node, tier
    # and how (Write, heredoc, `codex -o`, git commit, ...), in time order.
    hint: str | None = None
    writes: list[dict] = field(default_factory=list)
    # Q7 artifact recording.  These stay after the original fields so callers that
    # construct Artifact positionally retain the pre-Q7 argument order.
    bytes: int | None = None
    bytes_tier: str | None = None
    lines_added: int | None = None
    lines_added_tier: str | None = None
    lines_removed: int | None = None
    lines_removed_tier: str | None = None
    language: str | None = None
    language_tier: str | None = None
    tests_touched: int | None = None
    tests_touched_tier: str | None = None
    fate: str = "unknown"
    fate_tier: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Graph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    # Producer-specific node provenance which the native OCP emitter needs in
    # order to reproduce its own document exactly after an OCP read.  This is
    # deliberately not part of ``to_dict``: the public graph JSON contract is
    # unchanged, and generic OCP extensions are not presented as graph data.
    _ocp_node_ext: dict[str, dict] = field(default_factory=dict, repr=False)
    # Private, non-contract OCP emission state, excluded from ``to_dict`` and
    # equality.  Every modeled field outside ext.dev.loopmath.graph.meta, plus
    # ``emitter`` and ``edges_not_emitted``, is regenerated from graph content;
    # the pre-existing _ocp_node_ext is the separate preservation path for the
    # emitter's node-extension provenance.  Within meta, _ocp_source_meta keeps
    # nine named ocp_* absence markers for current-writer documents, plus the
    # source nodes_missing_tokens value only for historical documents where it
    # disagrees with the shared reader/writer rule.  The sets are closed: an
    # unrelated difference cannot become private round-trip state.
    # _ocp_loaded_meta is the full guard snapshot but is never emitted.  If
    # current meta differs from that guard, no delta is applied: deleted keys
    # stay dropped, explicit None stays null, changed values stay changed, and
    # no metadata member is recomputed.
    _ocp_source_meta: dict | None = field(default=None, repr=False, compare=False)
    _ocp_loaded_meta: dict | None = field(default=None, repr=False, compare=False)

    def node(self, node_id: str) -> GraphNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def to_dict(self) -> dict:
        return {
            "dagr_graph": 1,
            "meta": copy.deepcopy(self.meta),
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "artifacts": [asdict(a) for a in self.artifacts],
        }
