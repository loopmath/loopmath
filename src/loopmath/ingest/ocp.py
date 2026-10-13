"""Read OCP v0.2 documents into loopmath's session-level graph.

The native graph has one node per session attempt, while OCP separates a
logical node from its attempts.  A one-attempt OCP node keeps its node id (the
shape emitted by :func:`loopmath.graph.to_ocp`).  When an OCP node has retries,
each attempt becomes a graph node keyed by its globally unique attempt id.
This keeps every attempt and its cost instead of silently selecting one.

The reader deliberately does not re-grade or re-price OCP attempts.  The
``records_from_ocp`` view maps their terminal outcomes and measured costs to
the run-record rows consumed by ``loopmath analyze``.
"""

from __future__ import annotations

# Keep the original module namespace available to importers while the
# implementation lives in cohesive reader modules.
from .ocp_analysis import coverage_from_records, merge_graphs
from .ocp_common import (
    OCPError,
    Path,
    _COST_TO_GRAPH,
    _LAG_RE,
    _OUTCOME_TIERS,
    _REROUTED_RE,
    _TERMINAL_FALSE,
    _TERMINAL_TRUE,
    _attempt_graph_ids,
    _duration,
    _ext,
    _lag,
    _objects,
    _tokens,
    _unique,
    datetime,
    defaultdict,
    json,
    re,
    read_document,
)
from .ocp_convert import (
    Artifact,
    Counter,
    Graph,
    GraphEdge,
    GraphNode,
    _convert,
    _graph_node,
    copy,
    file_kind,
    from_ocp,
    load_ocp,
    records_from_ocp,
    summarize,
)


def _install_rebinding_bridge() -> None:
    from types import ModuleType

    from . import ocp_analysis, ocp_common, ocp_convert

    routes = (
        (
            ocp_common,
            frozenset(
                {
                    "OCPError",
                    "Path",
                    "_COST_TO_GRAPH",
                    "_LAG_RE",
                    "_unique",
                    "datetime",
                    "defaultdict",
                    "json",
                }
            ),
        ),
        (
            ocp_convert,
            frozenset(
                {
                    "Artifact",
                    "Counter",
                    "Graph",
                    "GraphEdge",
                    "GraphNode",
                    "OCPError",
                    "Path",
                    "_OUTCOME_TIERS",
                    "_REROUTED_RE",
                    "_TERMINAL_FALSE",
                    "_TERMINAL_TRUE",
                    "_attempt_graph_ids",
                    "_convert",
                    "_duration",
                    "_ext",
                    "_graph_node",
                    "_lag",
                    "_objects",
                    "_tokens",
                    "_unique",
                    "copy",
                    "defaultdict",
                    "file_kind",
                    "from_ocp",
                    "read_document",
                    "summarize",
                }
            ),
        ),
        (
            ocp_analysis,
            frozenset(
                {
                    "Artifact",
                    "Graph",
                    "GraphEdge",
                    "GraphNode",
                    "OCPError",
                    "copy",
                    "summarize",
                }
            ),
        ),
    )
    targets = {
        name: tuple(module for module, names in routes if name in names)
        for name in set().union(*(names for _, names in routes))
    }

    class _OCPModule(ModuleType):
        def __setattr__(self, name: str, value: object) -> None:
            super().__setattr__(name, value)
            for module in targets.get(name, ()):
                setattr(module, name, value)

    __import__("sys").modules[__name__].__class__ = _OCPModule


_install_rebinding_bridge()
del _install_rebinding_bridge


__all__ = [
    "OCPError",
    "coverage_from_records",
    "from_ocp",
    "load_ocp",
    "merge_graphs",
    "read_document",
    "records_from_ocp",
]
