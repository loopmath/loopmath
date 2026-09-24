"""The workflow graph dict and the `workflows show --html` page.

`workflow_graph` returns the view fixtures' format (spec 03 section 8):
`{config, label, nodes, edges, gates, workflow, title, budget_rounds}` with
piece nodes `{id, kind: "piece", role, setting, prediction?, width?}` and
artifact nodes `{id, kind: "artifact"}`. Both it and the page are lane 12's
`views.common.workflow_graph` and `graph_page`.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..types import Configuration, Setting, Workflow
from ..views import common as _views

RENDERER = "views.common.graph_page"


def workflow_graph(workflow: Workflow, settings: Mapping[str, Setting] | None = None,
                   config: Configuration | None = None, prediction: Any = None) -> dict[str, Any]:
    """The graph dict for one workflow (with settings and per-piece predictions when given)."""
    cfg = config.to_dict() if config is not None else {
        "id": None, "workflow": workflow.to_dict(),
        "settings": {k: v.to_dict() for k, v in (settings or {}).items()}}
    return _views.workflow_graph(cfg, prediction)


def workflow_page(workflow: Workflow, settings: Mapping[str, Setting] | None = None,
                  config: Configuration | None = None, data: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """(HTML, renderer): lane 12's workflow page."""
    graph = workflow_graph(workflow, settings, config)
    obj = {"schema": "loopmath.view.workflow/1", "graph": graph, **dict(data or {})}
    return _views.graph_page(graph, title=f"loopmath workflow: {workflow.id}", data=obj), RENDERER
