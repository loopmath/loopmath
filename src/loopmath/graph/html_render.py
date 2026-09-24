"""Render a graph as one self-contained interactive HTML document."""

from __future__ import annotations

import html
import json

from .html_common import COMMON_JS
from .html_costcurve import COSTCURVE_JS
from .html_css import CSS
from .html_data import build_html_data
from .html_force import FORCE_JS
from .html_swimlanes import SWIMLANES_JS
from .schema import Graph


def _script_data(data: dict) -> str:
    encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    escaped = (
        encoded.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return json.dumps(escaped, ensure_ascii=False, allow_nan=False)


def to_html(graph: Graph) -> str:
    """Return one offline HTML page containing the graph and all view assets."""
    data = build_html_data(graph)
    workspace = html.escape(str(data["run"]["workspace"]), quote=True)
    payload = _script_data(data)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>loopmath graph: {workspace}</title>
<style>{CSS}</style>
</head>
<body>
<div id="app" class="app">
  <header class="top">
    <div class="brand"><span class="k">loopmath</span> graph <span class="sep">/</span> <span class="ttl">{workspace}</span></div>
    <div id="stats" class="stats"></div>
  </header>
  <div id="filters" class="filters"></div>
  <div id="accounting" class="accounting"></div>
  <section class="tablewrap" data-attempt-table>
    <div class="tablehead"><span id="tablecount"></span><span class="hint">click a row to expand the attempt; hover to highlight it in every graph</span></div>
    <div class="tablescroll"><table id="table"></table></div>
  </section>
  <details class="viewsec" data-view="swim" open>
    <summary>Swimlanes</summary>
    <div class="viewbody">
      <p class="idea" data-idea></p>
      <div class="viewcontrols" data-controls></div>
      <div class="graphhead"><span class="graphtitle" data-graph-title></span><div class="legend" data-legend></div></div>
      <div class="graphwrap" data-wrap><div class="graph" data-graph></div></div>
    </div>
  </details>
  <details class="viewsec" data-view="force" open>
    <summary>Force</summary>
    <div class="viewbody">
      <p class="idea" data-idea></p>
      <div class="viewcontrols" data-controls></div>
      <div class="graphhead"><span class="graphtitle" data-graph-title></span><div class="legend" data-legend></div></div>
      <div class="graphwrap" data-wrap><div class="graph" data-graph></div></div>
    </div>
  </details>
  <details class="viewsec" data-view="cost" open>
    <summary>Cost curve</summary>
    <div class="viewbody">
      <p class="idea" data-idea></p>
      <div class="viewcontrols" data-controls></div>
      <div class="graphhead"><span class="graphtitle" data-graph-title></span><div class="legend" data-legend></div></div>
      <div class="graphwrap" data-wrap><div class="graph" data-graph></div></div>
    </div>
  </details>
  <footer class="foot" id="foot"></footer>
  <div id="tip" class="tip" hidden></div>
  <div id="card" class="card" hidden></div>
</div>
<script>const DATA = JSON.parse({payload});</script>
<script>{COMMON_JS}</script>
<script>{SWIMLANES_JS}</script>
<script>{FORCE_JS}</script>
<script>{COSTCURVE_JS}</script>
<script>App.init([Swimlanes, Force, CostCurve]);</script>
</body>
</html>
"""
