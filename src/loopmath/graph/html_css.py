"""Inline stylesheet for the graph visualizer."""

CSS = r"""
:root {
  color-scheme: light;
  --surface: #fcfcfb; --plane: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --focus: #0b0b0b;
  --c-lead: #2a78d6; --c-dev: #eb6834; --c-rev: #1baf7a; --c-plan: #4a3aa7; --c-other: #898781;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--plane); color: var(--ink); font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
@media (min-width: 1000px) { html, body { font-size: 15px; } }
.app { max-width: 1500px; margin: 0 auto; padding: 12px 16px 40px; }
.top { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 16px; }
.brand { font-size: 18px; font-weight: 600; }
.brand .k { color: var(--c-lead); }
.brand .sep { color: var(--muted); font-weight: 400; margin: 0 4px; }
.brand .ttl { font-weight: 500; }
.stats { display: flex; flex-wrap: wrap; gap: 6px 18px; }
.stat { display: flex; flex-direction: column; min-width: 70px; }
.stat b { font-size: 17px; font-weight: 600; }
.stat span { font-size: 12px; color: var(--muted); }
.filters, .viewcontrols { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; padding: 8px 10px; background: var(--surface); }
.filters { border: 1px solid var(--border); border-radius: 8px; margin: 10px 0; }
.accounting { margin: -2px 0 10px; padding: 7px 10px; border: 1px solid #d7c693; border-radius: 7px; background: #fff9e8; color: var(--ink2); font-size: 12px; }
.viewcontrols { border-bottom: 1px solid var(--border); }
.filters .grp, .viewcontrols .grp { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
.filters .grp > .gl, .viewcontrols .grp > .gl { font-size: 12px; color: var(--muted); margin-right: 2px; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border: 1px solid var(--border); border-radius: 999px; background: #fff; cursor: pointer; font-size: 13px; user-select: none; }
.chip.off { opacity: 0.4; }
.chip .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.chip.sq .dot { border-radius: 2px; }
select, button, input[type=range] { font: inherit; font-size: 13px; }
select { padding: 4px 6px; border: 1px solid var(--border); border-radius: 6px; background: #fff; }
button { padding: 4px 10px; border: 1px solid var(--border); border-radius: 6px; background: #fff; cursor: pointer; }
label.cb { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; }
label.cb input { width: 16px; height: 16px; }
.tablewrap { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 12px; }
.tablehead { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px 12px; padding: 6px 12px; font-size: 12px; color: var(--muted); border-bottom: 1px solid var(--border); }
.tablescroll { max-height: 38vh; overflow: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 5px 10px; text-align: left; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { position: sticky; top: 0; background: var(--surface); font-size: 12px; font-weight: 600; color: var(--ink2); cursor: pointer; z-index: 1; }
th.num, td.num { text-align: right; }
th .arrow { color: var(--muted); font-size: 10px; margin-left: 3px; }
tr[data-i] { cursor: pointer; }
tr[data-i]:hover { background: #f1f0ec; }
tr.rel td { background: #eef4fb; }
tr.focus td { background: #dbe9fb; }
tr.dim td { opacity: 0.45; }
td .lbl { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; }
td .lbl .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }
td .sub { display: block; font-size: 11.5px; color: var(--muted); max-width: 320px; overflow: hidden; text-overflow: ellipsis; font-weight: 400; }
td .tier { color: var(--muted); font-size: 11px; margin-left: 4px; }
.viewsec { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 12px; }
.viewsec > summary { cursor: pointer; display: flex; justify-content: space-between; gap: 12px; padding: 9px 12px; font-weight: 600; list-style-position: inside; }
.viewsec > summary::after { content: "fold"; color: var(--muted); font-size: 12px; font-weight: 400; }
.viewsec:not([open]) > summary::after { content: "unfold"; }
.viewbody { border-top: 1px solid var(--border); }
.idea { margin: 0; padding: 8px 12px; color: var(--ink2); max-width: 1100px; }
.graphhead { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px 16px; align-items: center; padding: 8px 12px; border-bottom: 1px solid var(--border); border-top: 1px solid var(--border); font-size: 13px; }
.graphtitle { font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--ink2); }
.legend .it { display: inline-flex; align-items: center; gap: 5px; }
.legend .sw { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
.legend .sw.sq { border-radius: 2px; }
.legend svg { vertical-align: middle; }
.graphwrap { position: relative; overflow: auto; }
.graph { min-height: 200px; }
.graph svg { display: block; }
.card { position: absolute; z-index: 5; width: 360px; max-width: calc(100% - 16px); max-height: 70vh; overflow: auto; background: #fff; border: 1px solid var(--border); box-shadow: 0 6px 24px rgba(0,0,0,0.14); border-radius: 10px; padding: 10px 12px 12px; font-size: 13px; }
.card h3 { margin: 0 24px 2px 0; font-size: 15px; display: flex; align-items: center; gap: 8px; }
.card h3 .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
.card .ttl { color: var(--ink2); margin-bottom: 8px; }
.card .x { position: absolute; top: 6px; right: 8px; border: 0; background: transparent; font-size: 18px; line-height: 1; padding: 4px 6px; color: var(--muted); }
.card dl { display: grid; grid-template-columns: 96px 1fr; gap: 3px 8px; margin: 0; }
.card dt { color: var(--muted); font-size: 12px; }
.card dd { margin: 0; overflow-wrap: anywhere; }
.card dd.mono, .card .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }
.card ul { margin: 2px 0 0; padding-left: 16px; }
.card li { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; overflow-wrap: anywhere; }
.card li.more, .card .none { color: var(--muted); font-family: inherit; font-size: 12px; }
.card .link { color: var(--c-lead); cursor: pointer; text-decoration: underline dotted; }
.card .evid { color: var(--muted); font-size: 12px; }
.tip { position: fixed; z-index: 9; pointer-events: none; background: #1f1f1e; color: #fff; padding: 6px 9px; border-radius: 6px; font-size: 12.5px; line-height: 1.35; max-width: 340px; box-shadow: 0 4px 14px rgba(0,0,0,0.25); }
.tip b { font-weight: 600; }
.tip .m { color: #c3c2b7; }
.foot { margin-top: 14px; font-size: 12px; color: var(--muted); line-height: 1.5; }
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.lanelbl { font-size: 12px; fill: var(--ink2); font-weight: 600; }
.axlbl { font-size: 11px; fill: var(--muted); }
.axis { stroke: var(--axis); stroke-width: 1; }
.grid { stroke: var(--grid); stroke-width: 1; }
.lanebg { fill: #f4f3ef; }
.lanebg.alt { fill: #fbfaf7; }
.node { stroke: #fff; stroke-width: 1.5; cursor: pointer; transition: opacity .12s; }
.node.dashed { stroke-dasharray: 3 2; stroke: #fff; }
.node.hollow { fill: #fff; }
.node.dim { opacity: 0.16; }
.node.rel { stroke: var(--focus); stroke-width: 1.5; }
.node.focus { stroke: var(--focus); stroke-width: 2.5; }
.nlbl { font-size: 10.5px; fill: var(--ink); pointer-events: none; }
.nlbl.dim { opacity: 0.2; }
.hit { fill: transparent; stroke: none; cursor: pointer; }
.edge { fill: none; pointer-events: stroke; transition: opacity .12s; }
.e-dep { stroke: #7a6a9c; stroke-width: 1.2; stroke-dasharray: 1 3; opacity: 0.8; }
.e-fan_in { stroke: #7a6a9c; stroke-width: 1.2; stroke-dasharray: 7 2 1 2; opacity: 0.8; }
.e-spawn { stroke: #52514e; stroke-width: 1.2; opacity: 0.7; }
.e-launch { stroke: #52514e; stroke-width: 1.2; stroke-dasharray: 4 3; opacity: 0.7; }
.e-artifact { stroke: #b3b1a9; stroke-width: 1; opacity: 0.45; }
.e-artifact.t-verified { stroke: #52514e; opacity: 0.8; }
.edge.dim { opacity: 0.05; }
.edge.on { stroke: var(--focus); opacity: 1; stroke-width: 1.8; }
.art { fill: #f0efec; stroke: #898781; stroke-width: 1; cursor: pointer; }
.art.dim { opacity: 0.15; }
.art.rel, .art.on { stroke: var(--focus); stroke-width: 1.8; fill: #fff; }
.art.focus { stroke: var(--focus); stroke-width: 2.5; fill: #fff; }
.small { font-size: 10px; fill: var(--muted); }
.alink { fill: none; stroke: #b3b1a9; stroke-width: 1; opacity: 0.55; pointer-events: none; transition: opacity .12s; }
.alink[data-edge-list] { pointer-events: stroke; cursor: help; }
.alink.dim { opacity: 0.05; }
.alink.on { stroke: var(--focus); stroke-width: 1.6; opacity: 1; }
.hide-heur .e-artifact.t-heuristic:not(.on) { opacity: 0; pointer-events: none; }
.seg { fill: none; stroke-width: 2.2; stroke-linecap: round; cursor: pointer; transition: opacity .12s; }
.seg.dashed { stroke-dasharray: 5 3; }
.seg.unavailable { stroke-dasharray: 2 2; }
.seg.dim { opacity: 0.12; }
.seg.rel { stroke-width: 3.4; }
.seg.focus { stroke-width: 5; }
.env { fill: none; stroke: #898781; stroke-width: 1.2; }
.alink.reads:not(.on) { opacity: 0; }
"""
