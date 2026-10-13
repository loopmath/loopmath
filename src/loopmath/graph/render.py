"""Graphviz DOT rendering of a `Graph`. Text only; run `dot -Tsvg` yourself."""

from __future__ import annotations

from .schema import Graph

_FILL = {"top": "#dbeafe", "subagent": "#dcfce7", "codex": "#fef3c7"}
_EDGE_STYLE = {
    "dep": "solid",
    "fan_in": "bold",
    "spawn": "solid",
    "launch": "dashed",
    "artifact": "dotted",
}


def _q(s: object) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def to_dot(graph: Graph, *, artifacts: str = "consumed") -> str:
    """`artifacts`: "consumed" draws only artifacts with at least one reader
    other than the writer, "all" draws every written path, "none" skips them."""
    out = [
        "digraph loopmath {",
        "  rankdir=LR;",
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8];',
    ]
    for n in graph.nodes:
        label = [n.role or n.source]
        if n.model:
            label.append(n.model)
        if n.usd is not None:
            label.append(f"${n.usd:.2f}")
        attrs = [f'label="{_q(chr(10).join(label))}"', f'fillcolor="{_FILL.get(n.source, "#eeeeee")}"']
        if n.phase == "post":
            attrs.append('style="filled,rounded,dashed"')
        if n.role_tier == "heuristic":
            attrs.append('color="#9ca3af"')
        out.append(f'  "{_q(n.id)}" [{", ".join(attrs)}];')
    drawn_art: set[str] = set()
    # Colour of an artifact connector follows the tier of the artifact edge it stands for.
    edge_tier = {(e.detail.get("path"), e.dst): e.tier for e in graph.edges if e.kind == "artifact"}
    best = {}
    for (p, _), t in edge_tier.items():
        best[p] = "verified" if best.get(p) == "verified" or t == "verified" else t
    tier_color = lambda t: "#111827" if t == "verified" else "#9ca3af"
    if artifacts != "none":
        for a in graph.artifacts:
            if artifacts == "consumed" and not a.consumers:
                continue
            drawn_art.add(a.id)
            name = a.id.rsplit("/", 1)[-1]
            out.append(f'  "{_q(a.id)}" [shape=note, fillcolor="#f3f4f6", label="{_q(name)}"];')
            out.append(f'  "{_q(a.producer)}" -> "{_q(a.id)}" [style=dotted, color="{tier_color(best.get(a.id))}"];')
            for c in a.consumers:
                out.append(f'  "{_q(a.id)}" -> "{_q(c)}" [style=dotted, color="{tier_color(edge_tier.get((a.id, c)))}"];')
    for e in graph.edges:
        if e.kind == "artifact":
            continue
        color = "#111827" if e.tier == "verified" else "#9ca3af"
        label = ""
        if e.kind == "launch" and "lag_s" in e.detail:
            label = f', label="+{e.detail["lag_s"]}s"'
        out.append(f'  "{_q(e.src)}" -> "{_q(e.dst)}" [style={_EDGE_STYLE.get(e.kind, "solid")}, color="{color}"{label}];')
    out.append("}")
    return "\n".join(out) + "\n"
