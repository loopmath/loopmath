"""Inline force layout copied from the accepted prototype."""

FORCE_JS = r"""
const Force = {
  key: 'force',
  idea: 'Two dimensions: clock time on the horizontal axis, pinned, and a free vertical axis settled by forces. Every attempt is a circle fixed at its start time; springs pull it toward the attempts it exchanged files with (spawn, launch and the declared scheduling edges dep and fan_in are drawn but do not pull, since every attempt hangs off the lead), and repulsion pushes neighbors in time apart, so clusters of attempts that worked on the same files gather vertically while the time order stays exact. The faint bar behind a circle is its duration. Consumed files join the simulation as visible square nodes pinned at their first write.',
  graphTitle: 'Force layout, horizontal position pinned to start time (UTC)',
  sizeNote: 'circle area = dollars, faint bar = duration, dashed outline (on circles large enough) = model tier reported rather than verified; the lead is pinned to the middle',
  colorBy: 'role',
  artifactMode: 'always',
  controls: [{ id: 'handoffs', label: 'handoff edges', value: 'all', options: [{ v: 'all', l: 'all (they shape the layout)' }, { v: 'verified', l: 'verified only' }, { v: 'none', l: 'none (structure edges only)' }] }],
  render(host, vis, state, api) {
    const { N, E, A, RUN, fmt, svg, text, color, radius, nodeClass, edgeClass, edgeVisible, artVisible, curve, timeTicks } = api;
    const W = Math.max(host.clientWidth || 1000, 900), H = 640, ML = 44, MR = 60, MT = 30, MB = 30;
    const span = RUN.plot_span_s, x = at => ML + at / span * (W - ML - MR), mode = state.extra.handoffs;
    const useEdge = edge => edgeVisible(edge) && (edge[2] !== 'artifact' || mode === 'all' || (mode === 'verified' && edge[3] === 'verified'));
    const band = { lead: 0.5, planner: 0.3, dev: 0.35, reviewer: 0.7, cli: 0.2, external: 0.9, unlabeled: 0.85 };
    const jitter = i => (((i * 9301 + 49297) % 233280) / 233280 - 0.5);
    const items = vis.map(i => { const n = N[i]; return { t: 'n', i, x: x(n.t0Plot), y: MT + (H - MT - MB) * ((band[n.role] || 0.5) + jitter(i) * 0.3), r: radius(n, 20, 4), vy: 0 }; });
    const byNode = new Map(items.map((item, k) => [item.i, k]));
    const artifacts = mode !== 'none' ? A.filter(artVisible).filter(a => a.t != null) : [], byArtifact = new Map();
    artifacts.forEach(a => { byArtifact.set(a.i, items.length); items.push({ t: 'a', i: a.i, x: x(a.t), y: MT + (H - MT - MB) * (0.5 + jitter(a.i + 7) * 0.6), r: 5, vy: 0 }); });
    const writerEdges = new Set(artifacts.flatMap(a => a.we.flat())), consumerEdges = new Set(artifacts.flatMap(a => a.ce.flat())), representedArtifactEdges = new Set([...writerEdges].filter(k => consumerEdges.has(k))), links = [];
    E.forEach((edge, k) => { if (!useEdge(edge)) return; if (edge[2] === 'artifact' && byArtifact.has(edge[4]) && representedArtifactEdges.has(k)) return; links.push({ a: byNode.get(edge[0]), b: byNode.get(edge[1]), w: edge[2] === 'artifact' ? (edge[3] === 'verified' ? 1 : 0.35) : 0, cls: edgeClass(edge), k }); });
    artifacts.forEach(a => { const pos = byArtifact.get(a.i); a.w.forEach((i, k) => { if (byNode.has(i)) links.push({ a: byNode.get(i), b: pos, w: 0.5, cls: 'alink', art: a.i, node: i, kind: 'write', edges: a.we[k] }); }); a.c.forEach((i, k) => { if (byNode.has(i)) links.push({ a: pos, b: byNode.get(i), w: 0.5, cls: 'alink', art: a.i, node: i, kind: 'read', edges: a.ce[k] }); }); });
    const cy = MT + (H - MT - MB) / 2, degree = items.map(() => 0);
    links.forEach(link => { if (link.w) { degree[link.a] += link.w; degree[link.b] += link.w; } });
    const ITER = 500, WIN = 110;
    for (let round = 0; round < ITER; round++) {
      const alpha = 1 - round / ITER;
      links.forEach(link => { if (!link.w) return; const a = items[link.a], b = items[link.b], dy = b.y - a.y; a.vy += dy * 0.03 * link.w / Math.sqrt(Math.max(1, degree[link.a])); b.vy -= dy * 0.03 * link.w / Math.sqrt(Math.max(1, degree[link.b])); });
      items.forEach(point => { point.vy += (cy - point.y) * 0.0004; });
      for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
        const a = items[i], b = items[j], dx = Math.abs(b.x - a.x); if (dx >= WIN) continue;
        const dy = b.y - a.y, distance = Math.abs(dy), sign = dy > 0 ? 1 : dy < 0 ? -1 : (i < j ? -1 : 1), minimum = a.r + b.r + 6, near = 1 - dx / WIN;
        let push = (220 * near * (0.3 + alpha)) / Math.max(distance, 6);
        if (dx < minimum && distance < minimum) push += (minimum - distance) * 0.35;
        a.vy -= sign * push; b.vy += sign * push;
      }
      items.forEach(point => { if (point.t === 'n' && N[point.i].role === 'lead') { point.y = cy; point.vy = 0; return; } point.y += point.vy; point.vy *= 0.45; point.y = Math.max(MT + point.r, Math.min(H - MB - point.r, point.y)); });
    }
    const root = svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, host);
    timeTicks(span).forEach(at => { const tx = x(at); svg('line', { x1: tx, y1: MT - 4, x2: tx, y2: H - MB, class: 'grid' }, root); text(root, tx + 3, MT - 9, fmt.clock(at) + '  +' + fmt.hm(at), 'axlbl'); });
    svg('line', { x1: ML, y1: MT, x2: W - MR, y2: MT, class: 'axis' }, root);
    const durationGroup = svg('g', {}, root), edgeGroup = svg('g', {}, root), nodeGroup = svg('g', {}, root);
    items.forEach(point => { if (point.t !== 'n') return; const n = N[point.i]; if (n.untimed || n.dur == null || n.dur < 1) return; svg('line', { x1: point.x, y1: point.y, x2: Math.max(x(n.t1), point.x + 1), y2: point.y, stroke: color(n, state.colorBy), 'stroke-width': 4, opacity: 0.22, 'stroke-linecap': 'round' }, durationGroup); });
    links.forEach(link => { const a = items[link.a], b = items[link.b], attrs = { d: curve(a.x, a.y, b.x, b.y, 0.4), class: link.cls }; if (link.k != null) attrs['data-edge'] = link.k; else { attrs['data-alink'] = link.art; attrs['data-edge-list'] = link.edges.join(','); attrs['data-connection-node'] = link.node; attrs['data-connection-kind'] = link.kind; } svg('path', attrs, edgeGroup); });
    items.forEach(point => {
      if (point.t === 'a') { svg('rect', { x: point.x - 5, y: point.y - 5, width: 10, height: 10, class: 'art', 'data-art': point.i }, nodeGroup); return; }
      const n = N[point.i];
      svg('circle', { cx: point.x, cy: point.y, r: point.r, fill: color(n, state.colorBy), class: point.r >= 7 ? nodeClass(n) : nodeClass(n).replace(' dashed', ''), 'data-node': point.i }, nodeGroup);
      if (point.r < 9) svg('circle', { cx: point.x, cy: point.y, r: 9, class: 'hit', 'data-node': point.i }, nodeGroup);
      if (point.r >= 7) text(nodeGroup, point.x + point.r + 3, point.y + 3.5, n.lbl, 'nlbl');
    });
  },
};
"""
