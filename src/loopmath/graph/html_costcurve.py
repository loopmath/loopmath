"""Inline cost curve layout copied from the accepted prototype."""

COSTCURVE_JS = r"""
const CostCurve = {
  key: 'cost',
  idea: 'Two dimensions: clock time on the horizontal axis and cumulative dollars on the vertical axis. Attempts are stacked in start order: each one is a segment that begins at the dollars already committed by earlier starts and rises by its own cost over its own duration, so slope is burn rate and the outline of the pile is the shape of the run. The thin gray line is dollars actually spent by each moment, assuming even burn within an attempt. Spawn and launch edges are arcs from the parent segment to the child start; handoff arcs run from the write to the read.',
  graphTitle: 'Cost curve: cumulative dollars by time',
  sizeNote: 'segment rise = dollars, segment length = duration, dashed = model tier reported rather than verified',
  colorBy: 'role',
  artifactMode: 'toggle',
  controls: [
    { id: 'stack', label: 'stack', value: 'start', options: [{ v: 'start', l: 'by start time' }, { v: 'role', l: 'by role, then start' }, { v: 'cost', l: 'by cost, largest first' }] },
    { id: 'handoffs', label: 'handoff arcs', value: 'verified', options: [{ v: 'verified', l: 'verified, plus all of the focused attempt' }, { v: 'all', l: 'all (%HANDOFFS% arcs)' }, { v: 'none', l: 'none' }] },
  ],
  render(host, vis, state, api) {
    const { N, E, A, RUN, ROLES, fmt, svg, text, color, dashed, edgeClass, edgeVisible, artVisible, arc, timeTicks } = api;
    const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
    const W = Math.max(host.clientWidth || 1000, 900), ML = 70, MR = 70, MT = 26, MB = 40, H = 600, plotH = H - MT - MB;
    const span = RUN.plot_span_s, x = at => ML + at / span * (W - ML - MR), roleRank = role => ROLES.indexOf(role), cost = n => n.usd == null ? 0 : n.usd;
    const ids = vis.filter(i => !N[i].untimed).sort((a, b) => {
      const left = N[a], right = N[b];
      if (state.extra.stack === 'role') return roleRank(left.role) - roleRank(right.role) || left.t0Plot - right.t0Plot;
      if (state.extra.stack === 'cost') return cost(right) - cost(left) || left.t0Plot - right.t0Plot;
      return left.t0Plot - right.t0Plot || a - b;
    });
    let cumulative = 0; ids.forEach(i => { N[i]._y0 = cumulative; cumulative += cost(N[i]); });
    const total = Math.max(cumulative, 1), y = value => MT + (1 - value / total) * plotH;
    const root = svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, host);
    if (state.extra.handoffs !== 'all') root.classList.add('hide-heur');
    const step = total > 400 ? 100 : total > 150 ? 50 : total > 40 ? 10 : total > 8 ? 2 : 1;
    for (let value = 0; value <= total; value += step) { svg('line', { x1: ML, y1: y(value), x2: W - MR, y2: y(value), class: 'grid' }, root); text(root, ML - 8, y(value) + 4, '$' + value, 'axlbl', { 'text-anchor': 'end' }); }
    timeTicks(span).forEach(at => { const tx = x(at); svg('line', { x1: tx, y1: MT, x2: tx, y2: H - MB + 4, class: 'grid' }, root); text(root, tx, H - MB + 16, fmt.clock(at), 'axlbl', { 'text-anchor': 'middle' }); text(root, tx, H - MB + 28, '+' + fmt.hm(at), 'axlbl', { 'text-anchor': 'middle' }); });
    svg('line', { x1: ML, y1: H - MB, x2: W - MR, y2: H - MB, class: 'axis' }, root); svg('line', { x1: ML, y1: MT, x2: ML, y2: H - MB, class: 'axis' }, root);
    const spent = (n, at) => n.usd == null || n.dur == null ? 0 : n.dur > 0 ? n.usd * clamp((at - n.t0Plot) / n.dur, 0, 1) : (at >= n.t0Plot ? n.usd : 0);
    const points = [], steps = 240;
    for (let k = 0; k <= steps; k++) { const at = span * k / steps; let value = 0; ids.forEach(i => { value += spent(N[i], at); }); points.push(`${x(at)},${y(value)}`); }
    svg('path', { d: 'M' + points.join('L'), class: 'env' }, root);
    { let value = 0; const at = span * 0.62; ids.forEach(i => { value += spent(N[i], at); }); text(root, x(at) + 8, y(value) + 40, 'gray line: dollars spent by this moment', 'axlbl'); }
    const edgeGroup = svg('g', {}, root), artifactGroup = svg('g', {}, root), nodeGroup = svg('g', {}, root);
    const pointAt = (n, at) => { const fraction = n.dur != null && n.dur > 0 ? clamp((at - n.t0Plot) / n.dur, 0, 1) : 0; return [n._p[0] + (n._p[2] - n._p[0]) * fraction, n._p[1] + (n._p[3] - n._p[1]) * fraction]; };
    ids.forEach(i => { const n = N[i], x1 = x(n.t0Plot), x2 = Math.max(x(n.t1), x1 + 1); n._p = [x1, y(n._y0), x2, y(n._y0 + cost(n))]; });
    const untimed = vis.filter(i => N[i].untimed);
    untimed.forEach((i, k) => { const n = N[i]; n._p = [ML + 8 + k * 70, H - 8, ML + 8 + k * 70, H - 8]; svg('circle', { cx: n._p[0], cy: n._p[1], r: 5, class: 'node hollow', fill: '#fff', stroke: '#898781', 'data-node': i }, nodeGroup); text(nodeGroup, n._p[0] + 9, n._p[1] + 4, n.lbl + ' (untimed)', 'nlbl'); });
    const placed = new Set([...ids, ...untimed]);
    E.forEach((edge, k) => {
      if (!edgeVisible(edge) || !placed.has(edge[0]) || !placed.has(edge[1])) return; const source = N[edge[0]], target = N[edge[1]]; let p1, p2;
      if (edge[2] === 'artifact') { if (state.extra.handoffs === 'none') return; const artifact = A[edge[4]], written = artifact && artifact.t != null ? artifact.t : source.t0Plot, read = edge[5] != null ? written + edge[5] : target.t0Plot; p1 = pointAt(source, written); p2 = pointAt(target, read); }
      else { p1 = pointAt(source, target.t0Plot); p2 = pointAt(target, target.t0Plot); }
      svg('path', { d: arc(p1[0], p1[1], p2[0], p2[1], edge[2] === 'artifact' ? 0.15 : 0.05), class: edgeClass(edge), 'data-edge': k }, edgeGroup);
    });
    if (state.artifacts) {
      const artifactEdges = new Map(); E.forEach((edge, k) => { if (edge[2] === 'artifact') { if (!artifactEdges.has(edge[4])) artifactEdges.set(edge[4], []); artifactEdges.get(edge[4]).push(k); } });
      artifactEdges.forEach((keys, j) => keys.sort((left, right) => A[j].w.indexOf(E[left][0]) - A[j].w.indexOf(E[right][0])));
      A.filter(artVisible).filter(a => a.t != null).forEach(a => {
        const writers = a.w.filter(i => placed.has(i) && !N[i].untimed); if (!writers.length) return;
        const writer = N[writers[0]], point = pointAt(writer, a.t);
        a.c.forEach(i => { if (!placed.has(i) || N[i].untimed) return; const edgeKey = (artifactEdges.get(a.i) || []).find(k => E[k][1] === i), read = edgeKey != null && E[edgeKey][5] != null ? a.t + E[edgeKey][5] : N[i].t0Plot, target = pointAt(N[i], read); svg('path', { d: arc(point[0], point[1], target[0], target[1], 0.12), class: 'alink', 'data-alink': a.i }, artifactGroup); });
        svg('rect', { x: point[0] - 4, y: point[1] - 4, width: 8, height: 8, class: 'art', 'data-art': a.i }, artifactGroup);
      });
    }
    ids.forEach(i => { const n = N[i], path = `M${n._p[0]},${n._p[1]}L${n._p[2]},${n._p[3]}`; svg('path', { d: path, class: 'seg' + (dashed(n) ? ' dashed' : '') + (n.usd == null || n.dur == null ? ' unavailable' : ''), stroke: color(n, state.colorBy), 'data-node': i }, nodeGroup); svg('path', { d: path, class: 'hit', stroke: 'transparent', 'stroke-width': 12, fill: 'none', 'data-node': i }, nodeGroup); if (cost(n) >= 5) text(nodeGroup, n._p[2] + 4, n._p[3] + 3, n.lbl, 'nlbl'); });
    text(root, ML + 6, MT + 12, `${ids.length} attempts stacked, ${fmt.usd(cumulative)}`, 'axlbl');
  },
};
"""
