"""Inline swimlanes layout copied from the accepted prototype."""

SWIMLANES_JS = r"""
const Swimlanes = {
  key: 'swim',
  idea: 'Two dimensions: role on the vertical axis, one swimlane per role packed into sub-rows so concurrent attempts never overlap, and clock time on the horizontal axis. Each attempt is a box: width is its duration, height is its dollars. Spawn and launch edges drop from the lead lane to each child at the moment it started. A declared scheduling edge, dep or fan_in, runs from the end of the source box to the start of the target box. Handoff arcs run from the write to the read. With artifacts on, an extra lane shows every consumed file at its first write, joined to its writers; the reads of a file appear when you hover or click it, or the attempt that read it.',
  graphTitle: 'Swimlanes: one lane per role, time left to right (UTC)',
  sizeNote: 'box width = duration, box height = dollars (square root, capped at $20), dashed outline = model tier reported rather than verified',
  colorBy: 'model',
  artifactMode: 'toggle',
  controls: [{ id: 'handoffs', label: 'handoff arcs', value: 'verified', options: [{ v: 'verified', l: 'verified, plus all of the focused attempt' }, { v: 'all', l: 'all (%HANDOFFS% arcs)' }, { v: 'none', l: 'none' }] }],
  render(host, vis, state, api) {
    const { N, E, A, RUN, ROLES, fmt, svg, text, color, nodeClass, edgeClass, edgeVisible, artVisible, vcurve, arc, timeTicks } = api;
    const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
    const namedLanes = ROLES.filter(role => role !== 'external' && role !== 'unlabeled');
    const laneOf = n => namedLanes.includes(n.role) ? n.role : 'outside';
    const laneOrder = [...namedLanes, 'outside'];
    const lanes = new Map(laneOrder.map(lane => [lane, []]));
    vis.forEach(i => lanes.get(laneOf(N[i])).push(i));
    const W = Math.max(host.clientWidth || 1000, 900), ML = 100, MR = 24, MT = 36, MB = 16, rowH = 30, laneGap = 6;
    const span = RUN.plot_span_s, x = at => ML + at / span * (W - ML - MR);
    let y = MT; const laneInfo = [];
    for (const [lane, ids] of lanes) {
      if (!ids.length) continue;
      ids.sort((a, b) => N[a].t0Plot - N[b].t0Plot || a - b);
      const rows = [];
      ids.forEach(i => {
        const n = N[i], x0 = x(n.t0Plot), x1 = Math.max(x(n.t1), x0 + 3);
        let row = rows.findIndex(end => end + 2 <= x0); if (row < 0) { row = rows.length; rows.push(0); }
        rows[row] = x1; n._row = row; n._x0 = x0; n._x1 = x1;
      });
      const height = rows.length * rowH; laneInfo.push({ lane, ids, y0: y, height });
      ids.forEach(i => { N[i]._y = y + N[i]._row * rowH + rowH / 2; });
      y += height + laneGap;
    }
    let artifacts = [];
    if (state.artifacts) {
      artifacts = A.filter(artVisible).filter(a => a.t != null).sort((a, b) => a.t - b.t);
      const rows = [];
      artifacts.forEach(a => { const ax = x(a.t); let row = rows.findIndex(end => end + 11 <= ax); if (row < 0) { row = rows.length; rows.push(0); } rows[row] = ax; a._x = ax; a._row = row; });
      const height = Math.max(1, rows.length) * 16 + 10; laneInfo.push({ lane: 'artifacts', ids: [], y0: y, height, artifacts: true });
      artifacts.forEach(a => { a._y = y + 12 + a._row * 16; }); y += height + laneGap;
    }
    const H = y + MB, root = svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, host);
    if (state.extra.handoffs !== 'all') root.classList.add('hide-heur');
    laneInfo.forEach((info, k) => {
      svg('rect', { x: ML, y: info.y0, width: W - ML - MR, height: info.height, class: 'lanebg' + (k % 2 ? ' alt' : '') }, root);
      const usd = info.ids.reduce((total, i) => total + (N[i].usd == null ? 0 : N[i].usd), 0);
      text(root, ML - 8, info.y0 + 14, info.lane, 'lanelbl', { 'text-anchor': 'end' });
      text(root, ML - 8, info.y0 + 27, info.artifacts ? `${artifacts.length} consumed` : `${info.ids.length} · ${fmt.usd(usd)}`, 'axlbl', { 'text-anchor': 'end' });
    });
    timeTicks(span).forEach(at => { const tx = x(at); svg('line', { x1: tx, y1: MT - 4, x2: tx, y2: H - MB, class: 'grid' }, root); text(root, tx + 3, MT - 9, fmt.clock(at) + '  +' + fmt.hm(at), 'axlbl'); });
    svg('line', { x1: ML, y1: MT, x2: W - MR, y2: MT, class: 'axis' }, root);
    const edgeGroup = svg('g', {}, root), artifactGroup = svg('g', {}, root), nodeGroup = svg('g', {}, root);
    const artifactEdges = new Map();
    E.forEach((edge, k) => { if (edge[2] === 'artifact') { if (!artifactEdges.has(edge[4])) artifactEdges.set(edge[4], []); artifactEdges.get(edge[4]).push(k); } });
    artifactEdges.forEach((keys, j) => keys.sort((left, right) => A[j].w.indexOf(E[left][0]) - A[j].w.indexOf(E[right][0])));
    E.forEach((edge, k) => {
      if (!edgeVisible(edge)) return; const source = N[edge[0]], target = N[edge[1]]; if (source._y == null || target._y == null) return;
      let path;
      if (edge[2] === 'artifact') {
        if (state.extra.handoffs === 'none') return;
        const artifact = A[edge[4]], written = artifact && artifact.t != null ? artifact.t : source.t0Plot, read = edge[5] != null ? written + edge[5] : target.t0Plot;
        const x1 = clamp(x(written), source._x0, source._x1), x2 = clamp(x(read), target._x0, target._x1);
        path = arc(x1, source._y, x2, target._y, 0.18);
      } else if (edge[2] === 'dep' || edge[2] === 'fan_in') { path = vcurve(source._x1, source._y, target._x0, target._y, 0.5); }
      else { const cx = clamp(target._x0, source._x0, source._x1); path = vcurve(cx, source._y, target._x0, target._y, 0.5); }
      svg('path', { d: path, class: edgeClass(edge), 'data-edge': k }, edgeGroup);
    });
    artifacts.forEach(a => {
      a.w.forEach(i => { const n = N[i]; if (n._y == null) return; const wx = clamp(x(a.t), n._x0, n._x1); svg('path', { d: vcurve(wx, n._y, a._x, a._y - 6, 0.4), class: 'alink', 'data-alink': a.i }, artifactGroup); });
      a.c.forEach(i => { const n = N[i]; if (n._y == null) return; const edgeKey = (artifactEdges.get(a.i) || []).find(k => E[k][1] === i), read = edgeKey != null && E[edgeKey][5] != null ? a.t + E[edgeKey][5] : n.t0Plot, rx = clamp(x(read), n._x0, n._x1); svg('path', { d: vcurve(a._x, a._y + 6, rx, n._y, 0.4), class: 'alink reads', 'data-alink': a.i }, artifactGroup); });
      svg('rect', { x: a._x - 5, y: a._y - 5, width: 10, height: 10, transform: `rotate(45 ${a._x} ${a._y})`, class: 'art', 'data-art': a.i }, artifactGroup);
    });
    vis.forEach(i => {
      const n = N[i]; if (n._y == null) return;
      const height = Math.max(6, Math.sqrt(Math.min(n.usd == null ? 0 : n.usd, 20) / 20) * (rowH - 4)), width = n._x1 - n._x0, group = svg('g', {}, nodeGroup);
      svg('rect', { x: n._x0, y: n._y - height / 2, width, height, rx: 2, fill: color(n, state.colorBy), class: nodeClass(n), 'data-node': i }, group);
      if (width < 12) svg('rect', { x: n._x0 - (12 - width) / 2, y: n._y - rowH / 2, width: 12, height: rowH, class: 'hit', 'data-node': i }, group);
      if (width > 46 && height >= 14) text(group, n._x0 + 5, n._y + 4, n.lbl, 'nlbl', { fill: '#fff' });
      else if (width > 46) text(group, n._x0 + 4, n._y - height / 2 - 3, n.lbl, 'nlbl');
    });
    vis.forEach(i => { delete N[i]._row; });
  },
};
"""
