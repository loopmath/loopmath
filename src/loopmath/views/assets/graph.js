// Shared graph component (lane 12).
// LM.Graph.render(host, graph, opts): a workflow graph {config, label, nodes, edges, gates}, drawn as a
//   layered left-to-right graph. Pieces are boxes, artifacts diamonds, gates are marked on the gated
//   piece's output edge, repair loops are dashed arcs underneath. Piece annotations come from
//   node.prediction (a PiecePrediction: plans, posterior) and node.realized (a run's attempts).
// LM.RunGraph.render(host, data, opts): today's graph/html_data object, drawn with today's Swimlanes,
//   Force and CostCurve layouts through the `api` object they expect.
LM.Graph = (() => {
  const { esc, fmt, num, ivPlain, tip, TAIL_NOTE, pulledUp, tailPlain } = LM;
  const NS = 'ht' + 'tp:' + '/' + '/www.w3.org/2000/svg';
  const mk = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const label = (parent, x, y, s, cls, attrs) => { const e = mk('text', Object.assign({ x, y, class: cls }, attrs || {}), parent); e.textContent = s; return e; };
  const ROLE_COLOR = { planner: '#4a3aa7', plan: '#4a3aa7', implementer: '#eb6834', implement: '#eb6834', worker: '#eb6834', reviewer: '#1a9a6c', review: '#1a9a6c', tester: '#2a78d6', test: '#2a78d6', referee: '#b7791f', select: '#b7791f', integrate: '#6a6a9c' };
  const roleColor = r => ROLE_COLOR[r] || '#898781';
  const STATUS_COLOR = { done: '#1a9a6c', failed: '#c8452f', rejected: '#b7791f', canceled: '#898781', lost: '#898781' };
  const PASS = new Set(['accept', 'pass']), FAIL = new Set(['reject', 'fail', 'error']);
  const clip = (s, n) => { s = String(s == null ? '' : s); return s.length > n ? s.slice(0, n - 1) + '…' : s; };
  const modelOf = s => !s ? null : typeof s.model === 'string' ? s.model : s.model ? (s.model.id || s.model.raw) : null;
  const PW = 184, AW = 16, GAPX = 64, GAPY = 20, M = 18;

  function structure(g) {
    const nodes = (g.nodes || []).filter(n => n && n.id != null), byId = new Map(nodes.map(n => [String(n.id), n]));
    const edges = (g.edges || []).map(e => Array.isArray(e) ? { from: e[0], to: e[1] } : e).filter(e => e && byId.has(String(e.from)) && byId.has(String(e.to))).map(e => ({ from: String(e.from), to: String(e.to) }));
    const ids = nodes.map(n => String(n.id)), out = new Map(ids.map(id => [id, []]));
    edges.forEach((e, k) => out.get(e.from).push(k));
    // Break cycles: an edge to a node still on the DFS stack is a back edge.
    const seen = new Map(), back = new Set();
    ids.forEach(root => {
      if (seen.get(root)) return;
      const stack = [[root, 0]]; seen.set(root, 1);
      while (stack.length) {
        const top = stack[stack.length - 1], list = out.get(top[0]);
        if (top[1] >= list.length) { seen.set(top[0], 2); stack.pop(); continue; }
        const k = list[top[1]++], t = edges[k].to;
        if (seen.get(t) === 1) back.add(k); else if (!seen.get(t)) { seen.set(t, 1); stack.push([t, 0]); }
      }
    });
    const layer = new Map(ids.map(id => [id, 0])), indeg = new Map(ids.map(id => [id, 0]));
    edges.forEach((e, k) => { if (!back.has(k)) indeg.set(e.to, indeg.get(e.to) + 1); });
    const queue = ids.filter(id => indeg.get(id) === 0);
    while (queue.length) {
      const id = queue.shift();
      out.get(id).forEach(k => { if (back.has(k)) return; const t = edges[k].to; layer.set(t, Math.max(layer.get(t), layer.get(id) + 1)); indeg.set(t, indeg.get(t) - 1); if (indeg.get(t) === 0) queue.push(t); });
    }
    const cols = [];
    ids.forEach(id => { const l = layer.get(id); (cols[l] = cols[l] || []).push(id); });
    for (let c = 0; c < cols.length; c++) if (!cols[c]) cols[c] = [];
    // One barycenter sweep so edges cross less.
    const rank = new Map(); cols[0].forEach((id, i) => rank.set(id, i));
    for (let c = 1; c < cols.length; c++) {
      const bary = id => { const ps = edges.filter((e, k) => !back.has(k) && e.to === id && rank.has(e.from)).map(e => rank.get(e.from)); return ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : 1e9; };
      cols[c] = cols[c].map((id, i) => [id, bary(id), i]).sort((a, b) => a[1] - b[1] || a[2] - b[2]).map(x => x[0]);
      cols[c].forEach((id, i) => rank.set(id, i));
    }
    return { nodes, byId, edges, back, cols };
  }

  const perRound = p => p && p.cost_per_round && p.cost_per_round.usd ? p.cost_per_round : null;
  function pieceLines(n) {
    const s = n.setting, p = n.prediction, r = n.realized, lines = [];
    const model = modelOf(s);
    lines.push({ t: clip(model ? `${model}/${s.effort || 'default'}` : 'setting not chosen', 30), c: 'lmg-t' });
    if (r) {
      lines.push({ t: clip(`spent ${fmt.usd(r.usd)}, ${fmt.tok(r.tokens)} tok`, 30), c: 'lmg-t' });
      lines.push({ t: `${r.attempts.length} attempt${r.attempts.length === 1 ? '' : 's'}, ${r.rounds || 0} round${r.rounds === 1 ? '' : 's'}`, c: 'lmg-t m' });
      if (p && p.cost) lines.push({ t: clip(`predicted ${fmt.usd(p.cost.usd.mean)} per run`, 30), c: 'lmg-t m' });
      if (p && perRound(p)) lines.push({ t: clip(`predicted ${fmt.usd(perRound(p).usd.mean)} per round`, 30), c: 'lmg-t m' });
      if (p) tailLines(lines, [p.cost && p.cost.usd, perRound(p) && perRound(p).usd]);
    } else if (p) {
      // D60: `cost` is the piece's expected share of one run; only `cost_per_round` is per round.
      // The unit leads, so a clipped line loses the end of its range, never the unit.
      const bare = v => fmt.usd(v).replace(/^\$/, '');
      const span = x => `${fmt.usd(x.mean)} (${bare(x.lo)} to ${bare(x.hi)})`;
      if (p.cost) lines.push({ t: clip(`per run ${span(p.cost.usd)}`, 30), c: 'lmg-t' });
      const once = (a, b) => a.mean === b.mean && a.lo === b.lo && a.hi === b.hi;  // a piece that runs once: same line twice
      if (perRound(p) && !(p.cost && once(perRound(p).usd, p.cost.usd))) lines.push({ t: clip(`per round ${span(perRound(p).usd)}`, 30), c: 'lmg-t m' });
      if (p.cost && p.cost.tokens) lines.push({ t: clip(`${fmt.tok(p.cost.tokens.mean)} tok per run, ${p.rounds ? fmt.rounds(p.rounds.mean) : 'n/a'} rounds`, 32), c: 'lmg-t m' });
      if (p.gate_pass) lines.push({ t: clip(`gate pass ${fmt.pct(p.gate_pass.mean)} per round`, 30), c: 'lmg-t' });
      tailLines(lines, [p.cost && p.cost.usd, perRound(p) && perRound(p).usd, p.cost && p.cost.tokens, p.rounds, p.gate_pass]);
    }
    return lines;
  }
  // D107: when any value the box shows has its mean above its upper end, the note in full over two
  // box lines (a box line holds 30 characters).
  function tailLines(lines, shown) {
    if (!shown.some(pulledUp)) return;
    const cut = TAIL_NOTE.lastIndexOf(' ', 30);
    lines.push({ t: TAIL_NOTE.slice(0, cut), c: 'lmg-t m tail' }, { t: TAIL_NOTE.slice(cut + 1), c: 'lmg-t m tail' });
  }

  function pieceTip(n) {
    const s = n.setting || {}, p = n.prediction, r = n.realized;
    let h = `<b>${esc(n.id)}</b> <span class="m">${esc(n.role || '')}${n.width > 1 ? ', ' + esc(n.width) + ' parallel copies' : ''}</span>`;
    if (n.setting) h += `<br>${esc(s.harness || 'harness n/a')}: ${esc(modelOf(s) || 'n/a')} / ${esc(s.effort || 'default')}${s.context_policy ? ' <span class="m">(context ' + esc(s.context_policy) + ')</span>' : ''}`;
    if (p) {
      if (p.cost) h += `<br>cost per run ${ivPlain(p.cost.usd, fmt.usd)}<br><span class="m">tokens per run ${ivPlain(p.cost.tokens, fmt.tok)}</span>`;
      if (perRound(p)) h += `<br>cost per round ${ivPlain(perRound(p).usd, fmt.usd)}${perRound(p).tokens ? `<br><span class="m">tokens per round ${ivPlain(perRound(p).tokens, fmt.tok)}</span>` : ''}`;
      if (p.rounds) h += `<br>rounds ${ivPlain(p.rounds, fmt.rounds)}`;
      if (p.gate_pass) h += `<br>gate pass per round ${ivPlain(p.gate_pass, v => fmt.pct(v))}`;
    }
    if (r) {
      h += `<br><b>realized</b> ${fmt.usd(r.usd)}, ${fmt.tok(r.tokens)} tokens`;
      r.attempts.forEach(a => { h += `<br><span class="m">round ${esc(a.round)}</span> ${esc(a.status || 'status n/a')}: ${attemptCost(a)}`; });
    }
    return h;
  }
  const attemptCost = a => `${fmt.usd(a.usd)}; in ${fmt.tok(a.streams && a.streams.in)}, cache read ${fmt.tok(a.streams && a.streams.cache_read)}, cache write ${fmt.tok(a.streams && a.streams.cache_write)}, out ${fmt.tok(a.streams && a.streams.out)}`;
  const attemptTip = a => `<b>attempt ${esc(a.id)}</b> <span class="m">round ${esc(a.round)}</span><br>${esc(a.harness || '')} ${esc(a.model || 'model n/a')} / ${esc(a.effort || 'n/a')}<br>${esc(a.status || 'status n/a')}${a.result ? ', outcome ' + esc(a.result) : ''}<br>${attemptCost(a)}<br><span class="m">${esc(fmt.dt(a.started_at))} to ${esc(fmt.dt(a.ended_at))}</span>`;
  const artifactTip = n => `<b>${esc(n.id)}</b> <span class="m">artifact${n.artifact_kind ? ', ' + esc(n.artifact_kind) : ''}</span>` + (n.versions && n.versions.length ? '<br>' + n.versions.map(v => `v${esc(v.version)} ${esc(v.path || v.id)}${v.supersedes ? ' <span class="m">(supersedes ' + esc(v.supersedes) + ')</span>' : ''}`).join('<br>') : '');

  function gateText(g, after) {
    if (g.results && g.results.length) return g.results.map(r => `r${r.round == null ? '?' : r.round} ${r.value}`).join(', ');
    if (after && after.realized) return 'gate: no verdict recorded';
    const pass = g.pass || (after && after.prediction && after.prediction.gate_pass);
    return pass ? `gate: pass ${fmt.pct(pass.mean)} (${fmt.pct(pass.lo)} to ${fmt.pct(pass.hi)})${tailPlain(pass)}` : `gate: ${g.rule || g.id}`;
  }
  function gateTip(g, after) {
    let h = `<b>gate ${esc(g.id)}</b> <span class="m">after ${esc(g.after)}${g.rule ? ', rule ' + esc(g.rule) : ''}</span>`;
    const pass = g.pass || (after && after.prediction && after.prediction.gate_pass);
    if (pass) h += `<br>pass chance per round ${ivPlain(pass, v => fmt.pct(v))}`;
    if (g.on_fail) h += `<br>on reject: back to ${esc(g.on_fail)}`;
    (g.results || []).forEach(r => { h += `<br>round ${esc(r.round == null ? '?' : r.round)}: ${esc(r.value)}${r.name ? ' <span class="m">(' + esc(r.name) + ')</span>' : ''}`; });
    return h;
  }

  function render(host, g, opts = {}) {
    host.innerHTML = '';
    if (!g || !Array.isArray(g.nodes) || !g.nodes.length) { host.innerHTML = '<p class="empty">No workflow graph for this configuration.</p>'; return; }
    const S = structure(g), box = new Map();
    S.nodes.forEach(n => {
      if (n.kind === 'artifact') { box.set(String(n.id), { w: Math.max(AW, Math.min(150, String(n.id).length * 6.4)), h: 34 + (n.versions && n.versions.length > 1 ? 12 : 0), art: true }); return; }
      const lines = pieceLines(n), chips = n.realized && n.realized.attempts.length ? 16 : 0;
      box.set(String(n.id), { w: PW, h: 24 + lines.length * 14 + chips + 6, lines });
    });
    const colW = S.cols.map(col => Math.max(40, ...col.map(id => box.get(id).w)));
    // A gate label sits on the gated piece's output edge, so the gap after that column fits it.
    const gates = (g.gates || []).map(gt => Object.assign({}, gt, { after: String(gt.after), on_fail: gt.on_fail == null ? null : String(gt.on_fail) }));
    const gapAfter = S.cols.map(col => Math.max(GAPX, ...gates.filter(gt => col.includes(gt.after)).map(gt => Math.min(230, gateText(gt, S.byId.get(gt.after)).length * 6.1 + 14) + 24)));
    const colH = S.cols.map(col => col.reduce((sum, id) => sum + box.get(id).h, 0) + GAPY * Math.max(0, col.length - 1));
    const H0 = Math.max(...colH, 60);
    let x = M;
    S.cols.forEach((col, c) => {
      let y = M + 22 + (H0 - colH[c]) / 2;
      col.forEach(id => { const b = box.get(id); b.x = x + (colW[c] - b.w) / 2; b.y = y; y += b.h + GAPY; });
      x += colW[c] + gapAfter[c];
    });
    // A back edge that returns to a gate's repair target is drawn as that gate's repair loop.
    const repairEdge = new Map();
    gates.forEach(gt => {
      if (!gt.on_fail || !box.has(gt.after) || !box.has(gt.on_fail)) return;
      const k = S.edges.findIndex((e, i) => S.back.has(i) && e.to === gt.on_fail && (e.from === gt.after || S.edges.some(f => f.from === gt.after && f.to === e.from)));
      repairEdge.set(gt.id, k);
    });
    const loops = gates.filter(gt => gt.on_fail && box.has(gt.after) && box.has(gt.on_fail));
    const otherBack = [...S.back].filter(k => ![...repairEdge.values()].includes(k));
    const below = (loops.length + otherBack.length) ? 30 + 22 * (loops.length + otherBack.length) : 0;
    const W = Math.max(x - gapAfter[gapAfter.length - 1] + M, 320), H = M + 22 + H0 + below + M;
    const svg = mk('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: 'lmg-svg', role: 'img' }, host);
    const title = g.label || g.config || '';
    if (title && !opts.noTitle) label(svg, M, M + 4, clip(title, Math.floor(W / 6.5)), 'lmg-title');
    const edgeG = mk('g', {}, svg), markG = mk('g', {}, svg), nodeG = mk('g', {}, svg);
    const anchorOut = id => { const b = box.get(id); return b.art ? [b.x + b.w / 2 + 8, b.y + 9] : [b.x + b.w, b.y + b.h / 2]; };
    const anchorIn = id => { const b = box.get(id); return b.art ? [b.x + b.w / 2 - 8, b.y + 9] : [b.x, b.y + b.h / 2]; };
    const bottom = id => { const b = box.get(id); return [b.x + b.w / 2, b.art ? b.y + 18 : b.y + b.h]; };
    S.edges.forEach((e, k) => {
      if (S.back.has(k)) return;
      const [x1, y1] = anchorOut(e.from), [x2, y2] = anchorIn(e.to), dx = (x2 - x1) * 0.5;
      mk('path', { d: `M${x1},${y1}C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`, class: 'lmg-edge', 'marker-end': 'url(#lmg-arrow)' }, edgeG);
    });
    const defs = mk('defs', {}, svg), marker = mk('marker', { id: 'lmg-arrow', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    mk('path', { d: 'M0,0L8,4L0,8z', class: 'lmg-arrowhead' }, marker);
    let lane = 0;
    const baseY = M + 22 + H0 + 14;
    const loopPath = (from, to) => { const [x1, y1] = bottom(from), [x2, y2] = bottom(to), yy = baseY + 22 * lane++; return { d: `M${x1},${y1}C${x1},${yy} ${x2},${yy} ${x2},${y2}`, mx: (x1 + x2) / 2, my: yy + 6 }; };
    loops.forEach(gt => {
      const after = S.byId.get(gt.after), k = repairEdge.get(gt.id), from = k != null && k >= 0 ? S.edges[k].from : gt.after;
      const p = loopPath(from, gt.on_fail), rejects = (gt.results || []).filter(r => FAIL.has(r.value)).length;
      mk('path', { d: p.d, class: 'lmg-repair', 'marker-end': 'url(#lmg-arrow)', 'data-gate': gt.id }, edgeG);
      let text;
      if (gt.results && gt.results.length) text = `repair: sent back ${rejects} time${rejects === 1 ? '' : 's'}`;
      else if (after && after.realized) { const extra = Math.max(0, (after.realized.rounds || 1) - 1); text = extra ? `repair: ${extra} extra round${extra === 1 ? '' : 's'}` : 'repair loop, not taken'; }
      else { const r = after && after.prediction && after.prediction.rounds; text = `repair loop${r ? ': ' + fmt.rounds(r.mean) + ' rounds (' + fmt.rounds(r.lo) + ' to ' + fmt.rounds(r.hi) + ')' + tailPlain(r) : ''}${g.budget_rounds ? ', at most ' + g.budget_rounds : ''}`; }
      label(markG, p.mx, p.my, text, 'lmg-loop', { 'text-anchor': 'middle', 'data-gate': gt.id });
    });
    otherBack.forEach(k => { const p = loopPath(S.edges[k].from, S.edges[k].to); mk('path', { d: p.d, class: 'lmg-back', 'marker-end': 'url(#lmg-arrow)' }, edgeG); });
    // Gate marks sit on the gated piece's first output edge.
    let right = 0;
    gates.forEach(gt => {
      if (!box.has(gt.after)) return;
      const after = S.byId.get(gt.after), k = S.edges.findIndex((e, i) => !S.back.has(i) && e.from === gt.after);
      const text = clip(gateText(gt, after), 40), w = Math.min(230, text.length * 6.1 + 14);
      let cx, cy;
      if (k >= 0) { const [x1, y1] = anchorOut(gt.after), [x2, y2] = anchorIn(S.edges[k].to); cx = (x1 + x2) / 2; cy = (y1 + y2) / 2; }
      else { const b = box.get(gt.after); cx = b.x + b.w + 10 + w / 2; cy = b.y + b.h / 2 + 12; }
      right = Math.max(right, cx + w / 2);
      const gg = mk('g', { 'data-gate': gt.id, class: 'lmg-gate' }, markG);
      const results = gt.results || [], last = results.length ? results[results.length - 1].value : null;
      mk('rect', { x: cx - w / 2, y: cy - 25, width: w, height: 15, rx: 7, class: 'lmg-gatebox' + (PASS.has(last) ? ' pass' : FAIL.has(last) ? ' fail' : '') }, gg);
      label(gg, cx, cy - 14, text, 'lmg-gatet', { 'text-anchor': 'middle' });
    });
    if (right + M > W) { svg.setAttribute('width', right + M); svg.setAttribute('viewBox', `0 0 ${right + M} ${H}`); }
    S.nodes.forEach((n, i) => {
      const id = String(n.id), b = box.get(id), grp = mk('g', { 'data-n': i }, nodeG);
      if (b.art) {
        const cx = b.x + b.w / 2, cy = b.y + 9;
        mk('rect', { x: cx - 7, y: cy - 7, width: 14, height: 14, transform: `rotate(45 ${cx} ${cy})`, class: 'lmg-art' }, grp);
        label(grp, cx, cy + 22, clip(id, 24), 'lmg-at', { 'text-anchor': 'middle' });
        if (n.versions && n.versions.length > 1) label(grp, cx, cy + 34, n.versions.map(v => 'v' + v.version).join(' > '), 'lmg-at m', { 'text-anchor': 'middle' });
        return;
      }
      const col = roleColor(n.role);
      mk('rect', { x: b.x, y: b.y, width: b.w, height: b.h, rx: 7, class: 'lmg-piece', stroke: col }, grp);
      mk('rect', { x: b.x, y: b.y, width: 5, height: b.h, rx: 2, fill: col }, grp);
      label(grp, b.x + 12, b.y + 16, clip(`${id}${n.role && n.role !== id ? ' (' + n.role + ')' : ''}${n.width > 1 ? ' x' + n.width : ''}`, 28), 'lmg-t1');
      b.lines.forEach((ln, j) => label(grp, b.x + 12, b.y + 31 + j * 14, ln.t, ln.c));
      if (n.realized && n.realized.attempts.length) {
        const y = b.y + b.h - 17;
        n.realized.attempts.slice(0, 14).forEach((a, j) => {
          const cg = mk('g', { 'data-att': `${i}:${j}` }, grp);
          mk('rect', { x: b.x + 12 + j * 12, y, width: 10, height: 10, rx: 2, fill: STATUS_COLOR[a.status] || '#b3b1a9' }, cg);
        });
      }
    });
    if (g.unplaced && g.unplaced.length) label(svg, M, H - 6, `${g.unplaced.length} attempt(s) not tied to a piece of this workflow`, 'lmg-at m');
    const find = t => t.closest ? t.closest('[data-att],[data-n],[data-gate]') : null;
    svg.addEventListener('mousemove', ev => {
      const el = find(ev.target);
      if (!el) { tip(null); return; }
      if (el.dataset.att) { const [i, j] = el.dataset.att.split(':').map(Number); tip(attemptTip(S.nodes[i].realized.attempts[j]), ev.clientX, ev.clientY); return; }
      if (el.dataset.gate) { const gt = gates.find(x => x.id === el.dataset.gate); tip(gateTip(gt, S.byId.get(gt.after)), ev.clientX, ev.clientY); return; }
      const n = S.nodes[+el.dataset.n]; tip(n.kind === 'artifact' ? artifactTip(n) : pieceTip(n), ev.clientX, ev.clientY);
    });
    svg.addEventListener('mouseleave', () => tip(null));
    svg.addEventListener('click', ev => { const el = find(ev.target); if (el && el.dataset.n && opts.onSelect) opts.onSelect(S.nodes[+el.dataset.n]); });
    if (!opts.noLegend) {
      const leg = document.createElement('div'); leg.className = 'lmg-legend';
      leg.innerHTML = 'Boxes are pieces (edge color is the role), diamonds are artifacts, arrows show what each piece reads and writes. ' +
        (S.nodes.some(n => n.realized) ? 'Squares under a piece are its attempts (green done, red failed, amber rejected); hover for the four token streams and dollars. The gate label lists each round\'s verdict.' : 'Costs are per run: the piece\'s expected share of one run, with its 80 percent range (per round too when the prediction gives it); the gate label is the chance the gate passes in one round.') +
        (loops.length ? ' The dashed arc is the repair loop taken when the gate rejects.' : '');
      host.appendChild(leg);
    }
  }
  return { render, structure };
})();

LM.RunGraph = (() => {
  const { esc, tip } = LM;
  const NS = 'ht' + 'tp:' + '/' + '/www.w3.org/2000/svg';
  const layouts = () => [typeof Swimlanes !== 'undefined' && Swimlanes, typeof Force !== 'undefined' && Force, typeof CostCurve !== 'undefined' && CostCurve].filter(Boolean);
  const ROLE_COLOR = { lead: '#2a78d6', dev: '#eb6834', reviewer: '#1baf7a', planner: '#4a3aa7', cli: '#898781', external: '#898781', unlabeled: '#898781' };
  const PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#b7791f', '#c8452f', '#6a6a9c'];

  // The per-graph `api` today's layouts read (same fields as graph/html_common.py builds).
  function prepare(D) {
    const RUN = D.run, E = D.edges || [], N = (D.nodes || []).map(n => Object.assign({}, n)), A = (D.artifacts || []).map(a => Object.assign({}, a));
    const TS = RUN.token_streams || [], TT = RUN.token_total_streams || [];
    N.forEach(n => {
      n.written = []; n.read = []; n.out = []; n.in = []; n.children = [];
      n.untimed = n.t0 == null; n.durationUnknown = n.dur == null;
      n.t0Plot = n.t0 == null ? 0 : n.t0; n.durPlot = n.dur == null ? 0 : n.dur; n.t1 = n.untimed ? n.t0Plot : n.t0Plot + n.durPlot;
      const tok = n.tok || {};
      n.tokComplete = !!n.tok_record && TS.every(k => Number.isFinite(tok[k]));
      n.tokTotal = n.tokComplete ? TT.reduce((s, k) => s + tok[k], 0) : null;
      n.modelKey = n.model || 'unknown';
    });
    A.forEach(a => { (a.w || []).forEach(i => N[i] && N[i].written.push(a.i)); (a.c || []).forEach(i => N[i] && N[i].read.push(a.i)); });
    E.forEach((e, k) => { if (N[e[0]]) N[e[0]].out.push(k); if (N[e[1]]) N[e[1]].in.push(k); });
    N.forEach(n => { if (Number.isInteger(n.parent) && N[n.parent]) N[n.parent].children.push(n.i); });
    const roles = [...new Set(N.map(n => n.role))], vocab = (RUN.role_vocabulary || []).filter(r => r !== 'external');
    const ROLES = [...vocab.filter(r => roles.includes(r)), ...roles.filter(r => !vocab.includes(r) && r !== 'external' && r !== 'unlabeled').sort(), ...roles.filter(r => r === 'external'), ...roles.filter(r => r === 'unlabeled')];
    const MODELS = [...new Set(N.map(n => n.modelKey))].sort();
    const MODEL_COLOR = {}; MODELS.forEach((m, i) => { MODEL_COLOR[m] = m === 'unknown' ? '#898781' : PALETTE[i % PALETTE.length]; });
    const MAX_USD = Math.max(1, ...N.map(n => n.usd == null ? 0 : n.usd));
    const fmt = {
      usd: v => v == null ? 'n/a' : (v < 0.1 ? '$' + v.toFixed(3) : '$' + v.toFixed(2)),
      dur: s => { if (s == null) return 'n/a'; s = Math.round(s); if (s < 60) return s + 's'; const m = Math.floor(s / 60), h = Math.floor(m / 60); return h ? h + 'h ' + String(m % 60).padStart(2, '0') + 'm' : m + 'm ' + String(s % 60).padStart(2, '0') + 's'; },
      tok: v => v == null ? 'n/a' : v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(0) + 'k' : String(v),
      int: v => v == null ? 'n/a' : Number(v).toLocaleString('en-US'),
      hm: sec => { const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60); return h + 'h' + String(m).padStart(2, '0'); },
      clock: sec => RUN.start ? new Date(Date.parse(RUN.clock_base) + sec * 1000).toISOString().slice(11, 16) + 'Z' : '+' + fmt.hm(sec),
      ts: iso => iso ? iso.replace('T', ' ').replace(/\.\d+/, '').replace('Z', ' UTC') : 'n/a',
      tier: v => v || 'unknown',
      tokenLabel: key => key.replaceAll('_', ' '),
    };
    const svg = (tag, attrs, parent) => { const el = document.createElementNS(NS, tag); for (const k in attrs) if (attrs[k] != null) el.setAttribute(k, attrs[k]); if (parent) parent.appendChild(el); return el; };
    const text = (parent, x, y, value, cls, attrs) => { const el = svg('text', Object.assign({ x, y, class: cls }, attrs || {}), parent); el.textContent = value; return el; };
    const curve = (x1, y1, x2, y2, bend = 0.5) => { const dx = (x2 - x1) * bend; return `M${x1},${y1}C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`; };
    const vcurve = (x1, y1, x2, y2, bend = 0.5) => { const dy = (y2 - y1) * bend; return `M${x1},${y1}C${x1},${y1 + dy} ${x2},${y2 - dy} ${x2},${y2}`; };
    const arc = (x1, y1, x2, y2, lift = 0.25) => { const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1; const nx = -dy / len, ny = dx / len; return `M${x1},${y1}Q${mx + nx * len * lift},${my + ny * len * lift} ${x2},${y2}`; };
    const timeTicks = (span, target = 8) => { const steps = [300, 600, 900, 1800, 3600, 7200]; const step = steps.find(v => span / v <= target) || 7200; const out = []; for (let at = 0; at <= span; at += step) out.push(at); return out; };
    const color = (n, mode) => mode === 'model' ? (MODEL_COLOR[n.modelKey] || '#898781') : (ROLE_COLOR[n.role] || '#898781');
    const dashed = n => n.mt !== 'verified';
    const radius = (n, max = 26, min = 4) => n.usd == null ? min : Math.max(min, Math.sqrt(n.usd / MAX_USD) * max);
    const all = new Set(N.map(n => n.i));
    const edgeVisible = e => all.has(e[0]) && all.has(e[1]);
    const artVisible = a => (a.c || []).length > 0;
    const edgeClass = e => `edge e-${e[2]} t-${e[3]}`;
    const nodeClass = n => 'node' + (dashed(n) ? ' dashed' : '') + (n.untimed || n.durationUnknown || n.usd == null ? ' hollow' : '');
    return { N, E, A, RUN, ROLES, MODELS, ROLE_COLOR, MODEL_COLOR, MAX_USD, fmt, esc, color, dashed, radius, svg, text, curve, vcurve, arc, timeTicks, edgeClass, nodeClass, edgeVisible, artVisible };
  }

  function nodeTip(api, n) {
    const { fmt, RUN } = api, tok = n.tok || {};
    const streams = (RUN.token_total_streams || []).map(k => `${esc(fmt.tokenLabel(String(k)))} ${fmt.int(tok[k])}`).join(', ');
    return `<b>${esc(n.lbl)}</b> <span class="m">${esc(n.title)}</span><br>${esc(n.role)} on ${esc(n.model || 'n/a')} <span class="m">(${esc(fmt.tier(n.mt))})</span>${n.effort ? ', effort ' + esc(n.effort) : ''}<br>${fmt.usd(n.usd)}, ${n.tokComplete ? fmt.tok(n.tokTotal) + ' tokens' : 'token total unavailable'}<br><span class="m">${streams}</span><br><span class="m">${n.untimed ? 'start unavailable' : 'start ' + fmt.clock(n.t0)}, duration ${fmt.dur(n.dur)}; wrote ${n.written.length}, read ${n.read.length} artifacts</span>`;
  }
  function artTip(api, a) {
    const { N } = api;
    return `<b>${esc(a.path)}</b><br>${esc(a.kind || 'unknown kind')}<br>written by ${(a.w || []).map(i => esc(N[i].lbl)).join(', ') || 'nobody in scope'}<br>read by ${(a.c || []).map(i => esc(N[i].lbl)).join(', ') || 'nobody'}`;
  }
  function edgeTip(api, e) {
    const { N, A } = api, s = N[e[0]], t = N[e[1]];
    return `<b>${esc(e[2] === 'artifact' ? 'handoff' : e[2])}</b> ${esc(s.lbl)} to ${esc(t.lbl)} <span class="m">(${esc(e[3])})</span>` + (e[2] === 'artifact' && A[e[4]] ? `<br>${esc(A[e[4]].path)}` : '');
  }

  function render(host, D, opts = {}) {
    host.innerHTML = '';
    if (!D || !D.run || !Array.isArray(D.nodes)) { host.innerHTML = '<p class="empty">No attempt graph for this run.</p>'; return; }
    if (!D.nodes.length) { host.innerHTML = '<p class="empty">This run has no attempts yet.</p>'; return; }
    const api = prepare(D), items = layouts(), views = {};
    if (!items.length) { host.innerHTML = '<p class="empty">The attempt layouts are not included in this page.</p>'; return; }
    items.forEach(l => { const extra = {}; (l.controls || []).forEach(c => { extra[c.id] = c.value; }); views[l.key] = { colorBy: l.colorBy || 'role', artifacts: false, extra }; });
    let current = opts.layout && views[opts.layout] ? opts.layout : items[0].key;
    const bar = document.createElement('div'); bar.className = 'lg-bar';
    const wrap = document.createElement('div'); wrap.className = 'lg lg-wrap';
    const note = document.createElement('p'); note.className = 'lmg-legend';
    host.append(bar, wrap, note);
    const names = { swim: 'timeline by role', force: 'force', cost: 'cost curve' };
    function draw() {
      const layout = items.find(l => l.key === current), view = views[current];
      bar.innerHTML = items.map(l => `<button data-layout="${l.key}" class="${l.key === current ? 'on' : ''}">${esc(names[l.key] || l.key)}</button>`).join(' ') +
        ` <label class="small m">color <select data-color><option value="role"${view.colorBy === 'role' ? ' selected' : ''}>by role</option><option value="model"${view.colorBy === 'model' ? ' selected' : ''}>by model</option></select></label>` +
        (layout.artifactMode !== 'always' ? ` <label class="small m"><input type="checkbox" data-arts${view.artifacts ? ' checked' : ''}> artifacts</label>` : '');
      wrap.innerHTML = '';
      const inner = document.createElement('div'); inner.className = 'graph'; wrap.appendChild(inner);
      layout.render(inner, api.N.map(n => n.i), view, api);
      note.textContent = `${layout.graphTitle}. ${layout.sizeNote ? layout.sizeNote + '.' : ''}`;
    }
    bar.addEventListener('click', ev => { const b = ev.target.closest('[data-layout]'); if (b) { current = b.dataset.layout; draw(); } });
    bar.addEventListener('change', ev => { const v = views[current]; if (ev.target.matches('[data-color]')) v.colorBy = ev.target.value; else if (ev.target.matches('[data-arts]')) v.artifacts = ev.target.checked; draw(); });
    const focus = new Set();
    wrap.addEventListener('mousemove', ev => {
      const el = ev.target.closest ? ev.target.closest('[data-node],[data-art],[data-edge]') : null;
      wrap.querySelectorAll('.focus').forEach(x => x.classList.remove('focus'));
      if (!el) { tip(null); return; }
      el.classList.add('focus');
      const html = el.dataset.node != null ? nodeTip(api, api.N[+el.dataset.node]) : el.dataset.art != null ? artTip(api, api.A[+el.dataset.art]) : edgeTip(api, api.E[+el.dataset.edge]);
      tip(html, ev.clientX, ev.clientY);
    });
    wrap.addEventListener('mouseleave', () => { tip(null); focus.clear(); });
    draw();
  }
  return { render, prepare };
})();

// Which drawing a graph object needs: today's html_data (run, indexed nodes) or a workflow graph.
LM.isRunGraph = g => !!(g && g.run && typeof g.run === 'object' && Array.isArray(g.nodes) && Array.isArray(g.edges) && Array.isArray(g.artifacts));
