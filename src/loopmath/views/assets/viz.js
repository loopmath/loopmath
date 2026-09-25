// Shared visual system for the results page and the planning page (lane 2E, 0.2): number formats, linear and
// log scales, distribution shapes, workflow graphs (a piece of width n drawn as n worker boxes), ridges, forests,
// score against cost, the payback chart, inline bars, tooltips and copy buttons. Pages render from #data only.
LM.Viz = (() => {
  const { esc, num, fmt } = LM;
  const NS = 'ht' + 'tp:' + '/' + '/www.w3.org/2000/svg';
  const el = (tag, attrs, parent, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  };
  const h = (tag, attrs, parent, html) => {
    const e = document.createElement(tag);
    for (const k in attrs || {}) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    if (parent) parent.appendChild(e);
    return e;
  };

  // ---------------------------------------------------------------- numbers
  const usd = v => fmt.usd(v);
  // Axis ticks: one significant figure under a dollar, whole dollars above.
  const usd0 = v => !num(v) ? 'n/a' : v >= 1 ? '$' + Math.round(v).toLocaleString('en-US') : '$' + Number(v.toPrecision(1));
  const pct = v => fmt.pct(v);
  const int = v => fmt.int(v);
  const score = v => !num(v) ? 'n/a' : Math.abs(v) >= 100 ? Math.round(v).toLocaleString('en-US') : Number(v.toPrecision(3)).toString();
  const runs = n => !num(n) ? 'n/a' : n === 1 ? '1 run' : `${fmt.int(n)} runs`;
  const range = (x, f) => !x || !num(x.lo) || !num(x.hi) ? '' : `${f(x.lo)} to ${f(x.hi)}`;
  const clamp01 = v => num(v) ? Math.max(0, Math.min(1, v)) : null;
  const modelName = m => !m ? '' : typeof m === 'string' ? m : m.id || m.raw || '';
  const clip = (s, n) => { s = String(s == null ? '' : s); return s.length > n ? s.slice(0, n - 1) + '…' : s; };

  // ---------------------------------------------------------------- scales
  const lin = (d0, d1, r0, r1) => {
    const span = (d1 - d0) || 1, f = v => r0 + (v - d0) / span * (r1 - r0);
    f.d = [d0, d1]; f.r = [r0, r1]; return f;
  };
  const log = (d0, d1, r0, r1) => {
    d0 = Math.max(d0, 1e-6); d1 = Math.max(d1, d0 * 1.0001);
    const a = Math.log(d0), b = Math.log(d1), f = v => r0 + (Math.log(Math.max(num(v) ? v : d0, d0 * 0.5)) - a) / (b - a) * (r1 - r0);
    f.d = [d0, d1]; f.r = [r0, r1]; f.log = true; return f;
  };
  const niceTicks = (d0, d1, n) => {
    const step0 = (d1 - d0) / Math.max(1, n || 4);
    if (!(step0 > 0)) return [d0];
    const mag = Math.pow(10, Math.floor(Math.log10(step0))), step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= step0) || step0;
    const out = []; for (let v = Math.ceil(d0 / step) * step; v <= d1 + 1e-9 && out.length < 50; v += step) out.push(+v.toFixed(10));
    return out;
  };
  const logTicks = (d0, d1) => {
    const pick = ms => { const out = []; for (let e = Math.floor(Math.log10(d0)); e <= Math.ceil(Math.log10(d1)); e++) for (const m of ms) { const v = m * Math.pow(10, e); if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(+v.toPrecision(6)); } return out; };
    let out = pick([1, 3]);
    if (out.length < 3) out = pick([1, 2, 5]);
    return out;
  };
  // A log domain around a set of values, padded a little on both sides.
  const logDomain = values => {
    const vs = values.filter(v => num(v) && v > 0);
    if (!vs.length) return [0.1, 10];
    const lo = Math.min(...vs), hi = Math.max(...vs);
    return [lo / 1.6, Math.max(hi * 1.6, lo * 4)];
  };

  // ---------------------------------------------------------------- distribution shapes
  // No quantiles in the JSON yet, so a distribution is drawn as a split normal through the mean and the 80% range;
  // on a log axis, as a normal in log space around the geometric middle of the range.
  const Z80 = 1.2816;
  function density(x, logScale) {
    if (logScale) {
      const lo = Math.log(Math.max(x.lo, 1e-6)), hi = Math.log(Math.max(x.hi, 1e-6)), c = (lo + hi) / 2, s = (hi - lo) / (2 * Z80) || 0.1;
      return v => Math.exp(-0.5 * Math.pow((Math.log(Math.max(v, 1e-9)) - c) / s, 2));
    }
    const m = num(x.mean) ? x.mean : (x.lo + x.hi) / 2;
    const sl = Math.max((m - x.lo) / Z80, 1e-9), sh = Math.max((x.hi - m) / Z80, 1e-9);
    return v => Math.exp(-0.5 * Math.pow((v - m) / (v < m ? sl : sh), 2));
  }
  // A filled half violin above a baseline ('up') or a mirrored one ('mid').
  function shapePath(x, sx, yBase, hgt, mode) {
    if (!x || !num(x.lo) || !num(x.hi)) return '';
    const f = density(x, !!sx.log), [r0, r1] = sx.r, n = 90, pts = [];
    for (let i = 0; i <= n; i++) {
      const v = sx.log ? Math.exp(Math.log(sx.d[0]) + (Math.log(sx.d[1]) - Math.log(sx.d[0])) * i / n) : sx.d[0] + (sx.d[1] - sx.d[0]) * i / n;
      pts.push([r0 + (r1 - r0) * i / n, f(v)]);
    }
    const keep = pts.filter(p => p[1] > 0.01 && Number.isFinite(p[1]));
    if (!keep.length) return '';
    const P = (x0, y0) => `${x0.toFixed(1)},${y0.toFixed(1)}`;
    if (mode === 'mid') {
      const top = keep.map(p => P(p[0], yBase - p[1] * hgt / 2)), bot = keep.slice().reverse().map(p => P(p[0], yBase + p[1] * hgt / 2));
      return 'M' + top.join('L') + 'L' + bot.join('L') + 'Z';
    }
    return `M${P(keep[0][0], yBase)}L` + keep.map(p => P(p[0], yBase - p[1] * hgt)).join('L') + `L${P(keep[keep.length - 1][0], yBase)}Z`;
  }

  // ---------------------------------------------------------------- tooltips and copy buttons
  const hover = (node, html) => {
    const text = () => typeof html === 'function' ? html() : html;
    node.addEventListener('mousemove', e => LM.tip(text(), e.clientX, e.clientY));
    node.addEventListener('mouseleave', () => LM.tip(null));
    node.addEventListener('touchstart', e => { const t = e.touches[0]; LM.tip(text(), t.clientX, t.clientY); }, { passive: true });
  };
  const copyBtn = (cmd, label) => `<div class="v-cmd"><code>${esc(cmd)}</code><button type="button" data-copy="${esc(cmd)}">${esc(label || 'copy')}</button></div>`;
  document.addEventListener('click', e => { const b = e.target.closest && e.target.closest('[data-copy]'); if (b) LM.copy(b.dataset.copy, b); });
  const onResize = fn => { let t, w = window.innerWidth; window.addEventListener('resize', () => { if (window.innerWidth === w) return; w = window.innerWidth; clearTimeout(t); t = setTimeout(fn, 120); }); };

  // ---------------------------------------------------------------- workflow graphs
  // Input: a view graph {config, label, nodes, edges, gates} (views/common.py workflow_graph). Pieces are boxes
  // with their role, model and effort, and their share of the predicted cost per run; a piece of width n is n
  // worker boxes (up to 6, then marked xn), each with 1/n of the piece's cost, as the cost model prices it.
  // Artifacts are small pills (the repo is left out). A gate's repair loop is a dashed arc back to the piece it reruns.
  const MAX_BOXES = 6;
  function shape(g) {
    const nodes = (g && g.nodes) || [];
    const pieces = nodes.filter(n => n && n.kind === 'piece').map(n => {
      const s = n.setting || {}, c = n.prediction && n.prediction.cost && n.prediction.cost.usd;
      return { id: String(n.id), role: n.role || String(n.id), width: Math.max(1, n.width || 1), model: modelName(s.model), effort: s.effort || '', cost: c && num(c.mean) ? c.mean : null };
    });
    const arts = nodes.filter(n => n && n.kind === 'artifact').map(n => String(n.id));
    const edges = ((g && g.edges) || []).map(e => Array.isArray(e) ? [String(e[0]), String(e[1])] : [String(e.from), String(e.to)]);
    return { pieces, arts, edges, gates: (g && g.gates) || [] };
  }
  function graph(host, g, opts = {}) {
    const mini = !!opts.mini, w = shape(g), byId = Object.fromEntries(w.pieces.map(p => [p.id, p]));
    const shown = a => !/^repo$/i.test(a);
    const ids = mini ? w.pieces.map(p => p.id) : [...w.arts.filter(shown), ...w.pieces.map(p => p.id)];
    const all = new Set([...w.arts, ...w.pieces.map(p => p.id)]);
    const edges = w.edges.filter(e => all.has(e[0]) && all.has(e[1]));
    const layer = {}; all.forEach(i => { layer[i] = 0; });
    for (let k = 0; k < all.size; k++) edges.forEach(([a, b]) => { if (layer[b] < layer[a] + 1) layer[b] = layer[a] + 1; });
    const cols = {}; ids.forEach(id => (cols[layer[id]] = cols[layer[id]] || []).push(id));
    const keys = Object.keys(cols).map(Number).sort((a, b) => a - b);
    const text2 = p => `${p.model}/${p.effort}` + (p.cost != null ? '  ' + usd(p.cost / p.width) : '');
    const longest = Math.max(12, ...w.pieces.map(p => Math.max(text2(p).length, (p.role + ' 1 of 9').length)));
    const BW = mini ? 16 : Math.max(128, Math.min(230, Math.round(longest * 6.7 + 16))), BH = mini ? 10 : 34;
    const GX = mini ? 10 : 30, GY = mini ? 3 : 8, AW = mini ? 0 : 64, AH = 18;
    const count = id => byId[id] ? Math.min(byId[id].width, MAX_BOXES) : 1;
    const colW = keys.map(c => Math.max(...cols[c].map(id => byId[id] ? BW : AW)));
    const heights = keys.map(c => cols[c].reduce((s, id) => s + (byId[id] ? count(id) * (BH + GY) : AH + GY), 0) - GY);
    const maxH = Math.max(BH, ...heights);
    const boxes = {}; let x = 4;
    keys.forEach((c, ci) => {
      let y = 4 + (maxH - heights[ci]) / 2;
      cols[c].forEach(id => {
        if (byId[id]) { const list = []; for (let k = 0; k < count(id); k++) { list.push({ x, y, w: BW, h: BH }); y += BH + GY; } boxes[id] = { piece: byId[id], list }; }
        else { boxes[id] = { art: id, list: [{ x: x + (colW[ci] - AW) / 2, y, w: AW, h: AH }] }; y += AH + GY; }
      });
      x += colW[ci] + GX;
    });
    const loops = (w.gates || []).filter(gt => gt && boxes[gt.after] && boxes[gt.on_fail]);
    const W = Math.max(x - GX + 4, 20), H = maxH + 8 + (loops.length ? (mini ? 6 : 14) : 0);
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'v-graph' + (mini ? ' mini' : ''), role: 'img', 'aria-label': (g && g.label) || 'workflow graph' }, host);
    const pairs = [];
    if (mini) {
      const into = {}; edges.forEach(([a, b]) => (into[b] = into[b] || []).push(a));
      const from = id => { const out = new Set(), seen = new Set(), st = [id]; while (st.length) { const n = st.pop(); (into[n] || []).forEach(a => { if (byId[a]) out.add(a); else if (!seen.has(a)) { seen.add(a); st.push(a); } }); } return [...out]; };
      w.pieces.forEach(p => from(p.id).forEach(a => { if (a !== p.id) pairs.push([a, p.id]); }));
    } else edges.forEach(e => { if (boxes[e[0]] && boxes[e[1]]) pairs.push(e); });
    pairs.forEach(([a, b]) => boxes[a].list.forEach(s => boxes[b].list.forEach(t => {
      const x1 = s.x + s.w, y1 = s.y + s.h / 2, x2 = t.x, y2 = t.y + t.h / 2, mx = (x1 + x2) / 2;
      el('path', { d: `M${x1},${y1}C${mx},${y1} ${mx},${y2} ${x2},${y2}`, class: 'v-edge' }, svg);
    })));
    loops.forEach(gt => {
      const a = boxes[gt.after].list, b = boxes[gt.on_fail].list, s = a[a.length - 1], t = b[b.length - 1];
      const yb = Math.max(s.y + s.h, t.y + t.h) + (mini ? 5 : 12);
      el('path', { d: `M${s.x + s.w / 2},${s.y + s.h}Q${(s.x + s.w / 2 + t.x + t.w / 2) / 2},${yb} ${t.x + t.w / 2},${t.y + t.h}`, class: 'v-loop' }, svg);
    });
    Object.values(boxes).forEach(b => b.list.forEach((r, k) => {
      if (b.art) {
        el('rect', { x: r.x, y: r.y, width: r.w, height: r.h, rx: 9, class: 'v-art' }, svg);
        el('text', { x: r.x + r.w / 2, y: r.y + 12.5, class: 'v-art-t', 'text-anchor': 'middle' }, svg, clip(b.art.replace(/_doc$/, ''), 10));
        return;
      }
      const p = b.piece;
      el('rect', { x: r.x, y: r.y, width: r.w, height: r.h, rx: mini ? 2 : 5, class: 'v-box' }, svg);
      if (!mini) {
        el('text', { x: r.x + 7, y: r.y + 13, class: 'v-box-r' }, svg, p.width > 1 ? `${p.role} ${k + 1} of ${p.width}` : p.role);
        el('text', { x: r.x + 7, y: r.y + 27, class: 'v-box-s' }, svg, text2(p));
      }
      if (p.width > MAX_BOXES && k === MAX_BOXES - 1) el('text', { x: r.x + r.w + 3, y: r.y + r.h, class: 'v-art-t' }, svg, 'x' + p.width);
    }));
    return svg;
  }
  const miniGraph = g => { const d = document.createElement('span'); graph(d, g, { mini: true }); return `<span class="v-gmini">${d.innerHTML}</span>`; };

  // ---------------------------------------------------------------- chart frames
  function frame(host, height) {
    const W = Math.max(280, Math.round(host.clientWidth || 600));
    const svg = el('svg', { viewBox: `0 0 ${W} ${height}`, width: '100%', height, class: 'v-chart' }, host);
    return { svg, W, H: height };
  }
  function axisX(svg, sx, y, ticks, f, grid, top) {
    ticks.forEach(t => {
      const x = sx(t);
      if (grid) el('line', { x1: x, x2: x, y1: grid[0], y2: grid[1], class: 'v-grid' }, svg);
      el('text', { x, y: top ? y - 6 : y + 14, 'text-anchor': 'middle', class: 'v-tick' }, svg, f(t));
    });
  }

  // Rows of distributions on one shared axis, with run dots and an optional target line.
  // rows: [{key, label, sub, x: {mean, lo, hi}, dots: [{v, cls, tip}], mark, dim, tip}]
  function ridge(host, rows, o) {
    host.innerHTML = '';
    const rowH = o.rowH || 32, cw = host.clientWidth || 600;
    const labelW = o.labelW != null ? o.labelW : Math.min(240, Math.max(118, cw * 0.34));
    const top = 22, H = top + rows.length * rowH + 22, { svg, W } = frame(host, H);
    const sx = (o.log ? log : lin)(o.domain[0], o.domain[1], labelW + 8, W - 14);
    const ticks = o.log ? logTicks(sx.d[0], sx.d[1]) : niceTicks(o.domain[0], o.domain[1], Math.max(3, Math.floor((W - labelW) / 90)));
    axisX(svg, sx, top, ticks, o.fmt || String, [top, H - 22], true);
    if (num(o.target)) {
      const x = sx(o.target);
      el('line', { x1: x, x2: x, y1: top - 2, y2: H - 20, class: 'v-target' }, svg);
      el('text', { x: x + 4, y: H - 8, class: 'v-target-t' }, svg, o.targetLabel || 'target');
    }
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2 + 2, base = y + rowH * 0.3;
      const g = el('g', { class: 'v-row' + (r.mark ? ' ' + r.mark : '') + (r.dim ? ' dim' : ''), 'data-key': r.key }, svg);
      const hit = el('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, class: 'v-hit' }, g);
      if (r.label) el('text', { x: 4, y: y + (r.sub ? -2 : 4), class: 'v-rl' }, g, clip(r.label, Math.floor(labelW / 6.6)));
      if (r.sub) el('text', { x: 4, y: y + 11, class: 'v-rs' }, g, clip(r.sub, Math.floor(labelW / 5.6)));
      if (r.x && num(r.x.lo) && num(r.x.hi)) {
        el('path', { d: shapePath(r.x, sx, base, rowH * 0.62, o.mode || 'up'), class: 'v-dist' }, g);
        el('line', { x1: sx(r.x.lo), x2: sx(r.x.hi), y1: base, y2: base, class: 'v-iv' }, g);
        if (num(r.x.mean)) el('line', { x1: sx(r.x.mean), x2: sx(r.x.mean), y1: base - 7, y2: base + 1, class: 'v-mean' }, g);
      }
      (r.dots || []).forEach(d => { const c = el('circle', { cx: sx(d.v), cy: base, r: 4, class: 'v-dot' + (d.cls ? ' ' + d.cls : '') }, g); if (d.tip) hover(c, d.tip); });
      if (r.tip) hover(hit, r.tip);
      if (o.onClick) hit.addEventListener('click', () => o.onClick(r));
    });
    return { svg, sx };
  }

  // Intervals on a log axis, faded towards their ends, so very wide ranges stay light.
  let forestId = 0;
  function forest(host, rows, o) {
    host.innerHTML = '';
    const id = 'vf' + (++forestId), rowH = o.rowH || 24, cw = host.clientWidth || 600;
    const labelW = o.labelW != null ? o.labelW : Math.min(270, Math.max(110, cw * 0.36));
    const top = 22, H = top + rows.length * rowH + 16, { svg, W } = frame(host, H);
    const sx = log(o.domain[0], o.domain[1], labelW + 8, W - 12), defs = el('defs', {}, svg);
    ['n', 'a', 'u'].forEach(k => {
      const lg = el('linearGradient', { id: `${id}-${k}`, x1: 0, x2: 1 }, defs);
      [[0, 0.1], [0.35, 0.45], [0.5, 0.55], [0.65, 0.45], [1, 0.1]].forEach(([off, op]) => el('stop', { offset: off, 'stop-opacity': op, class: 'v-stop-' + k }, lg));
    });
    axisX(svg, sx, top, logTicks(sx.d[0], sx.d[1]), o.fmt || usd0, [top, H - 12], true);
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2 + 1, g = el('g', { class: 'v-row' + (r.mark ? ' ' + r.mark : ''), 'data-key': r.key }, svg);
      el('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, class: 'v-hit' }, g);
      el('text', { x: 4, y: y + 4, class: 'v-rl' }, g, clip(r.label, Math.floor(labelW / 6.4)));
      if (r.x && num(r.x.lo) && num(r.x.hi)) {
        const k = r.mark === 'pick' ? 'a' : r.mark === 'ref' ? 'u' : 'n';
        el('rect', { x: sx(r.x.lo), y: y - 4, width: Math.max(2, sx(r.x.hi) - sx(r.x.lo)), height: 8, rx: 4, fill: `url(#${id}-${k})` }, g);
      }
      if (r.x && num(r.x.mean)) el('circle', { cx: sx(r.x.mean), cy: y, r: r.mark ? 5 : 4, class: 'v-fdot' }, g);
      if (r.tip) hover(g, r.tip);
    });
    return svg;
  }

  // Score against cost per run, one small panel per graph, the cheapest workflow whose mean reaches each level marked.
  // items: [{key, group, name, perf {mean, lo, hi}, cost {mean, lo, hi}, dots: [{usd, score}]}]
  function scoreCost(host, items, o) {
    host.innerHTML = '';
    const groups = {}; items.forEach(w => (groups[w.group] = groups[w.group] || []).push(w));
    const better = o.better === 'lower' ? -1 : 1, reaches = (v, L) => better > 0 ? v >= L : v <= L;
    const cheapest = {};
    (o.levels || []).forEach(L => {
      const ok = items.filter(w => w.perf && num(w.perf.mean) && reaches(w.perf.mean, L) && w.cost && num(w.cost.mean)).sort((a, b) => a.cost.mean - b.cost.mean);
      if (ok[0]) (cheapest[ok[0].key] = cheapest[ok[0].key] || []).push(L);
    });
    const wrap = h('div', { class: 'v-multiples' }, host);
    Object.keys(groups).forEach(k => {
      const cell = h('div', { class: 'v-mult', 'data-group': k }, wrap);
      h('div', { class: 'v-mult-h' }, cell, `${esc(k)} <span class="muted">${groups[k].length}</span>`);
      const H = 190, W = Math.max(220, Math.round(cell.clientWidth || 260)), L = 44, R = 10, T = 8, B = 28;
      const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, class: 'v-chart' }, cell);
      const sx = log(o.xDomain[0], o.xDomain[1], L, W - R), sy = lin(o.yDomain[0], o.yDomain[1], H - B, T);
      niceTicks(o.yDomain[0], o.yDomain[1], 4).forEach(v => { el('line', { x1: L, x2: W - R, y1: sy(v), y2: sy(v), class: 'v-grid' }, svg); el('text', { x: L - 5, y: sy(v) + 4, 'text-anchor': 'end', class: 'v-tick' }, svg, score(v)); });
      logTicks(sx.d[0], sx.d[1]).forEach(t => el('text', { x: sx(t), y: H - B + 14, 'text-anchor': 'middle', class: 'v-tick' }, svg, usd0(t)));
      if (num(o.target)) el('line', { x1: L, x2: W - R, y1: sy(o.target), y2: sy(o.target), class: 'v-target' }, svg);
      let nl = 0;
      groups[k].slice().sort((a, b) => a.cost.mean - b.cost.mean).forEach(w => {
        if (!w.perf || !num(w.perf.mean) || !w.cost || !num(w.cost.mean)) return;
        if (num(w.perf.lo) && num(w.perf.hi)) el('line', { x1: sx(w.cost.mean), x2: sx(w.cost.mean), y1: sy(w.perf.lo), y2: sy(w.perf.hi), class: 'v-xr' }, svg);
        (w.dots || []).forEach(d => { if (num(d.usd) && num(d.score)) el('circle', { cx: sx(d.usd), cy: sy(d.score), r: 2.6, class: 'v-dot small' + (d.cls ? ' ' + d.cls : '') }, svg); });
        const best = cheapest[w.key];
        const c = el('circle', { cx: sx(w.cost.mean), cy: sy(w.perf.mean), r: best ? 6 : 4.5, class: 'v-pt' + (best ? ' best' : ''), 'data-key': w.key }, svg);
        if (best) { const left = sx(w.cost.mean) > W * 0.62; el('text', { x: sx(w.cost.mean) + (left ? -9 : 9), y: sy(w.perf.mean) + (nl++ % 2 ? 15 : -7), 'text-anchor': left ? 'end' : 'start', class: 'v-lbl' }, svg, best.map(score).join(', ')); }
        hover(c, `<b>${esc(w.name)}</b><br>score ${esc(score(w.perf.mean))} (${esc(score(w.perf.lo))} to ${esc(score(w.perf.hi))})<br>cost per run ${esc(usd(w.cost.mean))} (${esc(usd(w.cost.lo))} to ${esc(usd(w.cost.hi))})` + (best ? `<br>cheapest mean cost to reach ${esc(best.map(score).join(', '))}` : ''));
      });
    });
    return { cheapest };
  }

  // The exploration bet: net saving after n future runs = saving per run x n - price now; payback where it crosses zero.
  // bets: [{name, gain, price {mean, lo, hi}, payback, payback_lo, payback_hi}]
  function payback(host, bets, o = {}) {
    host.innerHTML = '';
    const H = o.height || 230, { svg, W } = frame(host, H), L = 52, R = 16, T = 12, B = 34;
    const N = o.runs || Math.max(4, Math.min(20, Math.ceil(Math.max(...bets.map(b => num(b.payback_hi) ? b.payback_hi : b.payback || 1)) * 1.3)));
    const ys = bets.flatMap(b => [-b.price.hi, b.gain * N - b.price.lo]);
    const sx = lin(0, N, L, W - R), sy = lin(Math.min(...ys, 0) * 1.1, Math.max(...ys, 0) * 1.05 || 1, H - B, T);
    niceTicks(sy.d[0], sy.d[1], 4).forEach(v => { el('line', { x1: L, x2: W - R, y1: sy(v), y2: sy(v), class: Math.abs(v) < 1e-9 ? 'v-zero' : 'v-grid' }, svg); el('text', { x: L - 6, y: sy(v) + 4, 'text-anchor': 'end', class: 'v-tick' }, svg, (v < 0 ? '-' : '') + usd0(Math.abs(v))); });
    niceTicks(0, N, Math.min(N, 8)).forEach(n => el('text', { x: sx(n), y: H - B + 15, 'text-anchor': 'middle', class: 'v-tick' }, svg, String(n)));
    el('text', { x: (L + W - R) / 2, y: H - 3, 'text-anchor': 'middle', class: 'v-axis-t' }, svg, o.xLabel || 'future similar runs');
    bets.forEach((b, i) => {
      const cls = 'bet-' + i, end = b.gain * N - b.price.mean;
      el('path', { d: `M${sx(0)},${sy(-b.price.lo)}L${sx(N)},${sy(b.gain * N - b.price.lo)}L${sx(N)},${sy(b.gain * N - b.price.hi)}L${sx(0)},${sy(-b.price.hi)}Z`, class: 'v-betband ' + cls }, svg);
      el('line', { x1: sx(0), y1: sy(-b.price.mean), x2: sx(N), y2: sy(end), class: 'v-betline ' + cls }, svg);
      if (num(b.payback) && b.payback <= N) {
        if (num(b.payback_lo) && num(b.payback_hi)) el('line', { x1: sx(b.payback_lo), x2: sx(Math.min(b.payback_hi, N)), y1: sy(0), y2: sy(0), class: 'v-payrange ' + cls }, svg);
        el('circle', { cx: sx(b.payback), cy: sy(0), r: 5, class: 'v-paypt ' + cls }, svg);
      }
      // the lower of two close line ends is labelled under its line
      const under = bets.some(b2 => b2 !== b && Math.abs(sy(b2.gain * N - b2.price.mean) - sy(end)) < 16 && b2.gain * N - b2.price.mean > end);
      el('text', { x: sx(N) - 4, y: sy(end) + (under ? 16 : -8), 'text-anchor': 'end', class: 'v-lbl' }, svg, b.name);
    });
    return svg;
  }

  // Inline bars: a chance (0 to 1) and money on a log scale; the mean as a tick, the 80% range as a light band.
  const chanceBar = (x, cls) => !x || !num(x.mean) ? '' : `<span class="v-bar ${cls || ''}" title="${esc(pct(x.mean) + (num(x.lo) ? ` (${pct(x.lo)} to ${pct(x.hi)})` : ''))}">` +
    (num(x.lo) && num(x.hi) ? `<i class="r" style="left:${(clamp01(x.lo) * 100).toFixed(1)}%;width:${Math.max(1, (clamp01(x.hi) - clamp01(x.lo)) * 100).toFixed(1)}%"></i>` : '') +
    `<i class="m" style="left:${(clamp01(x.mean) * 100).toFixed(1)}%"></i></span>`;
  const moneyBar = (x, d0, d1, cls) => {
    if (!x || !num(x.mean)) return '';
    const f = v => Math.max(0, Math.min(100, (Math.log(Math.max(v, d0)) - Math.log(d0)) / (Math.log(d1) - Math.log(d0)) * 100));
    return `<span class="v-bar ${cls || ''}">` + (num(x.lo) && num(x.hi) ? `<i class="r" style="left:${f(x.lo).toFixed(1)}%;width:${Math.max(1, f(x.hi) - f(x.lo)).toFixed(1)}%"></i>` : '') + `<i class="m" style="left:${f(x.mean).toFixed(1)}%"></i></span>`;
  };
  // One distribution as an inline SVG on a fixed domain, with run dots and the target line (for rows and cells).
  function miniDist(x, dom, o = {}) {
    const W = o.w || 180, H = o.h || 30, base = H - 8, sx = (o.log ? log : lin)(dom[0], dom[1], 3, W - 3);
    let s = `<svg class="v-mini" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" aria-hidden="true">`;
    if (num(o.target)) s += `<line x1="${sx(o.target).toFixed(1)}" x2="${sx(o.target).toFixed(1)}" y1="1" y2="${H - 1}" class="v-target"/>`;
    if (x && num(x.lo) && num(x.hi)) {
      s += `<path d="${shapePath(x, sx, base, H - 12)}" class="v-dist${o.cls ? ' ' + o.cls : ''}"/>`;
      s += `<line x1="${sx(x.lo).toFixed(1)}" x2="${sx(x.hi).toFixed(1)}" y1="${base}" y2="${base}" class="v-iv"/>`;
    }
    if (x && num(x.mean)) s += `<line x1="${sx(x.mean).toFixed(1)}" x2="${sx(x.mean).toFixed(1)}" y1="${base - 7}" y2="${base + 1}" class="v-mean"/>`;
    (o.dots || []).forEach(d => { if (num(d.v)) s += `<circle cx="${sx(d.v).toFixed(1)}" cy="${base}" r="3.4" class="v-dot${d.cls ? ' ' + d.cls : ''}"><title>${esc(d.t || '')}</title></circle>`; });
    return s + '</svg>';
  }

  return { el, h, esc, num, usd, usd0, pct, int, score, runs, range, clamp01, modelName, clip, lin, log, niceTicks, logTicks, logDomain,
    density, shapePath, hover, copyBtn, onResize, graph, miniGraph, frame, axisX, ridge, forest, scoreCost, payback, chanceBar, moneyBar, miniDist };
})();
