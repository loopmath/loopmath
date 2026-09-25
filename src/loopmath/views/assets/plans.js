// Plans view (lane 12): `loopmath recommend --html`. Renders `loopmath.view.plans/1` from the embedded object only.
(() => {
  const { esc, num, fmt, iv, ivPlain, moneyIv, arrow, ivBar, support, shared, tip, copy, shq, TAIL_NOTE, pulledUp, tailHtml, tailPlain } = LM;
  const D = LM.data(), app = document.getElementById('app');
  const rule = D.rule || {}, target = rule.score || null, isScore = !!target;
  const task = D.task || {}, fit = D.fit || {};
  const NS = 'ht' + 'tp:' + '/' + '/www.w3.org/2000/svg';
  const mk = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };

  // ------------------------------------------------------------ candidates by configuration id
  const BY = new Map();
  const cid = c => typeof c === 'string' ? c : c && c.config ? (typeof c.config === 'string' ? c.config : c.config.id) : null;
  const add = c => { const id = cid(c); if (id && c.prediction && !BY.has(id)) BY.set(id, c); else if (id && BY.has(id) && c.numbers && !BY.get(id).numbers) BY.get(id).numbers = c.numbers; };
  (D.candidates || []).forEach(add);
  (D.alternatives || []).forEach(add);
  // A pick is {candidate | config, gain_per_run, ...}, {paused, would_have_been}, {none} or {same_as} (spec 02, 05).
  const pickOf = p => !p ? null : p.would_have_been ? p.would_have_been : p.candidate || p.config ? p : null;
  const pickCid = p => p ? cid(p.candidate || p) : null;
  const pickFor = k => { const p = EX[k]; return p && p.same_as ? pickOf(EX[p.same_as]) : pickOf(p); };
  const EX = D.exploration || {};
  ['best_value', 'max_gain'].forEach(k => { const p = pickOf(EX[k]); if (p && p.candidate) add(p.candidate); });
  // The baseline the page compares against: the usual, else the reference (recommend/2: kind usual, best_recorded
  // or default). Without a usual the page never says "your usual".
  const REF = D.reference || null;
  const BASE = D.usual && D.usual.config ? D.usual : REF && REF.config ? REF : null;
  const isUsual = !!(D.usual && D.usual.config) || !!(REF && REF.kind === 'usual');
  const baseMark = isUsual ? 'usual' : 'reference';
  const baseWords = isUsual ? 'your usual' : 'the reference';
  const baseIntro = isUsual ? 'This is your usual workflow.' : REF && REF.kind === 'best_recorded' ? 'This is the reference: no usual workflow, so your best recorded workflow stands in for it.' : 'This is the reference: no usual workflow and no recorded one, so the default workflow stands in for it.';
  if (BASE && BASE.prediction) add({ config: BASE.config, origin: isUsual ? 'usual' : 'reference', diff_vs_usual: [], prediction: BASE.prediction, numbers: BASE.numbers });
  const usualId = BASE ? (BASE.config.id || BASE.config) : null;
  const usualPred = BASE && BASE.prediction;
  // I12: a piece of width n > 1 reads '3 x gpt-5.6-sol/xhigh', as views/common.py config_label (the recommender's labels omit it)
  const modelName = m => m && typeof m === 'object' ? m.id || m.raw : m;
  const widthLabel = cfg => {
    const ps = cfg && cfg.workflow && cfg.workflow.pieces, ss = (cfg && cfg.settings) || {};
    if (!Array.isArray(ps) || !ps.some(p => (p.width || 1) > 1)) return null;
    const parts = ps.map(p => [p, ss[p.id] || p.setting]).filter(([, s]) => s && typeof s === 'object')
      .map(([p, s]) => `${(p.width || 1) > 1 ? p.width + ' x ' : ''}${modelName(s.model)}/${s.effort || 'default'}`);
    return `${cfg.workflow.id || 'workflow'}: ${parts.join(', ')}`;
  };
  const labelOf = id => { const c = BY.get(id), cfg = c && c.config; const w = widthLabel(cfg); if (w) return w; if (cfg && cfg.label) return cfg.label; if (id === usualId && BASE.label) return BASE.label; return (D.graphs && D.graphs[id] && D.graphs[id].label) || id; };
  const numbersOf = id => (BY.get(id) || {}).numbers || null;

  // ------------------------------------------------------------ what "success" means here
  const op = target && target.better === 'lower' ? '<=' : '>=';
  const gWords = isScore ? `chance of reaching ${target.name} ${op} ${fmt.x(target.target)}` : 'chance of an accepted result';
  const scoreOf = p => isScore && p && p.scores ? p.scores[target.name] : null;
  // The rule's target carries no unit; the score predictions do.
  const unit = !isScore ? '' : target.unit || ([...BY.values()].map(c => scoreOf(c.prediction)).find(s => s && s.unit) || {}).unit || '';
  // g: p_reach from the score head for score rules (when the score head was used), else p_success. Its 80 percent
  // range (P3a) is `numbers.p_reach` (recommend/2) when given; else, with success_from score_head, p_success holds
  // the same draws of g (belief/state.py sets g to the score head's reach draws), so its range is the reach range.
  function g(p, nb) {
    if (!p) return null;
    const s = scoreOf(p), c = v => num(v) ? Math.max(0, Math.min(1, v)) : null, pr = nb && nb.p_reach;
    if (isScore && pr && num(pr.mean) && num(pr.lo) && num(pr.hi)) return { mean: c(pr.mean), lo: c(pr.lo), hi: c(pr.hi), from: 'score' };
    if (isScore && p.success_from === 'score_head' && s && num(s.p_reach)) {
      const ps = p.p_success || {}, same = num(ps.mean) && Math.abs(ps.mean - s.p_reach) < 1e-6 && num(ps.lo) && num(ps.hi);
      return { mean: s.p_reach, lo: same ? c(ps.lo) : null, hi: same ? c(ps.hi) : null, from: 'score' };
    }
    if (!p.p_success) return null;
    return { mean: c(p.p_success.mean), lo: c(p.p_success.lo), hi: c(p.p_success.hi), from: 'success' };
  }
  // I13: cost per accepted result = cost per run + P(fail) x the rescue; the expected rescue is the second term.
  const RESCUE = D.rescue || null;
  function rescueOf(p, nb) {
    if (nb && num(nb.expected_rescue_usd)) return { usd: nb.expected_rescue_usd, fail: null };
    const gg = g(p, nb);
    return RESCUE && num(RESCUE.usd) && gg && num(gg.mean) ? { usd: (1 - gg.mean) * RESCUE.usd, fail: 1 - gg.mean } : null;
  }
  const rescueCell = (p, nb) => { const r = rescueOf(p, nb), gg = g(p, nb); if (!r) return 'n/a'; const fail = num(r.fail) ? r.fail : gg && num(gg.mean) ? 1 - gg.mean : null; return `${esc(fmt.usd(r.usd))}${num(fail) && RESCUE && num(RESCUE.usd) ? `<span class="sub">${esc(fmt.pct(fail))} x ${esc(fmt.usd(RESCUE.usd))}</span>` : ''}`; };
  function rescueHtml() {
    if (!RESCUE) return '';
    if (RESCUE.kind === 'none' || !num(RESCUE.usd) || RESCUE.usd <= 0) return '<p class="note">No rescue is priced (rescue: none), so the cost per accepted result is the cost per run.</p>';
    const what = RESCUE.basis || RESCUE.kind || 'rescue';
    return `<p class="note">Cost per accepted result = cost per run + chance of failing x the rescue. The rescue: <b>${esc(what)}</b>${RESCUE.of ? ` (${esc(RESCUE.of)})` : ''}, about <b>${esc(fmt.usd(RESCUE.usd))}</b>${num(RESCUE.tokens) ? ` (${esc(fmt.tok(RESCUE.tokens))} tokens)` : ''}. The tables show the expected rescue, chance of failing x ${esc(fmt.usd(RESCUE.usd))}, in its own column.</p>`;
  }
  // I15: the median run cost beside the mean, when recommend/2 gives it.
  const medianOf = nb => nb && nb.run_cost_usd && num(nb.run_cost_usd.median) ? nb.run_cost_usd : null;
  const medianText = nb => { const m = medianOf(nb); return m ? `median ${fmt.usd(m.median)}${m.median_basis && m.median_basis !== 'draws' ? ' (approx.)' : ''}` : ''; };
  const gText = x => !x || !num(x.mean) ? 'n/a' : (num(x.lo) ? `${fmt.pct(x.mean)} <span class="rng-t">(${fmt.pct(x.lo)} to ${fmt.pct(x.hi)})</span>` : fmt.pct(x.mean)) + tailHtml(x);
  const isShared = p => !!p && (p.support === 0 || (fit.n_runs && fit.n_runs.user === 0));

  // ------------------------------------------------------------ marked points
  const marks = new Map();  // config id -> [labels]
  const mark = (id, label) => { if (!id) return; if (!marks.has(id)) marks.set(id, []); if (!marks.get(id).includes(label)) marks.get(id).push(label); };
  mark(usualId, baseMark);
  mark(D.default_pick && D.default_pick.config, 'default pick');
  mark(D.goal && D.goal.config, `goal${D.goal && num(D.goal.level) ? ' (' + D.goal.level + '%)' : ''}`);
  const bv = EX.best_value, mg = EX.max_gain;
  if (pickOf(bv)) mark(pickCid(pickOf(bv)), (mg && mg.same_as ? 'best value and biggest gain' : 'best value') + (bv.paused ? ' (paused)' : ''));
  if (pickOf(mg)) mark(pickCid(pickOf(mg)), 'biggest gain' + (mg.paused ? ' (paused)' : ''));
  const pickKinds = id => ['best_value', 'max_gain'].filter(k => { const p = pickFor(k); return p && pickCid(p) === id; });

  let selected = null, yMode = 'g';

  // ------------------------------------------------------------ top block
  function topHtml() {
    const chain = (task.group_chain || []).map(([level, id]) => {
      const n = task.support ? task.support[level] : null;
      return `<span class="chip">${esc(level)} <b>${esc(id)}</b> ${support(n)}</span>`;
    }).join('');
    const feats = Object.entries(task.features || {}).map(([k, v]) => `<span class="chip">${esc(k)} ${esc(v)}</span>`).join('');
    const ruleText = isScore
      ? `score target: <b>${esc(target.name)} ${esc(op)} ${esc(fmt.x(target.target))}</b>${arrow(target.better)}${rule.requires && rule.requires.length ? ', and ' + rule.requires.map(esc).join(', ') + ' pass' : ''}`
      : `binary: <b>${(rule.requires || []).map(esc).join(' and ') || 'no verdicts required'}</b> must pass`;
    const nr = fit.n_runs || {};
    return `<div class="grid2">` +
      `<div><h3>Task</h3><p class="note"><b>${esc(task.title || '(untitled task)')}</b></p><dl class="kv"><dt>type</dt><dd>${esc([task.type, task.subtype].filter(Boolean).join(' / ') || 'n/a')}</dd><dt>repo</dt><dd>${esc(task.repo || 'n/a')}</dd>${task.base_commit ? `<dt>base commit</dt><dd class="mono">${esc(task.base_commit)}</dd>` : ''}</dl>${feats ? `<div class="chips" style="margin-top:6px">${feats}</div>` : ''}</div>` +
      `<div><h3>Acceptance rule</h3><p class="note">${esc(rule.name || 'default')}: ${esc(rule.definition || '')}</p><p class="note">${ruleText}</p><p class="small m">Late ${esc((rule.excludes_events || []).join(' or ') || 'events')} within ${esc(rule.window_days || 14)} days turn success into failure.</p></div>` +
      `<div><h3>Fit</h3><dl class="kv"><dt>fit</dt><dd class="mono">${esc(fit.id || 'n/a')}</dd><dt>age</dt><dd>${esc(fmt.age(fit.age_s))}${num(fit.age_s) && fit.age_s > 86400 ? ' <span class="warn">(older than a day)</span>' : ''}</dd><dt>runs</dt><dd>${fmt.int(nr.user)} of yours, ${fmt.int(nr.prior)} shared${nr.user === 0 ? ' ' + shared(true) : ''}</dd></dl><h3>Support by level</h3><div class="chips">${chain || '<span class="m">n/a</span>'}</div></div>` +
      `</div>`;
  }

  // ------------------------------------------------------------ chart
  function points() {
    const out = [];
    BY.forEach((c, id) => {
      const p = c.prediction, cost = p && p.cost && p.cost.usd, gg = g(p, c.numbers), s = scoreOf(p);
      if (!cost || !num(cost.mean) || cost.mean <= 0) return;
      const y = yMode === 'score' ? (s && s.value ? s.value : null) : gg;
      if (!y || !num(y.mean)) return;
      out.push({ id, c, x: cost.mean, xlo: num(cost.lo) && cost.lo > 0 ? cost.lo : null, xhi: num(cost.hi) ? cost.hi : null, y: y.mean, ylo: y.lo, yhi: y.hi });
    });
    return out;
  }
  function logTicks(a, b) {
    const out = [];
    for (let k = Math.floor(Math.log10(a)); k <= Math.ceil(Math.log10(b)); k++) [1, 2, 5].forEach(m => { const v = m * Math.pow(10, k); if (v >= a && v <= b) out.push(v); });
    return out;
  }
  function drawChart() {
    const host = document.getElementById('chart');
    host.innerHTML = '';
    const pts = points();
    if (!pts.length) { host.innerHTML = '<p class="empty">No candidate has both a cost and a success estimate to plot.</p>'; return; }
    const W = Math.max(340, Math.min(1100, host.clientWidth || 800)), H = W < 600 ? 320 : 400, ml = 58, mr = 18, mt = 18, mb = 46;
    const xs = pts.flatMap(p => [p.x, p.xlo, p.xhi]).filter(v => num(v) && v > 0);
    const x0 = Math.min(...xs) / 1.3, x1 = Math.max(...xs) * 1.3;
    let y0 = 0, y1 = 1;
    if (yMode === 'score') {
      const ys = pts.flatMap(p => [p.y, p.ylo, p.yhi]).concat(num(target.target) ? [target.target] : []).filter(num);
      const pad = (Math.max(...ys) - Math.min(...ys)) * 0.08 || 1;
      y0 = Math.min(...ys) - pad; y1 = Math.max(...ys) + pad;
    }
    const X = v => ml + (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) * (W - ml - mr);
    const Y = v => mt + (1 - (v - y0) / (y1 - y0)) * (H - mt - mb);
    const svg = mk('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: 'pl-svg', role: 'img' }, host);
    const grid = mk('g', {}, svg);
    logTicks(x0, x1).forEach(v => { mk('line', { x1: X(v), x2: X(v), y1: mt, y2: H - mb, class: 'pl-grid' }, grid); mk('text', { x: X(v), y: H - mb + 16, class: 'pl-tick', 'text-anchor': 'middle' }, grid).textContent = fmt.usd(v); });
    const yt = yMode === 'score' ? niceTicks(y0, y1) : [0, 0.2, 0.4, 0.6, 0.8, 1];
    yt.forEach(v => { mk('line', { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), class: 'pl-grid' }, grid); mk('text', { x: ml - 6, y: Y(v) + 4, class: 'pl-tick', 'text-anchor': 'end' }, grid).textContent = yMode === 'score' ? fmt.score(v) : fmt.pct(v); });
    mk('text', { x: (ml + W - mr) / 2, y: H - 8, class: 'pl-axis', 'text-anchor': 'middle' }, svg).textContent = 'expected cost per run, dollars (log scale)';
    mk('text', { x: 14, y: (mt + H - mb) / 2, class: 'pl-axis', 'text-anchor': 'middle', transform: `rotate(-90 14 ${(mt + H - mb) / 2})` }, svg).textContent = yMode === 'score' ? `expected ${target.name}${unit ? ' (' + unit + ')' : ''}` : gWords;
    // Reference: the goal level (success) or the score target.
    const ref = yMode === 'score' ? target.target : D.goal && num(D.goal.level) ? D.goal.level / 100 : null;
    if (num(ref) && ref >= y0 && ref <= y1) { mk('line', { x1: ml, x2: W - mr, y1: Y(ref), y2: Y(ref), class: 'pl-ref' }, svg); mk('text', { x: W - mr - 4, y: Y(ref) - 4, class: 'pl-reft', 'text-anchor': 'end' }, svg).textContent = yMode === 'score' ? `target ${fmt.score(ref, unit)}` : `goal ${fmt.pct(ref)}`; }
    // The curve as a step line through its rows; a segment into an uncertain row is dashed.
    const rows = (D.curve || []).filter(r => r.reached && r.config && BY.has(r.config)).map(r => ({ r, p: pts.find(q => q.id === r.config) })).filter(x => x.p).sort((a, b) => Math.min(...a.r.levels) - Math.min(...b.r.levels));
    const curveG = mk('g', {}, svg);
    rows.forEach((x, i) => {
      if (i) { const a = rows[i - 1].p, b = x.p; mk('path', { d: `M${X(a.x)},${Y(a.y)}H${X(b.x)}V${Y(b.y)}`, class: 'pl-curve' + (x.r.uncertain ? ' unc' : '') }, curveG); }
    });
    const crossG = mk('g', {}, svg), dotG = mk('g', {}, svg), labG = mk('g', {}, svg);
    pts.sort((a, b) => (marks.has(a.id) ? 1 : 0) - (marks.has(b.id) ? 1 : 0));
    pts.forEach(p => {
      const marked = marks.has(p.id), cls = marked ? ' mk' : '';
      if (num(p.xlo) && num(p.xhi)) mk('line', { x1: X(p.xlo), x2: X(p.xhi), y1: Y(p.y), y2: Y(p.y), class: 'pl-cross' + cls }, crossG);
      if (num(p.ylo) && num(p.yhi)) mk('line', { x1: X(p.x), x2: X(p.x), y1: Y(p.ylo), y2: Y(p.yhi), class: 'pl-cross' + cls }, crossG);
      const dot = mk('circle', { cx: X(p.x), cy: Y(p.y), r: marked ? 6.5 : 4, class: 'pl-dot' + cls + (p.id === selected ? ' sel' : ''), 'data-cfg': p.id }, dotG);
      if (marked) {
        dot.setAttribute('data-mark', marks.get(p.id).join(', '));
      }
    });
    // Labels go to the first of four spots around the dot that overlaps no earlier label.
    const boxes = [];
    const place = (text, px, py, cls, px6, attrs) => {
      const w = text.length * px6 + 4, h = 13;
      const spots = [[10, -8, 'start'], [10, 16, 'start'], [-10, -8, 'end'], [-10, 16, 'end'], [10, -22, 'start'], [-10, 30, 'end']];
      let best = spots[0];
      for (const sp of spots) {
        const x0 = sp[2] === 'start' ? px + sp[0] : px + sp[0] - w, y0 = py + sp[1] - 11;
        if (x0 < ml || x0 + w > W - 2) continue;
        if (!boxes.some(b => x0 < b[0] + b[2] && b[0] < x0 + w && y0 < b[1] + b[3] && b[1] < y0 + h)) { best = sp; break; }
      }
      const x0 = best[2] === 'start' ? px + best[0] : px + best[0] - w;
      boxes.push([x0, py + best[1] - 11, w, h]);
      mk('text', Object.assign({ x: px + best[0], y: py + best[1], class: cls, 'text-anchor': best[2] }, attrs), labG).textContent = text;
    };
    pts.filter(p => marks.has(p.id)).forEach(p => place(marks.get(p.id).join(', '), X(p.x), Y(p.y), 'pl-label', 6.6, { 'data-cfg': p.id }));
    rows.forEach(x => place(x.r.levels.map(l => l + '%').join(', ') + (x.r.uncertain ? ' (uncertain)' : ''), X(x.p.x), Y(x.p.y), 'pl-level', 5.6, {}));
    svg.addEventListener('mousemove', ev => {
      const el = ev.target.closest && ev.target.closest('[data-cfg]');
      if (!el) { tip(null); return; }
      const c = BY.get(el.dataset.cfg), p = c.prediction, nb = c.numbers, med = medianText(nb), r = rescueOf(p, nb);
      tip(`<b>${esc(labelOf(el.dataset.cfg))}</b>${marks.has(el.dataset.cfg) ? '<br>' + esc(marks.get(el.dataset.cfg).join(', ')) : ''}<br>${esc(gWords)}: ${gText(g(p, nb))}<br>cost ${p.cost.usd ? esc(`${fmt.usd(p.cost.usd.mean)} (${fmt.usd(p.cost.usd.lo)} to ${fmt.usd(p.cost.usd.hi)})${med ? ', ' + med : ''}`) : 'n/a'}, ${fmt.tok(p.cost.tokens && p.cost.tokens.mean)} tokens${tailPlain(p.cost.usd, p.cost.tokens)}${r ? `<br>expected rescue ${esc(fmt.usd(r.usd))}` : ''}<br><span class="m">cost per accepted result ${fmt.usd(p.ell && p.ell.usd && p.ell.usd.mean)}${tailPlain(p.ell && p.ell.usd)}; click for detail</span>`, ev.clientX, ev.clientY);
    });
    svg.addEventListener('mouseleave', () => tip(null));
    svg.addEventListener('click', ev => { const el = ev.target.closest && ev.target.closest('[data-cfg]'); if (el) select(el.dataset.cfg, true); });
  }
  function niceTicks(a, b) {
    const span = b - a, step0 = Math.pow(10, Math.floor(Math.log10(span / 5))), step = [1, 2, 5, 10].map(m => m * step0).find(s => span / s <= 6) || step0 * 10;
    const out = []; for (let v = Math.ceil(a / step) * step; v <= b; v += step) out.push(+v.toFixed(10)); return out;
  }

  // ------------------------------------------------------------ curve table
  const levelsText = r => (r.levels || []).map(l => l + '%').join(', ');
  function curveHtml() {
    const rows = (D.curve || []).map(r => {
      const goal = D.goal && r.config && r.config === D.goal.config && (!num(D.goal.level) || r.levels.includes(D.goal.level));
      if (!r.reached || !r.config) return `<tr class="${goal ? 'hl' : ''}"><td>${esc(levelsText(r))}</td><td colspan="${isScore ? 7 : 6}" class="m">No candidate reaches this level${r.uncertain ? '; uncertain' : ''}.</td></tr>`;
      const p = r.prediction || (BY.get(r.config) || {}).prediction || {}, s = scoreOf(p), nb = r.numbers || numbersOf(r.config);
      return `<tr class="row${goal ? ' hl' : ''}" data-cfg="${esc(r.config)}"><td>${esc(levelsText(r))}${goal ? ' <span class="badge mark">goal</span>' : ''}${r.uncertain ? ' <span class="badge unk" title="fewer than 80 percent of draws reach this level">uncertain</span>' : ''}</td>` +
        `<td>${esc(labelOf(r.config))}</td><td>${gStack(g(p, nb))} ${support(p.support)}${shared(isShared(p))}</td><td class="nw">${costStack(p.cost, nb)}</td><td class="nw">${rescueCell(p, nb)}</td><td class="nw">${p.ell ? stack(p.ell.usd, fmt.usd) : 'n/a'}</td>` +
        (isScore ? `<td class="nw">${s && s.value ? stack(s.value, v => fmt.score(v, s.unit)) : 'n/a'}</td>` : '') + `</tr>`;
    }).join('');
    return `<div class="tablescroll"><table><thead><tr><th>level</th><th>workflow</th><th>${esc(gWords)}</th><th>cost per run</th>${RESCUE_TH}<th title="${ELL_TITLE}">cost per accepted result</th>${isScore ? `<th>expected ${esc(target.name)}</th>` : ''}</tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  const ELL_TITLE = 'ell: expected dollars to an accepted result, a rescue included when a run fails: cost per run + expected rescue';
  const RESCUE_TH = '<th title="chance of failing x the rescue named at the top">expected rescue</th>';
  function altsHtml() {
    const alts = D.alternatives || [];
    if (!alts.length) return '<p class="empty">No alternatives.</p>';
    return `<div class="tablescroll"><table><thead><tr><th>workflow</th><th>change from ${baseWords}</th><th>${esc(gWords)}</th><th>cost per run</th>${RESCUE_TH}<th title="${ELL_TITLE}">cost per accepted result</th></tr></thead><tbody>` + alts.map(c => {
      const id = cid(c), p = c.prediction || {}, nb = c.numbers || numbersOf(id);
      return `<tr class="row" data-cfg="${esc(id)}"><td>${esc(labelOf(id))}${marks.has(id) ? ' <span class="badge mark">' + esc(marks.get(id).join(', ')) + '</span>' : ''}</td><td class="small">${(c.diff_vs_usual || []).map(esc).join('<br>') || `<span class="m">same as ${baseWords}</span>`}</td><td>${gText(g(p, nb))}${delta(p, 'g')}</td><td class="nw">${costStack(p.cost, nb)}${delta(p, 'cost')}</td><td class="nw">${rescueCell(p, nb)}</td><td class="nw">${p.ell ? fmt.usd(p.ell.usd.mean) + tailHtml(p.ell.usd) : 'n/a'}</td></tr>`;
    }).join('') + '</tbody></table></div>';
  }
  function deltaText(p, what) {
    if (!usualPred || !p) return '';
    if (what === 'g') { const a = g(p), b = g(usualPred); return a && b && num(a.mean) && num(b.mean) ? fmt.pp((a.mean - b.mean) * 100) : ''; }
    const a = p.cost && p.cost.usd && p.cost.usd.mean, b = usualPred.cost && usualPred.cost.usd && usualPred.cost.usd.mean;
    return num(a) && num(b) && b > 0 ? `${fmt.signed((a / b - 1) * 100, v => v.toFixed(0) + '%')} (${fmt.signed(a - b, fmt.usd)})` : '';
  }
  const delta = (p, what) => { const t = deltaText(p, what); return t ? `<span class="sub">${esc(t)} vs ${baseMark}</span>` : ''; };
  // A value with its 80 percent range on a second, muted line: keeps table columns narrow.
  // `also`: other values shown in the cell (the tokens beside dollars), whose tail the note covers too
  const stack = (x, f, extra = '', also = []) => !x || !num(x.mean) ? 'n/a' : `${esc(f(x.mean))}<span class="sub">${num(x.lo) ? esc(f(x.lo) + ' to ' + f(x.hi)) : ''}${esc(extra)}</span>${[x, ...also].some(pulledUp) ? `<span class="sub tail">${TAIL_NOTE}</span>` : ''}`;
  const costStack = (m, nb) => !m || !m.usd ? 'n/a' : stack(m.usd, fmt.usd, (medianOf(nb) ? `, ${medianText(nb)}` : '') + (m.tokens && num(m.tokens.mean) ? `, ${fmt.tok(m.tokens.mean)} tok` : ''), [m.tokens]);
  const gStack = x => !x || !num(x.mean) ? 'n/a' : num(x.lo) ? stack(x, fmt.pct) : fmt.pct(x.mean);

  // ------------------------------------------------------------ exploration
  const NOTE = { best_value: 'Best value optimizes gain per dollar spent now: the lowest payback.', max_gain: 'Biggest gain optimizes gain per future run, whatever the price (within the budget cap).' };
  const TITLE = { best_value: 'Best value', max_gain: 'Biggest gain' };
  // D96/D102: lane 5's look-ahead value in dollars, an expectation over every outcome of the trial run
  // (not a gain on the condition that it beats the goal); its cost, success and score parts belong to
  // the best workflow after learning, so they stay in the JSON. Worded as lane 6's recommend message.
  function savingText(gp) {
    if (!gp || !num(gp.usd)) return 'n/a';
    const v = gp.usd, dollars = v > 0 && v < 0.01 ? 'under $0.01' : '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return `expected to save about ${dollars} per future similar run`;
  }
  // whole runs, at least one, as in the recommendation message (lane 6)
  function paybackText(runs) {
    if (!num(runs)) return 'n/a';
    const n = Math.max(1, Math.round(runs));
    return `about ${n} similar run${n === 1 ? '' : 's'}`;
  }
  function pickHtml(kind, p, paused) {
    const id = pickCid(p), c = BY.get(id) || p.candidate || {}, gg = g(c.prediction, c.numbers || numbersOf(id));
    return `<p class="note"><a href="#" data-pick="${esc(id)}">${esc(labelOf(id))}</a>${p.auto_ok ? ' <span class="badge acc" title="payback is below explore.auto_payback_runs">auto ok</span>' : ''}</p>` +
      `<dl class="kv${paused ? ' m' : ''}"><dt>${esc(gWords)}</dt><dd>${gText(gg)}</dd>` +
      `<dt>chance it beats the recommended pick</dt><dd>${fmt.pct(p.p_beats_goal)}</dd>` +
      `<dd class="small m wide">Not the ${esc(gWords)}: the chance that, once tried, this workflow turns out cheaper per accepted result than the recommended pick (marked goal).</dd>` +
      `<dt>price now</dt><dd>${moneyIv(p.price)}</dd>` +
      `<dt>trying it once</dt><dd>${savingText(p.gain_per_run)}</dd>` +
      `<dt>pays for itself after</dt><dd>${paybackText(p.payback_runs)}</dd>` +
      ((p.runner_ups || []).length ? `<dt>runner-ups</dt><dd>${p.runner_ups.map(r => `<a href="#" data-pick="${esc(cid(r))}">${esc(labelOf(cid(r)))}</a>`).join('<br>')}</dd>` : '') + `</dl>`;
  }
  function exploreHtml() {
    const card = kind => {
      const p = EX[kind];
      let body;
      if (!p) body = '<p class="empty">Not reported.</p>';
      else if (p.same_as) body = `<p class="note">Same candidate as ${esc(TITLE[p.same_as] || p.same_as).toLowerCase()}: one run serves both.</p>`;
      else if (p.paused) body = `<p class="warn">Paused: ${esc(p.paused)}. It would have been:</p>` + (p.would_have_been ? pickHtml(kind, p.would_have_been, true) : '');
      else if (p.none != null || !pickOf(p)) body = `<p class="note">${esc(typeof p.none === 'string' ? p.none : 'No candidate qualifies: none gains more than 1 percent of the goal\'s expected dollars.')}</p>`;
      else body = pickHtml(kind, p, false);
      return `<div class="panel pick"><h3>${TITLE[kind]}</h3><p class="small m">${NOTE[kind]}</p>${body}</div>`;
    };
    const pair = D.pair;
    const pairText = pair && pair.members ? `<p class="note">Suggested pair: ${pair.members.map(m => `<a href="#" data-pick="${esc(m)}">${esc(labelOf(m))}</a>`).join(' next to ')}${pair.explore_pick ? ' <span class="m">(' + esc(TITLE[pair.explore_pick] || pair.explore_pick).toLowerCase() + ')</span>' : ''}.</p>${(pair.instructions || []).length ? '<ul class="small">' + pair.instructions.map(s => `<li>${esc(s)}</li>`).join('') + '</ul>' : ''}` : '';
    return `<div class="grid2">${card('best_value')}${card('max_gain')}</div>${pairText}`;
  }

  // ------------------------------------------------------------ side panel
  const graphFor = id => (D.graphs && D.graphs[id]) || null;
  function commandFor(id) {
    const kinds = pickKinds(id), source = id === usualId && isUsual ? 'usual' : kinds.length ? 'exploration' : 'alternative';
    const parts = ['loopmath run start'];
    if (task.type) parts.push('--type', shq(task.type));
    if (task.repo) parts.push('--repo', shq(task.repo));
    if (task.subtype) parts.push('--subtype', shq(task.subtype));
    if (task.title) parts.push('--title', shq(task.title));
    Object.entries(task.features || {}).forEach(([k, v]) => parts.push('--feature', shq(`${k}=${v}`)));
    if (task.base_commit) parts.push('--base-commit', shq(task.base_commit));
    parts.push('--config', shq(id), '--source', source);
    if (D.rec) parts.push('--rec', shq(D.rec));
    return parts.join(' ');
  }
  function panelHtml(id) {
    const c = BY.get(id), p = c.prediction || {}, s = scoreOf(p), cmd = commandFor(id);
    let h = `<h2><span>${esc(labelOf(id))}</span><button data-close>close</button></h2>`;
    h += `<p class="small m mono">${esc(id)}${c.origin ? ' <span class="badge src">' + esc(c.origin) + '</span>' : ''}${marks.has(id) ? ' ' + marks.get(id).map(m => `<span class="badge mark">${esc(m)}</span>`).join(' ') : ''}</p>`;
    const nb = c.numbers || null, gg = g(p, nb);
    h += `<p class="note">This configuration has a ${gg && num(gg.mean) ? fmt.pct(gg.mean) : 'unknown'} ${esc(gWords)} at about ${fmt.usd(p.cost && p.cost.usd && p.cost.usd.mean)} (${fmt.tok(p.cost && p.cost.tokens && p.cost.tokens.mean)} tokens) per run${tailPlain(gg, p.cost && p.cost.usd, p.cost && p.cost.tokens)}.</p>`;
    h += `<div class="cmd"><code>${esc(cmd)}</code><button data-copy="${esc(cmd)}">copy command</button></div>`;
    h += `<h3>Plan</h3><div id="p-graph" class="lmg"></div>`;
    h += `<h3>Change from ${baseWords}</h3>${id === usualId ? `<p class="note">${esc(baseIntro)}</p>` : (c.diff_vs_usual || []).length ? '<ul class="small">' + c.diff_vs_usual.map(l => `<li>${esc(l)}</li>`).join('') + '</ul>' : '<p class="m small">No lines recorded.</p>'}`;
    const rows = [];
    const row = (label, x, f, extra) => rows.push(`<tr><td>${label}</td><td>${x ? stack(x, f) : 'n/a'}${extra || ''}</td><td>${x ? ivBar(x, { min: f === pctF ? 0 : undefined, max: f === pctF ? 1 : undefined }) : ''}</td></tr>`);
    const pctF = v => fmt.pct(v);
    if (gg && num(gg.lo)) row(esc(gWords), gg, pctF, ' ' + support(p.support) + shared(isShared(p)));
    else rows.push(`<tr><td>${esc(gWords)}</td><td>${gStack(gg)} ${support(p.support)}${shared(isShared(p))}</td><td></td></tr>`);
    if (isScore && gg && gg.from === 'success') rows.push(`<tr><td colspan="3" class="warn small">Fewer than 5 runs of this type carry ${esc(target.name)}, so this chance comes from the success head, not the score.</td></tr>`);
    if (p.cost) { row('cost per run, dollars', p.cost.usd, fmt.usd, medianOf(nb) ? ` <span class="m">${esc(medianText(nb))}</span>` : ''); row('cost per run, tokens', p.cost.tokens, fmt.tok); }
    const resc = rescueOf(p, nb);
    if (resc) rows.push(`<tr><td>expected rescue</td><td>${rescueCell(p, nb)}</td><td></td></tr>`);
    if (p.ell) row('cost per accepted result (ell, a rescue included)', p.ell.usd, fmt.usd);
    if (p.rounds) row('rounds', p.rounds, fmt.rounds);
    Object.values(p.scores || {}).forEach(sc => { if (sc && sc.value) row(`${esc(sc.name)}${arrow(sc.better)}`, sc.value, v => fmt.score(v, sc.unit), ` ${num(sc.p_reach) ? '<span class="m">reach ' + fmt.pct(sc.p_reach) + '</span> ' : ''}${support(sc.support)}`); });
    h += `<h3>Prediction (80 percent ranges)</h3><table class="small"><tbody>${rows.join('')}</tbody></table>`;
    if (id !== usualId && usualPred) h += `<p class="note small">Against ${baseWords}: ${esc(deltaText(p, 'g') || 'n/a')} success, ${esc(deltaText(p, 'cost') || 'n/a')} cost per run.</p>`;
    pickKinds(id).forEach(k => { const pk = pickFor(k), paused = !!(EX[k].paused || (EX[k].same_as && EX[EX[k].same_as].paused)); h += `<h3>${TITLE[k]}${paused ? ' (paused)' : ''}</h3><p class="small m">${NOTE[k]}</p>` + pickHtml(k, pk, paused); });
    ['best_value', 'max_gain'].forEach(k => { const pk = pickOf(EX[k]); if (pk && (pk.runner_ups || []).some(r => cid(r) === id)) h += `<p class="note small">Runner-up for ${TITLE[k].toLowerCase()}.</p>`; });
    return h;
  }
  function select(id, scroll) {
    if (!id || !BY.has(id)) return;
    selected = id;
    const panel = document.getElementById('panel');
    panel.hidden = false;
    panel.innerHTML = panelHtml(id);
    const gr = graphFor(id);
    if (!gr) document.getElementById('p-graph').innerHTML = '<p class="m small">No plan graph was embedded for this configuration.</p>';
    else try {
      const host = document.getElementById('p-graph');
      LM.Graph.render(host, gr, { noTitle: true });
      const svg = host.querySelector('svg'), w = svg && +svg.getAttribute('width');
      if (w && host.clientWidth && w > host.clientWidth) { svg.style.width = Math.max(host.clientWidth, w * 0.7) + 'px'; svg.style.height = 'auto'; }
    } catch (e) { document.getElementById('p-graph').innerHTML = `<p class="warn">The plan graph could not be drawn: ${esc(e.message)}</p>`; }
    drawChart();  // the panel narrows the chart; this also marks the selected dot
    document.querySelectorAll('tr.row').forEach(el => el.classList.toggle('sel', el.dataset.cfg === id));
    if (history.replaceState) history.replaceState(null, '', '#cfg=' + encodeURIComponent(id));
    if (scroll && window.innerWidth < 1100 && panel.scrollIntoView) panel.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }

  // ------------------------------------------------------------ page
  const markButtons = [...marks.entries()].filter(([id]) => BY.has(id)).map(([id, ls]) => `<button data-pick="${esc(id)}">${esc(ls.join(', '))}</button>`).join(' ');
  app.innerHTML = `<header class="top"><h1><span class="k">loopmath</span> plans</h1><p class="lede">${esc(D.message || 'No recommendation message.')}</p>${rescueHtml()}</header>` +
    `<section class="panel">${topHtml()}</section>` +
    `<div class="pl-layout"><div class="pl-main">` +
    `<section class="panel"><h2><span>Success against cost</span><span class="small">${isScore ? `<button data-y="g" class="on">chance to reach</button> <button data-y="score">expected ${esc(target.name)}</button>` : ''}</span></h2>` +
    `<p class="small m">Each dot is one candidate (${BY.size}); the cross is its 80 percent range. The line steps through the success-cost curve; dashed steps are uncertain. Marked: ${markButtons || 'none'}</p><div id="chart" class="pl-chart"></div></section>` +
    `<section class="panel"><h2>Success-cost curve</h2><p class="small m">For each level, the cheapest configuration whose ${esc(gWords)} is at least that level.</p>${curveHtml()}</section>` +
    `<section class="panel"><h2>Exploration</h2>${exploreHtml()}</section>` +
    `<section class="panel"><h2>Alternatives</h2>${altsHtml()}</section>` +
    `</div><aside class="panel pl-panel" id="panel" hidden></aside></div>` +
    `<p class="foot">Generated ${esc(fmt.dt(D.generated_at))}${D.rec ? ', recommendation <span class="mono">' + esc(D.rec) + '</span>' : ''}. Ranges are 80 percent. Commands are for copying; this page runs nothing.</p>`;
  app.addEventListener('click', ev => {
    const pick = ev.target.closest('[data-pick]');
    if (pick) { ev.preventDefault(); select(pick.dataset.pick, true); return; }
    const row = ev.target.closest('tr.row[data-cfg]');
    if (row) { select(row.dataset.cfg, true); return; }
    const b = ev.target.closest('[data-copy]');
    if (b) { copy(b.dataset.copy, b); return; }
    if (ev.target.closest('[data-close]')) { const p = document.getElementById('panel'); p.hidden = true; selected = null; document.querySelectorAll('.sel').forEach(el => el.classList.remove('sel')); return; }
    const y = ev.target.closest('[data-y]');
    if (y) { yMode = y.dataset.y; document.querySelectorAll('[data-y]').forEach(el => el.classList.toggle('on', el === y)); drawChart(); }
  });
  drawChart();
  let resizeTimer = null;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawChart, 150); });
  const hash = /^#cfg=(.+)$/.exec(location.hash || '');
  if (hash && BY.has(decodeURIComponent(hash[1]))) select(decodeURIComponent(hash[1]));
})();
