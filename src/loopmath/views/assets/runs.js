// Runs view (lane 12): `loopmath runs --html`. Renders `loopmath.view.runs/1` from the embedded object only.
(() => {
  const { esc, num, fmt, money, iv, ivBar, support, tier, zBadge, uniq, options, copy, shq, arrow } = LM;
  const D = LM.data(), RUNS = Array.isArray(D.runs) ? D.runs : [], DETAILS = D.details || {};
  const byRun = new Map(RUNS.map(r => [r.run, r]));
  const F = { type: '', repo: '', source: '', from: '', to: '', slate: '', outcome: '' };
  Object.keys(F).forEach(k => { if (D.filters && D.filters[k] != null && typeof D.filters[k] !== 'object') F[k] = String(D.filters[k]); });
  // `--since` was applied by the builder (it may be relative, like 90d); the page does not apply it again.
  let selected = D.selected && D.selected.run ? D.selected.run : null;
  const app = document.getElementById('app');

  const outcomeKey = r => r.z === 1 ? 'accepted' : r.z === 0 ? 'not accepted' : 'unknown';
  const day = r => String(r.started_at || '').slice(0, 10);
  const task = r => r.task || {};
  const cfg = r => r.config || {};
  const rc = r => r.receipt || {};
  function pass(r) {
    const t = task(r);
    if (F.type && t.type !== F.type) return false;
    if (F.repo && t.repo !== F.repo) return false;
    if (F.source && r.source !== F.source) return false;
    if (F.from && day(r) < F.from) return false;
    if (F.to && day(r) > F.to) return false;
    if (F.slate === '(any)' && !r.slate) return false;
    if (F.slate === '(none)' && r.slate) return false;
    if (F.slate && F.slate[0] !== '(' && r.slate !== F.slate) return false;
    if (F.outcome && outcomeKey(r) !== F.outcome) return false;
    return true;
  }

  function totals(rows) {
    const usd = rows.reduce((s, r) => s + (r.cost && num(r.cost.usd) ? r.cost.usd : 0), 0);
    const tok = rows.reduce((s, r) => s + (r.cost && num(r.cost.tokens) ? r.cost.tokens : 0), 0);
    const known = rows.filter(r => r.z === 0 || r.z === 1), acc = known.filter(r => r.z === 1).length;
    const withRc = rows.filter(r => rc(r).predicted), inside = withRc.filter(r => rc(r).cost_in_interval === true).length;
    const days = rows.map(day).filter(Boolean).sort();
    return { n: rows.length, usd, tok, known: known.length, acc, withRc: withRc.length, inside, first: days[0], last: days[days.length - 1] };
  }

  // filters the builder already applied (`loopmath runs --type T --since 90d ...`), as flags
  const cliFilters = () => ['type', 'repo', 'since', 'slate'].filter(k => D.filters && D.filters[k] != null && D.filters[k] !== '' && typeof D.filters[k] !== 'object').map(k => `--${k} ${D.filters[k]}`);

  function lede(t) {
    if (!RUNS.length && cliFilters().length) return `No runs match ${esc(cliFilters().join(', '))} from the command line. Run <code>loopmath runs --html</code> without them to see every recorded run.`;
    if (!RUNS.length) return 'No runs recorded yet. Runs appear here after <code>loopmath run start</code> and <code>loopmath run finish</code>, or after <code>loopmath run import FILE.ocp.json</code>.';
    if (!t.n) return `None of the ${RUNS.length} recorded runs match these filters.`;
    let s = `${t.n} run${t.n === 1 ? '' : 's'}${t.first ? ` from ${esc(t.first)}${t.last !== t.first ? ' to ' + esc(t.last) : ''}` : ''} cost ${fmt.usd(t.usd)} (${fmt.tok(t.tok)} tokens).`;
    s += t.known ? ` ${fmt.pct(t.acc / t.known)} of the ${t.known} with a known outcome met their acceptance rule.` : ' None has a known outcome yet.';
    s += t.withRc ? ` ${t.withRc} had a prediction on record; cost landed inside its 80 percent range for ${t.inside} of them.` : ' None had a prediction on record.';
    return s;
  }

  // Newest first; slate members sit together at the position of their newest member.
  function grouped(rows) {
    const sorted = rows.slice().sort((a, b) => LM.byTime(b.started_at, a.started_at));
    const out = [], seen = new Set();
    sorted.forEach(r => {
      if (!r.slate) { out.push({ row: r }); return; }
      if (seen.has(r.slate)) return;
      seen.add(r.slate);
      const members = sorted.filter(x => x.slate === r.slate);
      out.push({ slate: r.slate, members });
      members.forEach(m => out.push({ row: m, member: true }));
    });
    return out;
  }

  function preferenceText(slate, members) {
    const pref = members.map(m => m.preference).find(p => p && (p.slate == null || p.slate === slate));
    if (!pref) return 'no preference recorded yet';
    const judge = typeof pref.judge === 'string' ? pref.judge : pref.judge ? [pref.judge.kind, pref.judge.model].filter(Boolean).join(' ') : 'unknown judge';
    const blinded = pref.blinded != null ? pref.blinded : pref.judge && pref.judge.blinded;
    const win = pref.winner ? byRun.get(pref.winner) : null;
    const who = pref.winner == null ? 'a tie' : win ? `${esc(cfg(win).workflow || cfg(win).label || pref.winner)} <span class="m mono">${esc(pref.winner)}</span>` : esc(pref.winner);
    return `preferred: <b>${who}</b> <span class="m">by ${esc(judge)}${blinded ? ', blinded' : blinded === false ? ', not blinded' : ''}${pref.tier ? ', ' + esc(pref.tier) : ''}</span>`;
  }

  // The row's score description: `score_meta` when the builder wrote it, else the matching score signal.
  function scoreMeta(r) {
    if (r.score_meta) return r.score_meta;
    const det = detailOf(r.run), sig = det && (det.signals || []).find(s => s.kind === 'score' && s.value === r.score);
    return sig ? { name: sig.name, unit: sig.unit, better: sig.better } : {};
  }
  function scoreCell(r) {
    if (!num(r.score)) return '';
    const m = scoreMeta(r);
    return `<span class="sub">${esc(m.name ? m.name + ' ' : 'score ')}${esc(fmt.score(r.score, m.unit))}${arrow(m.better)}</span>`;
  }
  function surpriseBar(s) {
    if (!num(s)) return '';
    const w = Math.min(30, Math.abs(s) / 2 * 30), left = s < 0 ? 30 - w : 30;
    return `<span class="surp" title="surprise ${fmt.x(s)}: standardized residual on log cost, negative means cheaper than predicted; full width is 2"><i style="left:${left.toFixed(1)}px;width:${Math.max(1, w).toFixed(1)}px"></i></span>`;
  }
  function receiptCell(r) {
    const x = rc(r);
    if (!x.predicted) return '<span class="m">no prediction</span>';
    const inside = x.cost_in_interval === true ? '<span class="badge acc">cost inside</span>' : x.cost_in_interval === false ? '<span class="badge rej">cost outside</span>' : '<span class="badge unk">cost n/a</span>';
    return inside + surpriseBar(x.surprise);
  }
  const SOURCES = { usual: 'src', alternative: 'src', exploration: 'src-exploration', designed: 'src', habit: 'src-habit', user_edit: 'src' };
  function rowHtml(r, member) {
    const t = task(r), c = cfg(r);
    return `<tr class="row${member ? ' member' : ''}${r.run === selected ? ' sel' : ''}" data-run="${esc(r.run)}">` +
      `<td class="nw">${esc(fmt.date(r.started_at))}<span class="sub">${esc(fmt.time(r.started_at))}${r.state && r.state !== 'finished' ? ' ' + esc(r.state) : ''}</span></td>` +
      `<td>${esc(t.title || '(untitled)')}<span class="sub">${esc([t.type, t.subtype, t.repo].filter(Boolean).join(', '))}</span></td>` +
      `<td>${esc(c.workflow || 'n/a')}<span class="sub" title="${esc(c.label || '')}">${esc(c.label || c.id || '')}</span></td>` +
      `<td>${r.source ? `<span class="badge ${SOURCES[r.source] || 'src'}">${esc(r.source)}</span>` : '<span class="m">n/a</span>'}</td>` +
      `<td class="num nw">${r.cost ? money(r.cost.usd, r.cost.tokens) : 'n/a'}</td>` +
      `<td class="num">${num(r.rounds) ? r.rounds : 'n/a'}</td>` +
      `<td>${zBadge(r.z)} ${tier(r.tier)}${scoreCell(r)}</td>` +
      `<td class="nw">${receiptCell(r)}</td></tr>`;
  }

  function tableHtml(rows) {
    if (!rows.length) return `<p class="empty">${RUNS.length ? 'No runs match these filters.' : 'No runs yet.'}</p>`;
    const body = grouped(rows).map(g => g.slate
      ? `<tr class="grp member"><td colspan="8">slate <span class="mono">${esc(g.slate)}</span>: ${g.members.length} run${g.members.length === 1 ? '' : 's'} on the same task and base commit; ${preferenceText(g.slate, g.members)}</td></tr>`
      : rowHtml(g.row, g.member)).join('');
    return `<div class="tablescroll"><table><thead><tr><th>date</th><th>task</th><th>workflow</th><th>source</th><th class="num">cost</th><th class="num">rounds</th><th>outcome</th><th>receipt</th></tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function filtersHtml() {
    const types = uniq(RUNS.map(r => task(r).type)), repos = uniq(RUNS.map(r => task(r).repo)), sources = uniq(RUNS.map(r => r.source)), slates = uniq(RUNS.map(r => r.slate));
    return `<div class="filters">` +
      `<label>type<select data-f="type">${options(types, F.type, 'all types')}</select></label>` +
      `<label>repo<select data-f="repo">${options(repos, F.repo, 'all repos')}</select></label>` +
      `<label>source<select data-f="source">${options(sources, F.source, 'all sources')}</select></label>` +
      `<label>from<input type="date" data-f="from" value="${esc(F.from)}"></label>` +
      `<label>to<input type="date" data-f="to" value="${esc(F.to)}"></label>` +
      `<label>slate<select data-f="slate"><option value="">all runs</option><option value="(any)"${F.slate === '(any)' ? ' selected' : ''}>in a slate</option><option value="(none)"${F.slate === '(none)' ? ' selected' : ''}>not in a slate</option>${slates.map(s => `<option value="${esc(s)}"${F.slate === s ? ' selected' : ''}>${esc(s)}</option>`).join('')}</select></label>` +
      `<label>rule outcome<select data-f="outcome">${options(['accepted', 'not accepted', 'unknown'], F.outcome, 'any outcome')}</select></label>` +
      `<button data-reset>reset</button></div>` +
      (D.filters && D.filters.since ? `<p class="small m">Rows were already limited on the command line to runs since ${esc(D.filters.since)}${D.filters.since_at ? ' (' + esc(fmt.dt(D.filters.since_at)) + ')' : ''}; the filters here narrow them further.</p>` : '');
  }

  function statsHtml(t) {
    const stat = (v, l) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`;
    return `<div class="stats">` + stat(`${t.n}<span class="m">/${RUNS.length}</span>`, 'runs shown') + stat(`${fmt.usd(t.usd)} <span class="m small">${fmt.tok(t.tok)} tok</span>`, 'spend') +
      stat(t.known ? fmt.pct(t.acc / t.known) : 'n/a', `accepted (${t.acc} of ${t.known} known)`) +
      stat(t.withRc ? fmt.pct(t.inside / t.withRc) : 'n/a', `cost inside its range (${t.inside} of ${t.withRc} receipts)`) + `</div>`;
  }

  // Detail for one run: from `selected`, else from `details`, else row fields only.
  function detailOf(run) {
    if (D.selected && D.selected.run === run) return D.selected;
    return DETAILS[run] || null;
  }
  function realizedSuccess(r) { return r.z === 1 ? 1 : r.z === 0 ? 0 : null; }
  function receiptBlock(r, det) {
    const x = rc(r), p = x.predicted;
    if (!p) return '<p class="empty">No prediction was recorded before this run, so there is nothing to compare.</p>';
    const rows = [];
    const line = (label, interval, realized, f, o) => rows.push(`<tr><td>${label}</td><td>${iv(interval, f)}</td><td>${ivBar(interval, Object.assign({ realized, wide: true }, o || {}))}</td><td class="num">${num(realized) ? esc(f(realized)) : '<span class="m">n/a</span>'}</td></tr>`);
    // Success is a chance; the realized outcome is 0 or 1, so its dot is never marked as outside.
    if (p.p_success) line('success', p.p_success, realizedSuccess(r), v => fmt.pct(v), { min: 0, max: 1, binary: true });
    if (p.cost && p.cost.usd) line('cost, dollars', p.cost.usd, r.cost && r.cost.usd, fmt.usd, { log: true });
    if (p.cost && p.cost.tokens) line('cost, tokens', p.cost.tokens, r.cost && r.cost.tokens, fmt.tok, { log: true });
    if (p.rounds) line('rounds', p.rounds, r.rounds, fmt.rounds);
    Object.values(p.scores || {}).forEach(s => {
      if (!s || !s.value) return;
      const m = scoreMeta(r), sigs = (det && det.signals || []).filter(x => x.kind === 'score' && x.name === s.name && num(x.value));
      const real = sigs.length ? sigs[sigs.length - 1].value : m.name === s.name ? r.score : null;
      line(`${esc(s.name)}${arrow(s.better)}`, s.value, real, v => fmt.score(v, s.unit));
    });
    const sup = p.support ? Object.entries(p.support).filter(([, n]) => num(n)).map(([k, n]) => `${esc(k)} ${support(n)}`).join(', ') : '';
    return `<table class="small"><thead><tr><th>quantity</th><th>predicted (80 percent range)</th><th>range and realized dot</th><th class="num">realized</th></tr></thead><tbody>${rows.join('')}</tbody></table>` +
      `<p class="note">${x.cost_in_interval === true ? 'Cost landed inside the predicted range.' : x.cost_in_interval === false ? 'Cost landed outside the predicted range.' : ''}${num(x.surprise) ? ` Surprise ${fmt.x(x.surprise)} (the standardized residual on log cost; negative means cheaper than predicted).` : ''}${sup ? ` Support: ${sup}.` : ''}</p>`;
  }

  const sourceText = s => !s ? '' : typeof s === 'object' ? [s.kind, s.ref].filter(Boolean).join(': ') : String(s);
  function signalsBlock(sigs, doc) {
    if (!sigs || !sigs.length) return '<p class="empty">No signals recorded for this run.</p>';
    const ended = doc && doc.run && doc.run.ended_at;
    const items = sigs.slice().sort((a, b) => LM.byTime(a.observed_at, b.observed_at)).map(s => {
      const late = s.late === true || (LM.ts(ended) !== null && LM.ts(s.observed_at) !== null && LM.ts(s.observed_at) > LM.ts(ended));
      const val = s.kind === 'score' ? `${esc(fmt.score(s.value, s.unit))}${arrow(s.better)}${s.target != null ? ' <span class="m">target ' + esc(s.target) + '</span>' : ''}` : esc(s.value == null ? 'n/a' : typeof s.value === 'object' ? JSON.stringify(s.value) : s.value);
      return `<tr${late ? ' class="hl"' : ''}><td>${esc(fmt.dt(s.observed_at))}</td><td>${esc(s.kind || '')}</td><td>${esc(s.name || '')}</td><td>${val}</td><td>${tier(s.tier)}</td><td class="m small">${esc([sourceText(s.source), s.at_attempt ? 'attempt ' + s.at_attempt : ''].filter(Boolean).join(', '))}${late ? ' <b>late, after the run ended</b>' : ''}</td></tr>`;
    });
    return `<table class="small"><thead><tr><th>observed</th><th>kind</th><th>name</th><th>value</th><th>tier</th><th>source</th></tr></thead><tbody>${items.join('')}</tbody></table>`;
  }

  function renderDetail() {
    const host = document.getElementById('detail');
    if (!selected || !byRun.has(selected)) { host.hidden = true; host.innerHTML = ''; return; }
    const r = byRun.get(selected), det = detailOf(selected), t = task(r), c = cfg(r);
    host.hidden = false;
    let h = `<h2><span>${esc(t.title || selected)}</span><button data-close>close</button></h2>`;
    h += `<p class="lede">${esc(c.workflow || 'This run')} ran ${esc(fmt.dt(r.started_at))} as ${esc(r.source ? 'the ' + r.source + ' choice' : 'a run with no recorded source')}, cost ${fmt.usd(r.cost && r.cost.usd)} (${fmt.tok(r.cost && r.cost.tokens)} tokens) over ${num(r.rounds) ? r.rounds : 'an unknown number of'} round${r.rounds === 1 ? '' : 's'}, and ${r.z === 1 ? 'met' : r.z === 0 ? 'did not meet' : 'has no known result for'} its acceptance rule.</p>`;
    h += `<dl class="kv"><dt>run</dt><dd class="mono">${esc(selected)}</dd><dt>task</dt><dd>${esc([t.type, t.subtype, t.repo].filter(Boolean).join(', '))}</dd><dt>configuration</dt><dd>${esc(c.label || '')} <span class="m mono">${esc(c.id || '')}</span></dd><dt>outcome</dt><dd>${zBadge(r.z)} ${tier(r.tier)} ${num(r.q) ? '<span class="m">q ' + fmt.x(r.q) + '</span>' : ''} ${scoreCell(r)}</dd>${r.slate ? `<dt>slate</dt><dd class="mono">${esc(r.slate)}</dd>` : ''}</dl>`;
    h += `<h3>Receipt: predicted against realized</h3>${receiptBlock(r, det)}`;
    if (!det) {
      const cmd = `loopmath runs --run ${shq(selected)} --html`;
      h += `<h3>Graph, signals and raw document</h3><p class="note">This page carries full detail for the newest runs only. For this run:</p><div class="cmd"><code>${esc(cmd)}</code><button data-copy="${esc(cmd)}">copy command</button></div>`;
      host.innerHTML = h; return;
    }
    const wf = det.workflow || (det.graph && !LM.isRunGraph(det.graph) ? det.graph : null), rg = det.graph && LM.isRunGraph(det.graph) ? det.graph : null;
    h += `<h3>Workflow and attempts</h3><div id="d-wf" class="lmg"></div>`;
    if (rg) h += `<h3>Attempts over time</h3><div id="d-rg"></div>`;
    else if (det.graph_error) h += `<p class="note">The attempt timeline is unavailable: ${esc(det.graph_error)}.</p>`;
    h += `<h3>Signals</h3>${signalsBlock(det.signals, det.doc)}`;
    h += `<details class="raw" id="d-raw"><summary>Raw OCP document</summary><pre></pre></details>`;
    host.innerHTML = h;
    try { LM.Graph.render(document.getElementById('d-wf'), wf, {}); } catch (e) { document.getElementById('d-wf').innerHTML = `<p class="warn">The workflow graph could not be drawn: ${esc(e.message)}</p>`; }
    if (rg) { try { LM.RunGraph.render(document.getElementById('d-rg'), rg, {}); } catch (e) { document.getElementById('d-rg').innerHTML = `<p class="warn">The attempt layouts could not be drawn: ${esc(e.message)}</p>`; } }
    const raw = document.getElementById('d-raw');
    raw.addEventListener('toggle', () => { const pre = raw.querySelector('pre'); if (raw.open && !pre.textContent) pre.textContent = JSON.stringify(det.doc, null, 2); });
  }

  function draw() {
    const rows = RUNS.filter(pass), t = totals(rows);
    document.getElementById('lede').innerHTML = lede(t);
    document.getElementById('stats').innerHTML = statsHtml(t);
    document.getElementById('table').innerHTML = tableHtml(rows);
  }
  function select(run, scroll) {
    selected = run;
    document.querySelectorAll('tr.row').forEach(tr => tr.classList.toggle('sel', tr.dataset.run === run));
    renderDetail();
    if (run && history.replaceState) history.replaceState(null, '', '#run=' + encodeURIComponent(run));
    if (run && scroll) { const el = document.getElementById('detail'); if (el.scrollIntoView) el.scrollIntoView({ block: 'start', behavior: 'smooth' }); }
  }

  app.innerHTML = `<header class="top"><h1><span class="k">loopmath</span> runs</h1><p class="lede" id="lede"></p></header>` +
    `<section class="panel">${filtersHtml()}<div id="stats" style="margin-top:10px"></div></section>` +
    `<section class="panel"><div id="table"></div></section><section class="panel" id="detail" hidden></section>` +
    `<p class="foot">Generated ${esc(fmt.dt(D.generated_at))}. ${RUNS.length} run${RUNS.length === 1 ? '' : 's'} in this page${Object.keys(DETAILS).length ? `, full detail for ${Object.keys(DETAILS).length}` : ''}. Receipt ranges are 80 percent; the dot is the realized value, red when outside.` +
    (Array.isArray(D.notes) ? D.notes.map(n => ' ' + esc(n)).join('') : '') + `</p>`;
  app.querySelector('.filters').addEventListener('change', ev => { const k = ev.target.dataset.f; if (k) { F[k] = ev.target.value; draw(); } });
  app.querySelector('[data-reset]').addEventListener('click', () => { Object.keys(F).forEach(k => { F[k] = ''; }); app.querySelectorAll('[data-f]').forEach(el => { el.value = ''; }); draw(); });
  document.getElementById('table').addEventListener('click', ev => { const tr = ev.target.closest('tr.row'); if (tr) select(tr.dataset.run === selected ? null : tr.dataset.run, tr.dataset.run !== selected); });
  document.getElementById('detail').addEventListener('click', ev => {
    if (ev.target.closest('[data-close]')) { select(null); return; }
    const b = ev.target.closest('[data-copy]'); if (b) copy(b.dataset.copy, b);
  });
  const hash = /^#run=(.+)$/.exec(location.hash || '');
  if (hash && byRun.has(decodeURIComponent(hash[1]))) selected = decodeURIComponent(hash[1]);
  draw();
  renderDetail();
  LM.autoSort();  // P7: every table sorts by a click on its column header, again to reverse
})();
