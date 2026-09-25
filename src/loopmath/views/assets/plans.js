// Planning page (lane 2E, 0.2; D119 Z3 direction C in the paper look): `loopmath recommend --html`. It leads with one
// decision, the pick, then puts the rest in details sections whose closed line still answers the question.
// Renders `loopmath.view.plans/1` from the embedded object only; recommend/1 objects (no `numbers`) still render.
// 0.2.1 (lane 21B): the chance-against-cost chart at the top (P9), dot-and-line intervals from `bands` instead of
// distribution shapes (P6), a sortable options table (P7), "Copy option" buttons that copy `option <n>: <label>` (P4)
// and info icons on run cost and cost per accepted result (P5). Without `bands`, `option`, `p_accepted_within` or the
// retry rescue fields (0.2.0 objects), the page falls back to the 80% range and the 0.2.0 wording.
(() => {
  const V = LM.Viz, { esc, num, fmt, TAIL_NOTE, pulledUp } = LM;
  const D = LM.data(), app = document.getElementById('app');
  const rule = D.rule || {}, target = rule.score || null, isScore = !!target;
  const task = D.task || {}, fit = D.fit || {}, EX = D.exploration || {}, RESCUE = D.rescue || null;
  const op = target && target.better === 'lower' ? '<=' : '>=';
  const targetText = isScore ? `${target.name} ${op} ${fmt.x(target.target)}` : '';

  // ------------------------------------------------------------ candidates by configuration id
  const BY = new Map();
  const cid = c => typeof c === 'string' ? c : c && c.config ? (typeof c.config === 'string' ? c.config : c.config.id) : null;
  const add = c => { const id = cid(c); if (!id) return; if (c.prediction && !BY.has(id)) BY.set(id, c); else if (BY.has(id) && c.numbers && !BY.get(id).numbers) BY.get(id).numbers = c.numbers; };
  (D.candidates || []).forEach(add);
  (D.alternatives || []).forEach(add);
  // An exploration entry is {candidate, gain_per_run, ...}, {paused, would_have_been}, {none} or {same_as} (spec 02, 05).
  const pickOf = p => !p ? null : p.would_have_been ? p.would_have_been : p.candidate || p.config ? p : null;
  const pickCid = p => p ? cid(p.candidate || p) : null;
  const pickFor = k => { const p = EX[k]; return p && p.same_as ? pickOf(EX[p.same_as]) : pickOf(p); };
  const paused = k => { const p = EX[k]; return !!(p && (p.paused || (p.same_as && EX[p.same_as] && EX[p.same_as].paused))); };
  ['best_value', 'max_gain'].forEach(k => { const p = pickOf(EX[k]); if (p && p.candidate) add(p.candidate); });
  // The baseline: the usual, else the reference (recommend/2: usual, best_recorded or default). Without a usual
  // the page never says "your usual".
  const REF = D.reference || null;
  const BASE = D.usual && D.usual.config ? D.usual : REF && REF.config ? REF : null;
  const isUsual = !!(D.usual && D.usual.config) || !!(REF && REF.kind === 'usual');
  if (BASE && BASE.prediction) add({ config: BASE.config, origin: isUsual ? 'usual' : 'reference', diff_vs_usual: [], prediction: BASE.prediction, numbers: BASE.numbers });
  const refId = BASE ? cid(BASE) : null;
  const refWords = isUsual ? 'your usual workflow' : REF && REF.kind === 'best_recorded' ? 'your best recorded workflow' : 'the default workflow';
  const refShort = isUsual ? 'your usual' : 'the reference';
  const refWhy = isUsual || !BASE ? '' : REF && REF.kind === 'best_recorded' ? 'No usual workflow, so your best recorded workflow stands in as the reference.' : 'No usual workflow and no recorded one, so the default workflow stands in as the reference.';
  const goalId = (D.goal && D.goal.config) || (D.default_pick && D.default_pick.config) || null;
  // With fewer than five scored runs of the task type, or no score head, the fit prices a miss with the success head's
  // chance of an accepted result (`success_from: success_head`, recommend's score_backed), and `numbers.p_reach` holds
  // the score head's chance to reach, which no total uses. The page then shows and prices by the chance the model
  // used, calls it the chance of an accepted result, and shows the score estimate apart where there is one.
  const FB = isScore && [...BY.values()].some(c => c.prediction && c.prediction.success_from === 'success_head');
  const gWords = isScore && !FB ? `chance of reaching ${targetText}` : 'chance of an accepted result';
  const gShort = isScore && !FB ? `chance to reach ${fmt.x(target.target)}` : 'chance of an accepted result';
  const fbNote = FB ? `Too few ${target.name} scores for this kind of task to predict ${targetText}, so each chance here is of an accepted result.` : '';

  // I12: a piece of width n > 1 reads '3 x gpt-5.6-sol/xhigh', as views/common.py config_label.
  const widthLabel = cfg => {
    const ps = cfg && cfg.workflow && cfg.workflow.pieces, ss = (cfg && cfg.settings) || {};
    if (!Array.isArray(ps) || !ps.some(p => (p.width || 1) > 1)) return null;
    const parts = ps.map(p => [p, ss[p.id] || p.setting]).filter(([, s]) => s && typeof s === 'object')
      .map(([p, s]) => `${(p.width || 1) > 1 ? p.width + ' x ' : ''}${V.modelName(s.model)}/${s.effort || 'default'}`);
    return `${cfg.workflow.id || 'workflow'}: ${parts.join(', ')}`;
  };
  const labelOf = id => {
    const c = BY.get(id), cfg = c && c.config, w = widthLabel(cfg);
    if (w) return w;
    if (c && c.label) return c.label;
    if (id === refId && BASE && BASE.label) return BASE.label;
    return (D.graphs && D.graphs[id] && D.graphs[id].label) || id;
  };
  const splitLabel = id => { const s = labelOf(id), i = s.indexOf(': '); return i > 0 ? [s.slice(0, i), s.slice(i + 2)] : [s, '']; };

  // ------------------------------------------------------------ the numbers of one option
  // g: P(reach) for score rules, else P(success). Its 80 percent range (P3a) is `numbers.p_reach` (recommend/2); else,
  // with success_from score_head, p_success holds the same draws, so its range is the reach range.
  const scoreOf = p => isScore && p && p.scores ? p.scores[target.name] : null;
  function g(p, nb) {
    const c = V.clamp01, pr = nb && nb.p_reach;
    if (isScore && !FB && pr && num(pr.mean)) return { mean: c(pr.mean), lo: c(pr.lo), hi: c(pr.hi) };
    if (!p) return null;
    const s = scoreOf(p);
    if (isScore && !FB && p.success_from === 'score_head' && s && num(s.p_reach)) {
      const ps = p.p_success || {}, same = num(ps.mean) && Math.abs(ps.mean - s.p_reach) < 1e-6 && num(ps.lo) && num(ps.hi);
      return { mean: s.p_reach, lo: same ? c(ps.lo) : null, hi: same ? c(ps.hi) : null };
    }
    return p.p_success ? { mean: c(p.p_success.mean), lo: c(p.p_success.lo), hi: c(p.p_success.hi) } : null;
  }
  // The score estimate's chance to reach the target, shown apart under the fallback where the score head exists.
  function scoreReach(p, nb) {
    if (!FB) return null;
    const pr = nb && nb.p_reach, s = scoreOf(p);
    return pr && num(pr.mean) ? V.clamp01(pr.mean) : s && num(s.p_reach) ? V.clamp01(s.p_reach) : null;
  }
  const priced = !!(RESCUE && RESCUE.kind !== 'none' && num(RESCUE.usd) && RESCUE.usd > 0);
  function row(id) {
    const c = BY.get(id);
    if (!c) return null;
    const p = c.prediction || {}, nb = c.numbers || null, gg = g(p, nb);
    const run = nb && nb.run_cost_usd ? nb.run_cost_usd : p.cost && p.cost.usd ? p.cost.usd : null;
    const ell = nb && nb.cost_per_accepted_usd ? nb.cost_per_accepted_usd : p.ell && p.ell.usd ? p.ell.usd : null;
    const rescue = nb && num(nb.expected_rescue_usd) ? nb.expected_rescue_usd : priced && gg && num(gg.mean) ? (1 - gg.mean) * RESCUE.usd : null;
    const ch = CHOSEN[id] || {}, bands = (nb && nb.bands) || ch.bands || null, pw = (nb && nb.p_accepted_within) || ch.p_accepted_within;
    return { id, c, label: labelOf(id), g: gg, sr: scoreReach(p, nb), run, ell, rescue, support: num(p.support) ? p.support : null, origin: c.origin || '', tokens: p.cost && p.cost.tokens,
      bands, paw: pw && num(pw.mean) ? pw : null };
  }
  // 21M (0.2.1): each choice has `option`, its 1-based number (else its place in `choices`); the page and the plan
  // skill number options the same way. The pair runs two options side by side.
  const OPTS = (D.choices || []).map((c, i) => c && c.key ? { n: num(c.option) ? c.option : i + 1, key: c.key, config: c.config, label: c.label || '', c } : null).filter(Boolean);
  const CHOSEN = {}, OPT = {};
  OPTS.forEach(o => { if (o.key !== 'pair' && o.config && !(o.config in OPT)) { OPT[o.config] = o; CHOSEN[o.config] = o.c; } });
  const pairOpt = OPTS.find(o => o.key === 'pair' && (o.c.members || []).length === 2) || null;
  const ROWS = [...BY.keys()].map(row).filter(r => r && r.ell && num(r.ell.mean)).sort((a, b) => a.ell.mean - b.ell.mean);
  const R = new Map(ROWS.map(r => [r.id, r]));
  const pick = R.get(goalId) || null, ref = refId ? R.get(refId) || null : null;
  // p_accepted_within: the chance of an accepted result within K attempts, shown as its own column when present.
  const PAW = ROWS.some(r => r.paw);
  const K = (ROWS.find(r => r.paw && num(r.paw.attempts)) || { paw: {} }).paw.attempts || (RESCUE && num(RESCUE.max_attempts) ? RESCUE.max_attempts : 3);
  const pawHead = `chance within ${K} attempt${K === 1 ? '' : 's'}`;
  // P7 (0.2.1): the options table sorts by any column; the first click takes the useful direction (chances and
  // runs high first, costs low first), the next reverses it. Missing values stay last either way.
  const COLS = [
    { k: 'label', h: 'workflow', dir: 1, v: r => r.label.toLowerCase() },
    { k: 'g', h: gShort, dir: -1, v: r => r.g && num(r.g.mean) ? r.g.mean : null },
    ...(PAW ? [{ k: 'paw', h: pawHead, dir: -1, v: r => r.paw ? r.paw.mean : null }] : []),
    { k: 'run', h: 'cost per run', num: true, info: 'run', dir: 1, v: r => r.run && num(r.run.mean) ? r.run.mean : null },
    { k: 'rescue', h: 'expected rescue', num: true, dir: 1, v: r => num(r.rescue) ? r.rescue : null },
    { k: 'ell', h: 'cost per accepted result', num: true, info: 'ell', dir: 1, v: r => r.ell.mean },
    { k: 'support', h: 'runs behind it', num: true, dir: -1, v: r => r.support },
  ];
  let sortKey = 'ell', sortDir = 1;
  function sorted() {
    const col = COLS.find(c => c.k === sortKey) || COLS[0];
    return ROWS.map((r, i) => [r, col.v(r), i]).sort((a, b) => {
      const x = a[1], y = b[1];
      if (x == null || y == null) return x == null && y == null ? a[2] - b[2] : x == null ? 1 : -1;
      return (x < y ? -1 : x > y ? 1 : 0) * sortDir || a[2] - b[2];
    }).map(e => e[0]);
  }

  // ------------------------------------------------------------ marks and commands
  const tryKinds = id => ['best_value', 'max_gain'].filter(k => { const p = pickFor(k); return p && pickCid(p) === id; });
  const TITLE = { best_value: 'best value to try', max_gain: 'biggest gain to try' };
  function marks(id) {
    const m = [];
    if (OPT[id]) m.push(['opt', 'option ' + OPT[id].n]);
    if (id === goalId) m.push(['pick', 'recommended' + (D.goal && num(D.goal.level) ? ` (${D.goal.level}% row)` : '')]);
    if (D.default_pick && D.default_pick.config === id && id !== goalId) m.push(['other', 'default pick']);
    if (id === refId) m.push(['ref', refShort]);
    tryKinds(id).forEach(k => m.push(['try', TITLE[k] + (paused(k) ? ' (paused)' : '')]));
    return m;
  }
  const markHtml = id => marks(id).map(([k, t]) => `<span class="v-mk ${k}">${esc(t)}</span>`).join('');
  // P4 (0.2.1): a copy button hands over the option, not a command: the user pastes it into the conversation with
  // their agent, which types the commands. A numbered option copies `option <n>: <label>`; any other workflow copies
  // `workflow <config id>: <label>`, which the agent can start with `--config`.
  const COPY = 'Copy option';
  const copyText = id => OPT[id] ? `option ${OPT[id].n}: ${OPT[id].label || labelOf(id)}` : `workflow ${id}: ${labelOf(id)}`;
  const pairText = (a, b) => pairOpt && pairOpt.c.members[0] === a && pairOpt.c.members[1] === b ? `option ${pairOpt.n}: ${pairOpt.label || labelOf(a) + ' + ' + labelOf(b)}`
    : `${copyText(a)}, and beside it ${copyText(b)}`;

  // ------------------------------------------------------------ text helpers
  const usd = V.usd, pct = V.pct;
  const rng = (x, f) => x && num(x.lo) && num(x.hi) ? `${f(x.lo)} to ${f(x.hi)}` : '';
  const tail = (...xs) => xs.some(pulledUp) ? ` <span class="v-tail">${esc(TAIL_NOTE)}</span>` : '';
  const median = r => r && r.run && num(r.run.median) ? `median ${usd(r.run.median)}${r.run.median_basis && r.run.median_basis !== 'draws' ? ' (approx.)' : ''}` : '';
  const supportText = n => !num(n) ? 'n/a' : n === 0 ? 'none' : fmt.int(n);
  // D96/D102: the look-ahead value in dollars, as lane 6's recommend message words it.
  const savingText = gp => !gp || !num(gp.usd) ? 'n/a' : `expected to save about ${gp.usd > 0 && gp.usd < 0.01 ? 'under $0.01' : '$' + gp.usd.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} per future similar run`;
  // I20 (0.2.1, lane 21A): one payback number everywhere: ceil(price now / gain), at least 1, as message.py words it.
  const paybackText = n => { if (!num(n)) return 'n/a'; const k = Math.max(1, Math.ceil(n - 1e-9)); return `about ${k} similar run${k === 1 ? '' : 's'}`; };
  // D119 Z2: lane 2A's sentence, shown verbatim; the page checks only that it is set.
  const strategy = (D.goal && D.goal.strategy) || ((D.choices || []).find(c => c && c.key === 'goal') || {}).strategy || null;
  const strategyText = strategy && typeof strategy.text === 'string' && strategy.text ? strategy.text : null;

  // P5 (0.2.1): what run cost and cost per accepted result mean, on hover and tap of a small info icon. 21M's
  // `rescue.text` is quoted when set; else the sentence is built from the retry fields; else the 0.2.0 wording.
  const RETRY = !!(RESCUE && RESCUE.kind === 'retry');
  function rescueSentence() {
    if (!RESCUE) return '';
    if (typeof RESCUE.text === 'string' && RESCUE.text) return RESCUE.text;
    if (RETRY) {
      const d = num(RESCUE.decay) ? RESCUE.decay : 0.5, n = num(RESCUE.max_attempts) ? RESCUE.max_attempts : K;
      const each = d === 0.5 ? 'half the chance' : d === 1 ? 'the same chance' : `${fmt.x(d)} times the chance`;
      const ch = RESCUE.chance && num(RESCUE.chance.mean) ? ` (chance per run ${pct(RESCUE.chance.mean)})` : '';
      return `A miss is fixed by retrying with ${RESCUE.of || 'the rescue workflow'}${ch}. Each extra attempt has ${each} of the one before, up to ${n} attempts in all, counting the first run.`;
    }
    if (!priced) return 'No rescue is priced (rescue: none), so the cost per accepted result is the cost per run.';
    return `A miss is rescued by ${RESCUE.basis || RESCUE.kind}${RESCUE.of ? ` (${RESCUE.of})` : ''}: about ${usd(RESCUE.usd)}.`;
  }
  const INFO = {
    run: 'Run cost: what the agents cost for one run of this workflow (the mean of the fitted cost). It leaves out fixing a miss.',
    ell: (() => {
      if (!RESCUE || !priced) return 'Cost per accepted result: the run cost plus the expected cost of fixing a miss. ' + (rescueSentence() || 'No rescue is priced here, so it equals the run cost.');
      // 21M's rescue.text may already say what cost per accepted result is; then it stands alone.
      const own = typeof RESCUE.text === 'string' && /cost per accepted result/i.test(RESCUE.text);
      let s = (own ? '' : 'Cost per accepted result: the run cost plus the expected cost of fixing a miss. ') + rescueSentence();
      if (RETRY) s += ` The cost of those retries is divided by the chance that they end in an accepted result${num(RESCUE.p_accepted) ? ` (${pct(RESCUE.p_accepted)})` : ''}, then weighted by the chance that the first run misses.`;
      else s += ' The rescue is weighted by the chance of a miss: cost per run + chance of a miss x the rescue.';
      if (PAW) s += ` The chance of an accepted result within ${K} attempts is its own column; the chance column is for one run.`;
      return s;
    })(),
  };
  // A legend entry per interval level, drawn as a short line of the width the chart uses.
  const bandKey = levels => levels.map(([lv, w]) => `<span><svg width="22" height="8" aria-hidden="true"><line x1="1" x2="21" y1="4" y2="4" stroke="currentColor" stroke-width="${w}" stroke-linecap="round"/></svg>${lv}% range</span>`).join('');
  const info = k => `<button type="button" class="v-info" data-info="${k}" aria-label="${esc(INFO[k])}">i</button>`;

  // Three ways to run the task: the pick, the reference, the best option to try.
  const THREE = (() => {
    const out = [], seen = new Set(), put = (r, label, mark) => { if (r && r.ell && !seen.has(r.id)) { seen.add(r.id); out.push({ r, label, mark }); } };
    put(pick, 'recommended', 'pick'); put(ref, refShort, 'ref');
    const bvId = pickCid(pickFor('best_value')), mgId = pickCid(pickFor('max_gain')), tryId = bvId || mgId;
    put(tryId ? R.get(tryId) : null, tryId === bvId ? 'best value to try' : 'biggest gain to try', 'try');
    const cheap = (D.choices || []).find(c => c && c.key === 'cheapest_run');
    if (out.length < 3 && cheap) put(R.get(cheap.config), 'cheapest with a 50% chance', '');
    return out;
  })();

  // ------------------------------------------------------------ the page
  const bits = [`<b class="v-brand">loopmath <span>plan</span></b>`];
  if (task.type) bits.push(`<span>${esc(task.type)}${task.repo ? ' on ' + esc(task.repo) : ''}${task.title ? ': ' + esc(task.title) : ''}</span>`);
  if (isScore) bits.push(`<span>target <b>${esc(targetText)}</b></span>`);
  else if (rule.name) bits.push(`<span>rule <b>${esc(rule.name)}</b></span>`);
  bits.push(`<span>${ROWS.length} options priced</span>`);
  let h = `<div class="v-wrap"><div class="v-top">${bits.join('')}</div>`;
  h += pick ? chartBlock() + hero() + threeBlock() + details() : `<section class="v-hero"><p class="v-kicker">Recommended</p><h1>No pick in this recommendation.</h1><p class="v-lede">${esc(D.message || '')}</p></section>`;
  h += `<p class="v-foot">loopmath recommend ${esc(D.rec || '')}, fit ${esc(fit.id || 'n/a')}${D.generated_at ? ', page written ' + esc(fmt.dt(D.generated_at)) : ''}. Ranges are 80% unless marked. The page runs nothing: copy an option and paste it into the conversation with your agent.</p></div>`;
  app.innerHTML = h;

  // P9 (0.2.1): chance against cost at the top, every numbered option as a point (the pair runs two of them).
  function chartOpts() {
    const out = [], seen = new Set();
    OPTS.forEach(o => {
      if (o.key === 'pair' || !o.config || seen.has(o.config)) return;
      seen.add(o.config);
      const r = R.get(o.config), c = o.c;
      const x = r || { id: o.config, label: o.label || labelOf(o.config), g: c.chance || null, run: c.run_cost_usd || null, ell: c.cost_per_accepted_usd || null,
        rescue: num(c.expected_rescue_usd) ? c.expected_rescue_usd : null, support: null, bands: c.bands || null, paw: c.p_accepted_within && num(c.p_accepted_within.mean) ? c.p_accepted_within : null };
      out.push({ o, r: x });
    });
    return out;
  }
  function chartBlock() {
    const n = chartOpts().length;
    if (!n) return '';
    return `<section class="v-block v-cc" id="cc"><p class="v-kicker">Your options</p><h2>${esc(isScore && !FB ? `Chance to reach ${fmt.x(target.target)}` : 'Chance of an accepted result')} against cost</h2>` +
      `<p class="v-sub">${n} numbered option${n === 1 ? '' : 's'}${pairOpt ? `; option ${pairOpt.n} runs two of them side by side` : ''}. Up and to the left is better. The faint dots are the other priced workflows.</p>` +
      `<div class="v-seg" role="group" aria-label="cost axis"><button type="button" data-x="ell" aria-pressed="true">cost per accepted result</button><button type="button" data-x="run" aria-pressed="false">run cost</button></div>` +
      `<div id="ccplot"></div><div class="v-legend" id="cclegend">${bandKey([['50', 2.4], ['80', 1.2], ['90', 0.6]])}<span><span class="dt" style="background:var(--accent)"></span>recommended</span><span><span class="dt" style="background:var(--dot);opacity:.6;width:6px;height:6px"></span>other workflows</span></div></section>`;
  }

  function hero() {
    const [gName, gSettings] = splitLabel(pick.id);
    let s = `<section class="v-hero" id="pick"><p class="v-kicker">Recommended${D.goal && num(D.goal.level) ? `: the ${esc(D.goal.level)}% row of the curve` : ''}</p>` +
      `<h1>Run ${esc(gName)}${gSettings ? ' with ' + esc(gSettings) : ''}.</h1>` +
      (strategyText ? `<p class="v-strategy" id="strategy">${esc(strategyText)}</p>` : '') +
      `<div class="v-gfull" id="pg"></div><p class="v-gcap" id="pgcap"></p>` +
      `<div class="v-nums" id="nums">` +
      `<div class="lead"><div class="n">${usd(pick.ell.mean)}</div><div class="l">per accepted result ${info('ell')}</div><div class="r">${rng(pick.ell, usd)}${tail(pick.ell)}</div></div>` +
      `<div><div class="n">${pick.g && num(pick.g.mean) ? pct(pick.g.mean) : 'n/a'}</div><div class="l">${esc(gShort)}${pick.paw ? ' in one run' : ''}</div><div class="r">${rng(pick.g, pct)}</div>` +
      (pick.paw ? `<div class="r" id="paw">${pct(pick.paw.mean)} within ${esc(K)} attempts${num(pick.paw.lo) ? ` (${rng(pick.paw, pct)})` : ''}</div>` : '') + `</div>` +
      `<div><div class="n">${pick.run ? usd(pick.run.mean) : 'n/a'}</div><div class="l">a run ${info('run')}</div><div class="r">${median(pick)}${tail(pick.run)}</div></div></div>`;
    if (num(pick.support) && pick.support < 3) {
      s += `<div class="v-warn" id="thin"><i>!</i><span><b>${pick.support === 0 ? 'No runs of this workflow yet.' : `Only ${V.runs(pick.support)} of this workflow.`}</b> The estimate comes mostly from the fitted model, not from runs of this workflow${pick.g && num(pick.g.lo) ? `, so the ${esc(gWords)} ranges from ${pct(pick.g.lo)} to ${pct(pick.g.hi)}` : ''}.</span></div>`;
    }
    let vs = '';
    if (ref && ref.id === pick.id) vs = `This is ${esc(refWords)}.`;
    else if (ref) {
      const d = ref.ell.mean - pick.ell.mean, name = `${esc(refShort)}, ${esc(ref.label)} (${usd(ref.ell.mean)})`;
      vs = Math.abs(d) < 0.005 ? `About the same per accepted result as ${name}.` : `<b>${usd(Math.abs(d))} ${d > 0 ? 'less' : 'more'}</b> per accepted result than ${name}.`;
    }
    if (refWhy) vs += ' ' + esc(refWhy);
    if (FB) vs += ` <span id="fallback">${esc(fbNote)}${num(pick.sr) ? ` The score estimate gives the pick a ${pct(pick.sr)} chance to reach ${esc(fmt.x(target.target))}; no total uses it.` : ''}</span>`;
    if (priced) vs += ` A miss is rescued by ${esc(RESCUE.basis || RESCUE.kind)}${RESCUE.of ? ` (${esc(RESCUE.of)})` : ''}: about <b>${usd(RESCUE.usd)}</b>.`;
    else if (RESCUE) vs += ' No rescue is priced (rescue: none), so the cost per accepted result is the cost per run.';
    return s + `<p class="v-vs" id="vs">${vs.trim()}</p>` + V.copyBtn(copyText(pick.id), COPY) + `</section>`;
  }

  function threeBlock() {
    let s = `<section class="v-block" id="ways"><h2>${THREE.length === 3 ? 'Three' : THREE.length === 2 ? 'Two' : 'One'} way${THREE.length === 1 ? '' : 's'} to run this task</h2>` +
      `<p class="v-sub">Cost per accepted result, log scale. The dot is the mean; the lines are the 80%, 90% and 95% ranges, thinner as they widen.</p><div id="three"></div>`;
    if (D.pair && (D.pair.members || []).length === 2) {
      const [a, b] = D.pair.members, bet = pickFor(D.pair.explore_pick || 'best_value');
      s += `<p class="v-pairline" id="pairline">Or run the pick and <b>${esc(labelOf(b))}</b> side by side, then let a blinded referee choose.` +
        (bet && bet.price && bet.price.usd ? ` It costs ${usd(bet.price.usd.mean)} more now; trying it once is ${esc(savingText(bet.gain_per_run))}; there is a ${pct(bet.p_beats_goal)} chance it beats the recommended pick.` : '') + `</p>` +
        V.copyBtn(pairText(a, b), COPY);
    }
    return s + `</section>`;
  }

  function details() {
    const sec = (id, title, answer, body) => `<details class="v-more" id="${id}"><summary><span class="st">${title}</span><span class="sa">${answer}</span></summary><div class="v-body">${body}</div></details>`;
    const miss = pick.g && num(pick.g.mean) ? 1 - pick.g.mean : null;
    let out = '';
    // the arithmetic
    const eq = [['cost per run', pick.run ? usd(pick.run.mean) : 'n/a', `mean of the fitted cost${median(pick) ? '; ' + median(pick) : ''}${pick.run && num(pick.run.lo) ? `; 80% range ${rng(pick.run, usd)}` : ''}`]];
    if (priced) {
      eq.push(['chance of a miss', num(miss) ? pct(miss) : 'n/a', `1 minus the ${pick.g ? pct(pick.g.mean) : 'n/a'} ${esc(gWords)}`]);
      eq.push(['rescue', usd(RESCUE.usd), RETRY ? `per fixed miss: retrying with ${esc(RESCUE.of || 'the rescue workflow')}, up to ${esc(num(RESCUE.max_attempts) ? RESCUE.max_attempts : K)} attempts in all`
        : `${esc(RESCUE.basis || RESCUE.kind)}${RESCUE.of ? ': ' + esc(RESCUE.of) : ''}`]);
      eq.push(['expected rescue', usd(pick.rescue), `${num(miss) ? pct(miss) : 'n/a'} x ${usd(RESCUE.usd)}`]);
    }
    const tot = ['per accepted result', usd(pick.ell.mean), `80% range ${rng(pick.ell, usd)}${pulledUp(pick.ell) ? '; ' + esc(TAIL_NOTE) : ''}`];
    out += sec('d-math', 'The arithmetic', priced ? `${pick.run ? usd(pick.run.mean) : 'n/a'} a run + ${num(miss) ? pct(miss) : 'n/a'} x ${usd(RESCUE.usd)} rescue = ${usd(pick.ell.mean)}` : `${usd(pick.ell.mean)} per accepted result; no rescue priced`,
      `<div class="v-math"><div class="eq">${eq.map(e => e.map(x => `<span>${x}</span>`).join('')).join('')}${tot.map(x => `<span class="tot">${x}</span>`).join('')}</div></div>` +
      `<p class="v-note">Cost per accepted result = cost per run + chance of a miss x the rescue. Parts are rounded to the cent, so they may not add up exactly.${FB ? ' ' + esc(fbNote) : ''}</p>`);
    // every option
    out += sec('d-all', `All ${ROWS.length} options`, ROWS.length ? `cheapest per accepted result first; ${esc(ROWS[0].label)} leads at ${usd(ROWS[0].ell.mean)}` : 'none',
      `<div class="v-scroll"><table class="v-t" id="tall"><thead><tr>${COLS.map(c => `<th class="v-sort${c.num ? ' num' : ''}" data-sort="${c.k}" aria-sort="${c.k === sortKey ? (sortDir > 0 ? 'ascending' : 'descending') : 'none'}" tabindex="0">` +
        `${esc(c.h)}${c.info ? ' ' + info(c.info) : ''}<span class="v-arr" aria-hidden="true"></span></th>`).join('')}</tr></thead><tbody></tbody></table></div>` +
      (ROWS.length > 12 ? `<button type="button" class="v-btn" id="more">Show all ${ROWS.length}</button>` : '') +
      `<p class="v-note">Click a column head to sort, again to reverse. Click a row for its graph and to copy it. Ranges are 80%.</p>`);
    // how wide
    const widest = ROWS.filter(r => r.g && num(r.g.lo) && num(r.g.hi) && r.g.hi - r.g.lo > 0.8).length;
    out += sec('d-wide', 'How wide the estimates are', `${widest} of ${ROWS.length} chances span more than 80 points; the pick runs from ${rng(pick.ell, usd) || 'n/a'}`,
      `<div class="v-legend"><span><span class="dt" style="background:var(--accent)"></span>recommended</span><span><span class="dt no"></span>${esc(refShort)}</span>${bandKey([['80', 2.6], ['90', 1.3], ['95', 0.6]])}</div><div id="forest"></div>`);
    // exploration
    const bets = betList().filter(b => !b.none), bv = bets.find(b => b.kind === 'best_value') || bets[0];
    out += sec('d-try', 'Worth trying something new', bv ? `${esc(bv.label)}: ${usd(bv.price.mean)} now, pays for itself after ${paybackText(bv.payback)}` : exploreNone(),
      (bets.length ? `<div class="v-legend">${bets.map((b, i) => `<span><span class="sw" style="background:${i ? 'var(--ink)' : 'var(--accent)'}"></span>${esc(b.title)}</span>`).join('')}</div><div id="pay"></div>` +
        `<p class="v-note">Net saving after n future similar runs is the saving per run times n, minus the price now. The payback range comes from the price range only.</p>` : '') +
      `<div class="v-bets">${['best_value', 'max_gain'].map(betCard).join('')}</div>` +
      (D.pair && (D.pair.instructions || []).length ? `<h3>Running the pair</h3><ul class="v-note">${D.pair.instructions.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : ''));
    // sources
    const origins = {}; ROWS.forEach(r => { const k = r.origin || 'other'; origins[k] = (origins[k] || 0) + 1; });
    const ORIGIN = { catalog: 'from the catalog', edit: 'edits of the reference', recorded: 'you recorded',
      user: 'given with --workflow', front: 'found by the search', thompson: 'found in a search draw', polish: 'near the pick' };  // front, thompson, polish: lane 2A's search (0.2)
    const n = fit.n_runs || {};
    out += sec('d-src', 'Where these numbers come from', `fit ${esc(fit.id || 'n/a')}, recommendation ${esc(D.rec || 'n/a')}`,
      `<dl class="v-kv"><dt>fit</dt><dd class="v-mono">${esc(fit.id || 'n/a')}${fit.at ? ' <span class="m">' + esc(fmt.dt(fit.at)) + '</span>' : ''}</dd>` +
      (num(n.user) || num(n.prior) ? `<dt>runs in the fit</dt><dd>${fmt.int(n.user || 0)} yours, ${fmt.int(n.prior || 0)} from the shipped prior</dd>` : '') +
      `<dt>recommendation</dt><dd class="v-mono">${esc(D.rec || 'n/a')}</dd>` +
      `<dt>rule</dt><dd>${esc(rule.definition || rule.name || 'n/a')}</dd>` +
      `<dt>${esc(refShort)}</dt><dd>${ref ? esc(ref.label) + (num(ref.support) ? ', ' + V.runs(ref.support) : '') : 'none'}${refWhy ? '<br><span class="m">' + esc(refWhy) + '</span>' : ''}</dd>` +
      `<dt>rescue</dt><dd>${RESCUE ? (priced ? `${usd(RESCUE.usd)}: ${esc(RESCUE.basis || RESCUE.kind)}${RESCUE.of ? ' (' + esc(RESCUE.of) + ')' : ''}` : 'none priced') : 'n/a'}</dd>` +
      `<dt>options</dt><dd>${Object.entries(origins).map(([k, v]) => `${v} ${esc(ORIGIN[k] || k)}`).join(', ')}</dd>` +
      `<dt>ranges</dt><dd>80 percent</dd></dl>` +
      (D.message ? `<h3>The recommendation, as printed</h3><p class="v-note" id="message">${esc(D.message)}</p>` : ''));
    return out;
  }
  function exploreNone() {
    const p = EX.best_value;
    return p && p.none != null ? esc(typeof p.none === 'string' ? p.none : 'no candidate qualifies') : 'nothing to try';
  }
  function betList() {
    return ['best_value', 'max_gain'].map(k => {
      const p = pickFor(k), title = k === 'best_value' ? 'best value' : 'biggest gain';
      if (!p || (EX[k] && EX[k].same_as) || !p.price || !p.price.usd || !p.gain_per_run || !num(p.gain_per_run.usd) || p.gain_per_run.usd <= 0) return { kind: k, title, none: true };
      const pr = p.price.usd, gain = p.gain_per_run.usd, id = pickCid(p);
      return { kind: k, title, name: title, label: labelOf(id), id, gain, price: pr, payback: num(p.payback_runs) ? p.payback_runs : pr.mean / gain,
        payback_lo: num(pr.lo) ? pr.lo / gain : null, payback_hi: num(pr.hi) ? pr.hi / gain : null };
    });
  }
  function betCard(kind) {
    const e = EX[kind], title = kind === 'best_value' ? 'Best value' : 'Biggest gain';
    const note = kind === 'best_value' ? 'The lowest payback: most saving per dollar spent now.' : 'The most saving per future run, whatever the price (within the budget cap).';
    let body;
    if (!e) body = '<p class="v-note">Not reported.</p>';
    else if (e.same_as) body = `<p class="v-note">Same candidate as ${esc(e.same_as === 'best_value' ? 'best value' : e.same_as)}: one run serves both.</p>`;
    else if (e.none != null || !pickOf(e)) body = `<p class="v-note">${esc(typeof e.none === 'string' ? e.none : 'No candidate qualifies.')}</p>`;
    else {
      const p = pickOf(e), id = pickCid(p), r = R.get(id) || row(id), gg = r ? r.g : null;
      body = (e.paused ? `<p class="v-warn"><i>!</i><span>Paused: ${esc(e.paused)}. It would have been:</span></p>` : '') +
        `<p class="v-note"><b>${esc(labelOf(id))}</b>${p.auto_ok ? ' <span class="v-mk try" title="payback is below explore.auto_payback_runs">auto ok</span>' : ''}</p>` +
        `<dl class="v-kv"><dt>${esc(gWords)}</dt><dd>${gg && num(gg.mean) ? pct(gg.mean) + (num(gg.lo) ? ` <span class="m">(${pct(gg.lo)} to ${pct(gg.hi)})</span>` : '') : 'n/a'}</dd>` +
        `<dt>chance it beats the recommended pick</dt><dd>${pct(p.p_beats_goal)}</dd>` +
        `<dd class="v-wide">Not the ${esc(gWords)}: the chance that, once tried, this workflow turns out cheaper per accepted result than the recommended pick.</dd>` +
        `<dt>price now</dt><dd>${p.price && p.price.usd ? usd(p.price.usd.mean) + ` <span class="m">(${rng(p.price.usd, usd)})</span>` + tail(p.price.usd, p.price.tokens) : 'n/a'}</dd>` +
        `<dt>trying it once</dt><dd>${esc(savingText(p.gain_per_run))}</dd>` +
        `<dt>pays for itself after</dt><dd>${esc(paybackText(p.payback_runs))}</dd></dl>` +
        V.copyBtn(copyText(id), COPY);
    }
    return `<div class="v-bet" data-kind="${kind}"><h3>${title}${paused(kind) ? ' (paused)' : ''}</h3><p class="v-note">${note}</p>${body}</div>`;
  }

  if (!pick) return;

  // ------------------------------------------------------------ the options table
  let showAll = false, openRow = null;
  function table() {
    const body = document.querySelector('#tall tbody');
    const all = sorted(), list = showAll ? all : all.filter((r, i) => i < 12 || marks(r.id).length);
    const ORIGIN = { catalog: 'catalog', edit: 'edit of the reference', recorded: 'recorded', front: 'found by the search', thompson: 'found in a search draw', polish: 'near the pick' };
    body.innerHTML = list.map(r => {
      const cls = r.id === goalId ? 'pick' : r.id === refId ? 'ref' : '';
      return `<tr class="v-opt ${cls}" data-cfg="${esc(r.id)}" tabindex="0"><td><span class="v-row-name">${D.graphs && D.graphs[r.id] ? V.miniGraph(D.graphs[r.id]) : ''}<span class="lbl">${esc(r.label)}<span class="sub">${markHtml(r.id)}${esc(ORIGIN[r.origin] || r.origin)}</span></span></span></td>` +
        `<td class="nw">${r.g && num(r.g.mean) ? V.chanceBar(r.g, cls) + pct(r.g.mean) : 'n/a'}<span class="sub">${rng(r.g, pct)}</span></td>` +
        (PAW ? `<td class="nw v-paw">${r.paw ? V.chanceBar(r.paw, cls) + pct(r.paw.mean) : 'n/a'}<span class="sub">${rng(r.paw, pct)}</span></td>` : '') +
        `<td class="num">${r.run ? usd(r.run.mean) : 'n/a'}<span class="sub">${median(r)}</span>${tail(r.run)}</td>` +
        `<td class="num">${num(r.rescue) ? usd(r.rescue) : 'n/a'}</td>` +
        `<td class="num"><b>${usd(r.ell.mean)}</b><span class="sub">${rng(r.ell, usd)}</span>${tail(r.ell)}</td>` +
        `<td class="num">${supportText(r.support)}</td></tr>` + (openRow === r.id ? detailRow(r) : '');
    }).join('');
    body.querySelectorAll('.v-opt-graph').forEach(el => { if (D.graphs && D.graphs[el.dataset.cfg]) V.graph(el, D.graphs[el.dataset.cfg]); });
  }
  function detailRow(r) {
    const diff = ((r.c && r.c.diff_vs_usual) || []).filter(Boolean);
    return `<tr class="v-opt-detail" data-for="${esc(r.id)}"><td colspan="${COLS.length}"><div class="v-gfull v-opt-graph" data-cfg="${esc(r.id)}"></div>` +
      (diff.length ? `<p class="v-note">Against ${esc(refShort)}: ${diff.map(esc).join('; ')}.</p>` : '') +
      (num(r.sr) ? `<p class="v-note v-sr">Score estimate: a ${pct(r.sr)} chance to reach ${esc(fmt.x(target.target))}, from too few scores; no total uses it.</p>` : '') +
      V.copyBtn(copyText(r.id), COPY) + `</td></tr>`;
  }
  const toggleRow = tr => { openRow = openRow === tr.dataset.cfg ? null : tr.dataset.cfg; table(); };
  const sortBy = th => {
    const col = COLS.find(c => c.k === th.dataset.sort);
    if (!col) return;
    sortDir = sortKey === col.k ? -sortDir : col.dir; sortKey = col.k;
    tall.querySelectorAll('th.v-sort').forEach(x => x.setAttribute('aria-sort', x.dataset.sort === sortKey ? (sortDir > 0 ? 'ascending' : 'descending') : 'none'));
    table();
  };
  const tall = document.getElementById('tall');
  tall.addEventListener('click', e => {
    if (e.target.closest('[data-copy]') || e.target.closest('.v-info')) return;
    const th = e.target.closest('th.v-sort'); if (th) { sortBy(th); return; }
    const tr = e.target.closest('tr.v-opt'); if (tr) toggleRow(tr);
  });
  tall.addEventListener('keydown', e => {
    if (e.key !== 'Enter' || e.target.closest('.v-info')) return;
    const th = e.target.closest('th.v-sort'); if (th) { sortBy(th); return; }
    const tr = e.target.closest('tr.v-opt'); if (tr) toggleRow(tr);
  });
  // P5: the info icons show their text on hover, and on a tap or click (again to hide).
  let infoOpen = null;
  const infoAt = b => { const r = b.getBoundingClientRect(); LM.tip(esc(INFO[b.dataset.info]), r.left + r.width / 2, r.bottom); };
  document.addEventListener('mouseover', e => { const b = e.target.closest && e.target.closest('.v-info'); if (b) infoAt(b); });
  document.addEventListener('mouseout', e => { const b = e.target.closest && e.target.closest('.v-info'); if (b && b !== infoOpen) LM.tip(null); });
  document.addEventListener('click', e => {
    const b = e.target.closest && e.target.closest('.v-info');
    if (b) { e.preventDefault(); if (infoOpen === b) { infoOpen = null; LM.tip(null); } else { infoOpen = b; infoAt(b); } return; }
    if (infoOpen) { infoOpen = null; LM.tip(null); }
  });
  const more = document.getElementById('more');
  if (more) more.addEventListener('click', () => { showAll = !showAll; more.textContent = showAll ? 'Show fewer' : `Show all ${ROWS.length}`; table(); });
  table();

  // ------------------------------------------------------------ charts (the forest and payback draw when their section opens)
  const tipFor = r => `<b>${esc(r.label)}</b><br>${usd(r.ell.mean)} per accepted result (${rng(r.ell, usd)})${pulledUp(r.ell) ? '; ' + esc(TAIL_NOTE) : ''}` +
    `<br>${r.run ? usd(r.run.mean) + ' a run' : ''}${r.g && num(r.g.mean) ? ', ' + pct(r.g.mean) + ' ' + esc(gWords) : ''}<br>${r.support ? V.runs(r.support) + ' behind it' : 'no runs of this workflow yet'}`;
  function drawMain() {
    const pg = document.getElementById('pg'); pg.innerHTML = '';
    const graph = D.graphs && D.graphs[goalId];
    if (graph) {
      V.graph(pg, graph);
      const nodes = graph.nodes || [], wide = nodes.some(n => n.kind === 'piece' && (n.width || 1) > 1), loops = (graph.gates || []).some(gt => gt && gt.on_fail);
      document.getElementById('pgcap').textContent = 'Each box is an agent with its model and effort and its share of the predicted cost per run.' +
        (wide ? ' A piece of width n is drawn as n workers.' : '') + (loops ? ' The dashed loop sends failed work back.' : '');
    }
    const host = document.getElementById('three');
    ivRows(host, THREE.map(t => ivRow(t.r, { key: t.r.id, label: t.label, sub: t.r.label, mark: t.mark })), { rowH: 54, labelW: Math.min(230, Math.max(128, (host.clientWidth || 600) * 0.3)) });
    drawCC();
  }

  // ------------------------------------------------------------ intervals (P6, P9): a dot for the mean, lines for ranges
  // `bands` (21M, 0.2.1) holds central intervals by level; without a level, the 80% range is the lo/hi pair. Wider
  // ranges get thinner lines, so a 0 to 100% range stays light and the dots stay readable.
  const BANDKEY = { ell: 'cost_per_accepted_usd', run: 'run_cost_usd', g: 'chance' };
  const W6 = { '80': 2.6, '90': 1.3, '95': 0.6 }, W9 = { '50': 2.2, '80': 1, '90': 0.5 };
  function levels(r, k, want) {
    const b = r.bands && r.bands[BANDKEY[k]], x = r[k], out = [];
    want.forEach(lv => {
      const iv = b && Array.isArray(b[lv]) && num(b[lv][0]) && num(b[lv][1]) ? b[lv] : lv === '80' && x && num(x.lo) && num(x.hi) ? [x.lo, x.hi] : null;
      if (iv) out.push([lv, iv[0], iv[1]]);
    });
    return out;
  }
  const lvText = (r, k, f, want) => { const l = levels(r, k, want || ['50', '80', '90']); return l.length ? ` <span class="m">(${l.map(([lv, a, b]) => `${lv}%: ${f(a)} to ${f(b)}`).join('; ')})</span>` : ''; };
  const ivRow = (r, o) => ({ ...o, mean: r.ell.mean, levels: levels(r, 'ell', ['95', '90', '80']), tip: tipFor(r) });
  function ivRows(host, rows, o) {
    host.innerHTML = '';
    const rowH = o.rowH || 24, cw = host.clientWidth || 600;
    const labelW = o.labelW != null ? o.labelW : Math.min(270, Math.max(110, cw * 0.36));
    const dom = V.logDomain(rows.flatMap(r => [r.mean, ...r.levels.flatMap(l => [l[1], l[2]])]));
    const top = 22, H = top + rows.length * rowH + 10, { svg, W } = V.frame(host, H);
    const sx = V.log(dom[0], dom[1], labelW + 8, W - 14);
    V.axisX(svg, sx, top, V.logTicks(sx.d[0], sx.d[1]), V.usd0, [top, H - 8], true);
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2 + 2, g = V.el('g', { class: 'v-row v-ivrow' + (r.mark ? ' ' + r.mark : ''), 'data-key': r.key }, svg);
      const hit = V.el('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, class: 'v-hit' }, g);
      if (r.label) V.el('text', { x: 4, y: y + (r.sub ? -2 : 4), class: 'v-rl' }, g, V.clip(r.label, Math.floor(labelW / 6.6)));
      if (r.sub) V.el('text', { x: 4, y: y + 11, class: 'v-rs' }, g, V.clip(r.sub, Math.floor(labelW / 5.6)));
      r.levels.forEach(([lv, lo, hi]) => V.el('line', { x1: sx(lo), x2: sx(hi), y1: y, y2: y, class: 'v-ivl', 'data-level': lv, 'stroke-width': W6[lv] }, g));
      if (num(r.mean)) V.el('circle', { cx: sx(r.mean), cy: y, r: r.mark ? 5 : 4, class: 'v-ivdot' }, g);
      if (r.tip) V.hover(hit, r.tip);
    });
  }

  // ------------------------------------------------------------ chance against cost (P9)
  let xMode = 'ell';
  const ccTip = (o, r) => `<b>option ${o.n}: ${esc(o.label || r.label)}</b><br>${esc(gWords)}: ${r.g && num(r.g.mean) ? pct(r.g.mean) : 'n/a'}${lvText(r, 'g', pct)}` +
    (r.paw ? `<br>${pct(r.paw.mean)} within ${esc(K)} attempts` : '') +
    `<br>cost per accepted result: ${usd(r.ell && r.ell.mean)}${lvText(r, 'ell', usd)}<br>run cost: ${r.run ? usd(r.run.mean) : 'n/a'}${lvText(r, 'run', usd)}` +
    (num(r.support) ? `<br>${r.support ? V.runs(r.support) + ' behind it' : 'no runs of this workflow yet'}` : '');
  function drawCC() {
    const host = document.getElementById('ccplot');
    if (!host) return;
    host.innerHTML = '';
    const k = xMode, LV = ['90', '80', '50'], xv = r => r[k] && num(r[k].mean) ? r[k].mean : null;
    const pts = chartOpts().filter(({ r }) => num(xv(r)) && r.g && num(r.g.mean));
    const bg = ROWS.filter(r => !OPT[r.id] && num(xv(r)) && r.g && num(r.g.mean)).slice(0, CAP);
    const dom = V.logDomain([...pts.flatMap(({ r }) => [xv(r), ...levels(r, k, LV).flatMap(l => [l[1], l[2]])]), ...bg.map(xv)]);
    const H = Math.round(Math.max(260, Math.min(380, (host.clientWidth || 600) * 0.5))), { svg, W } = V.frame(host, H);
    const L = 44, RR = 16, T = 12, B = 38, sx = V.log(dom[0], dom[1], L, W - RR), sy = V.lin(0, 1, H - B, T), el = V.el;
    [0, 0.25, 0.5, 0.75, 1].forEach(v => { el('line', { x1: L, x2: W - RR, y1: sy(v), y2: sy(v), class: 'v-grid' }, svg); el('text', { x: L - 6, y: sy(v) + 4, 'text-anchor': 'end', class: 'v-tick' }, svg, pct(v)); });
    V.logTicks(sx.d[0], sx.d[1]).forEach(t => { el('line', { x1: sx(t), x2: sx(t), y1: T, y2: H - B, class: 'v-grid' }, svg); el('text', { x: sx(t), y: H - B + 15, 'text-anchor': 'middle', class: 'v-tick' }, svg, V.usd0(t)); });
    el('text', { x: (L + W - RR) / 2, y: H - 4, 'text-anchor': 'middle', class: 'v-axis-t' }, svg, (k === 'ell' ? 'cost per accepted result' : 'run cost') + ', log scale');
    bg.forEach(r => { const c = el('circle', { cx: sx(xv(r)), cy: sy(r.g.mean), r: 2.2, class: 'v-ccbg' }, svg); V.hover(c, `<b>${esc(r.label)}</b><br>${pct(r.g.mean)} ${esc(gWords)}<br>${usd(xv(r))} ${k === 'ell' ? 'per accepted result' : 'a run'}<br>not a numbered option: copy it from the table`); });
    const placed = [];
    pts.forEach(({ o, r }) => {
      const x = sx(xv(r)), y = sy(r.g.mean), g = el('g', { class: 'v-ccpt' + (r.id === goalId ? ' pick' : ''), 'data-opt': o.n, 'data-key': r.id }, svg);
      levels(r, k, LV).forEach(([lv, lo, hi]) => el('line', { x1: sx(lo), x2: sx(hi), y1: y, y2: y, class: 'v-ccl', 'data-level': lv, 'data-axis': 'x', 'stroke-width': W9[lv] }, g));
      levels(r, 'g', LV).forEach(([lv, lo, hi]) => el('line', { x1: x, x2: x, y1: sy(V.clamp01(lo)), y2: sy(V.clamp01(hi)), class: 'v-ccl', 'data-level': lv, 'data-axis': 'y', 'stroke-width': W9[lv] }, g));
      el('circle', { cx: x, cy: y, r: 5.5, class: 'v-ccdot' }, g);
      V.hover(el('circle', { cx: x, cy: y, r: 13, class: 'v-hit' }, g), ccTip(o, r));
      placed.push({ o, x, y, g });
    });
    // Option numbers beside their points, moved round the point until they clear the other labels and points.
    const boxes = [], OFF = [[9, -6, 'start'], [9, 14, 'start'], [-9, -6, 'end'], [-9, 14, 'end'], [0, -11, 'middle'], [0, 21, 'middle']];
    const hits = (b) => boxes.some(q => b.x0 < q.x1 && q.x0 < b.x1 && b.y0 < q.y1 && q.y0 < b.y1) || placed.some(p => p.x > b.x0 - 5 && p.x < b.x1 + 5 && p.y > b.y0 - 5 && p.y < b.y1 + 5);
    placed.forEach(p => {
      const t = String(p.o.n), w = 7 * t.length + 2;
      const box = ([dx, dy, a]) => { const x0 = a === 'start' ? p.x + dx : a === 'end' ? p.x + dx - w : p.x - w / 2; return { x0, x1: x0 + w, y0: p.y + dy - 10, y1: p.y + dy + 1 }; };
      const at = OFF.find(off => !hits(box(off))) || OFF[0];
      boxes.push(box(at));
      el('text', { x: p.x + at[0], y: p.y + at[1], 'text-anchor': at[2], class: 'v-cclbl' }, p.g, t);
    });
  }
  const CAP = 200;
  document.querySelectorAll('#cc .v-seg button').forEach(b => b.addEventListener('click', () => {
    xMode = b.dataset.x;
    document.querySelectorAll('#cc .v-seg button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    drawCC();
  }));
  const draw = {
    'd-wide': () => {
      const list = ROWS.slice(0, 16);
      [pick, ref].forEach(r => { if (r && !list.includes(r)) list.push(r); });
      ivRows(document.getElementById('forest'), list.map(r => ivRow(r, { key: r.id, label: r.label, mark: r.id === goalId ? 'pick' : r.id === refId ? 'ref' : '' })), {});
    },
    'd-try': () => { const host = document.getElementById('pay'), bets = betList().filter(b => !b.none); if (host && bets.length) V.payback(host, bets, { height: 230 }); },
  };
  document.querySelectorAll('details.v-more').forEach(d => d.addEventListener('toggle', () => { if (d.open && draw[d.id]) draw[d.id](); }));
  drawMain();
  if (location.hash === '#open') document.querySelectorAll('details.v-more').forEach(d => { d.open = true; });
  V.onResize(() => { drawMain(); Object.keys(draw).forEach(k => { const d = document.getElementById(k); if (d && d.open) draw[k](); }); });
})();
