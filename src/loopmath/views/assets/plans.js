// Planning page (lane 2E, 0.2; D119 Z3 direction C in the paper look): `loopmath recommend --html`. It leads with one
// decision, the pick, then puts the rest in details sections whose closed line still answers the question.
// Renders `loopmath.view.plans/1` from the embedded object only; recommend/1 objects (no `numbers`) still render.
(() => {
  const V = LM.Viz, { esc, num, fmt, shq, TAIL_NOTE, pulledUp } = LM;
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
    return { id, c, label: labelOf(id), g: gg, sr: scoreReach(p, nb), run, ell, rescue, support: num(p.support) ? p.support : null, origin: c.origin || '', tokens: p.cost && p.cost.tokens };
  }
  const ROWS = [...BY.keys()].map(row).filter(r => r && r.ell && num(r.ell.mean)).sort((a, b) => a.ell.mean - b.ell.mean);
  const R = new Map(ROWS.map(r => [r.id, r]));
  const pick = R.get(goalId) || null, ref = refId ? R.get(refId) || null : null;

  // ------------------------------------------------------------ marks and commands
  const tryKinds = id => ['best_value', 'max_gain'].filter(k => { const p = pickFor(k); return p && pickCid(p) === id; });
  const TITLE = { best_value: 'best value to try', max_gain: 'biggest gain to try' };
  function marks(id) {
    const m = [];
    if (id === goalId) m.push(['pick', 'recommended' + (D.goal && num(D.goal.level) ? ` (${D.goal.level}% row)` : '')]);
    if (D.default_pick && D.default_pick.config === id && id !== goalId) m.push(['other', 'default pick']);
    if (id === refId) m.push(['ref', refShort]);
    tryKinds(id).forEach(k => m.push(['try', TITLE[k] + (paused(k) ? ' (paused)' : '')]));
    return m;
  }
  const markHtml = id => marks(id).map(([k, t]) => `<span class="v-mk ${k}">${esc(t)}</span>`).join('');
  const sourceFor = id => id === refId && isUsual ? 'usual' : id !== goalId && tryKinds(id).length ? 'exploration' : 'alternative';
  function command(id, source, extra) {
    const parts = ['loopmath run start'];
    if (task.type) parts.push('--type', shq(task.type));
    if (task.repo) parts.push('--repo', shq(task.repo));
    if (task.subtype) parts.push('--subtype', shq(task.subtype));
    if (task.title) parts.push('--title', shq(task.title));
    // Lane 2C (0.2): the run's time budget goes as --horizon, and it is the one the prediction used, given or filled
    // in from the fit, as `run start --rec` stores it (rec_features), so the run is recorded under the budget it was priced for.
    const hz = task.horizon ? (!task.horizon.from || task.horizon.from === 'none' ? null : num(task.horizon.seconds) && task.horizon.seconds > 0 ? String(task.horizon.seconds) : 'none')
      : (task.features || {}).horizon_s != null ? String(task.features.horizon_s) : null;
    Object.entries(task.features || {}).filter(([k]) => k !== 'horizon_s').forEach(([k, v]) => parts.push('--feature', shq(`${k}=${v}`)));
    if (hz) parts.push('--horizon', shq(hz));
    if (task.base_commit) parts.push('--base-commit', shq(task.base_commit));
    parts.push('--config', shq(id), '--source', source || sourceFor(id));
    if (D.rec) parts.push('--rec', shq(D.rec));
    return parts.concat(extra || []).join(' ');
  }
  // Lane 2D (0.2): an option the stored recommendation names in `choices` starts with `run start --rec REC
  // --choice KEY`, which takes the task, configuration, source and rule from the recommendation. Every other
  // option keeps the full command above.
  const CHOICE = {};
  (D.choices || []).forEach(c => { if (c && c.key && c.key !== 'pair' && c.config && !(c.config in CHOICE)) CHOICE[c.config] = c.key; });
  const pairChoice = (D.choices || []).find(c => c && c.key === 'pair' && (c.members || []).length === 2);
  const byChoice = key => `loopmath run start --rec ${shq(D.rec)} --choice ${shq(key)}`;
  const startCmd = id => D.rec && CHOICE[id] ? byChoice(CHOICE[id]) : command(id);

  // ------------------------------------------------------------ text helpers
  const usd = V.usd, pct = V.pct;
  const rng = (x, f) => x && num(x.lo) && num(x.hi) ? `${f(x.lo)} to ${f(x.hi)}` : '';
  const tail = (...xs) => xs.some(pulledUp) ? ` <span class="v-tail">${esc(TAIL_NOTE)}</span>` : '';
  const median = r => r && r.run && num(r.run.median) ? `median ${usd(r.run.median)}${r.run.median_basis && r.run.median_basis !== 'draws' ? ' (approx.)' : ''}` : '';
  const supportText = n => !num(n) ? 'n/a' : n === 0 ? 'none' : fmt.int(n);
  // D96/D102: the look-ahead value in dollars, as lane 6's recommend message words it.
  const savingText = gp => !gp || !num(gp.usd) ? 'n/a' : `expected to save about ${gp.usd > 0 && gp.usd < 0.01 ? 'under $0.01' : '$' + gp.usd.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} per future similar run`;
  const paybackText = n => { if (!num(n)) return 'n/a'; const k = Math.max(1, Math.round(n)); return `about ${k} similar run${k === 1 ? '' : 's'}`; };
  // D119 Z2: lane 2A's sentence, shown verbatim; the page checks only that it is set.
  const strategy = (D.goal && D.goal.strategy) || ((D.choices || []).find(c => c && c.key === 'goal') || {}).strategy || null;
  const strategyText = strategy && typeof strategy.text === 'string' && strategy.text ? strategy.text : null;

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
  h += pick ? hero() + threeBlock() + details() : `<section class="v-hero"><p class="v-kicker">Recommended</p><h1>No pick in this recommendation.</h1><p class="v-lede">${esc(D.message || '')}</p></section>`;
  h += `<p class="v-foot">loopmath recommend ${esc(D.rec || '')}, fit ${esc(fit.id || 'n/a')}${D.generated_at ? ', page written ' + esc(fmt.dt(D.generated_at)) : ''}. Ranges are 80%. The page runs nothing: copy a command to start a run.</p></div>`;
  app.innerHTML = h;

  function hero() {
    const [gName, gSettings] = splitLabel(pick.id);
    let s = `<section class="v-hero" id="pick"><p class="v-kicker">Recommended${D.goal && num(D.goal.level) ? `: the ${esc(D.goal.level)}% row of the curve` : ''}</p>` +
      `<h1>Run ${esc(gName)}${gSettings ? ' with ' + esc(gSettings) : ''}.</h1>` +
      (strategyText ? `<p class="v-strategy" id="strategy">${esc(strategyText)}</p>` : '') +
      `<div class="v-gfull" id="pg"></div><p class="v-gcap" id="pgcap"></p>` +
      `<div class="v-nums" id="nums">` +
      `<div class="lead"><div class="n">${usd(pick.ell.mean)}</div><div class="l">per accepted result</div><div class="r">${rng(pick.ell, usd)}${tail(pick.ell)}</div></div>` +
      `<div><div class="n">${pick.g && num(pick.g.mean) ? pct(pick.g.mean) : 'n/a'}</div><div class="l">${esc(gShort)}</div><div class="r">${rng(pick.g, pct)}</div></div>` +
      `<div><div class="n">${pick.run ? usd(pick.run.mean) : 'n/a'}</div><div class="l">a run</div><div class="r">${median(pick)}${tail(pick.run)}</div></div></div>`;
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
    return s + `<p class="v-vs" id="vs">${vs.trim()}</p>` + V.copyBtn(startCmd(pick.id), 'copy command') + `</section>`;
  }

  function threeBlock() {
    let s = `<section class="v-block" id="ways"><h2>${THREE.length === 3 ? 'Three' : THREE.length === 2 ? 'Two' : 'One'} way${THREE.length === 1 ? '' : 's'} to run this task</h2>` +
      `<p class="v-sub">Cost per accepted result, log scale. The shape spans the 80% range; the tick is the mean.</p><div id="three"></div>`;
    if (D.pair && (D.pair.members || []).length === 2) {
      const [a, b] = D.pair.members, bet = pickFor(D.pair.explore_pick || 'best_value');
      s += `<p class="v-pairline" id="pairline">Or run the pick and <b>${esc(labelOf(b))}</b> side by side, then let a blinded referee choose.` +
        (bet && bet.price && bet.price.usd ? ` It costs ${usd(bet.price.usd.mean)} more now; trying it once is ${esc(savingText(bet.gain_per_run))}; there is a ${pct(bet.p_beats_goal)} chance it beats the recommended pick.` : '') + `</p>` +
        V.copyBtn(D.rec && pairChoice && pairChoice.members[0] === a && pairChoice.members[1] === b ? byChoice('pair')
          : command(a, sourceFor(a), ['--new-slate']) + '\n' + command(b, 'exploration', ['--slate', 'SLT']), 'copy both');
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
      eq.push(['rescue', usd(RESCUE.usd), `${esc(RESCUE.basis || RESCUE.kind)}${RESCUE.of ? ': ' + esc(RESCUE.of) : ''}`]);
      eq.push(['expected rescue', usd(pick.rescue), `${num(miss) ? pct(miss) : 'n/a'} x ${usd(RESCUE.usd)}`]);
    }
    const tot = ['per accepted result', usd(pick.ell.mean), `80% range ${rng(pick.ell, usd)}${pulledUp(pick.ell) ? '; ' + esc(TAIL_NOTE) : ''}`];
    out += sec('d-math', 'The arithmetic', priced ? `${pick.run ? usd(pick.run.mean) : 'n/a'} a run + ${num(miss) ? pct(miss) : 'n/a'} x ${usd(RESCUE.usd)} rescue = ${usd(pick.ell.mean)}` : `${usd(pick.ell.mean)} per accepted result; no rescue priced`,
      `<div class="v-math"><div class="eq">${eq.map(e => e.map(x => `<span>${x}</span>`).join('')).join('')}${tot.map(x => `<span class="tot">${x}</span>`).join('')}</div></div>` +
      `<p class="v-note">Cost per accepted result = cost per run + chance of a miss x the rescue. Parts are rounded to the cent, so they may not add up exactly.${FB ? ' ' + esc(fbNote) : ''}</p>`);
    // every option
    out += sec('d-all', `All ${ROWS.length} options`, ROWS.length ? `cheapest per accepted result first; ${esc(ROWS[0].label)} leads at ${usd(ROWS[0].ell.mean)}` : 'none',
      `<div class="v-scroll"><table class="v-t" id="tall"><thead><tr><th>workflow</th><th>${esc(gShort)}</th><th class="num">cost per run</th><th class="num">expected rescue</th><th class="num">cost per accepted result</th><th class="num">runs behind it</th></tr></thead><tbody></tbody></table></div>` +
      (ROWS.length > 12 ? `<button type="button" class="v-btn" id="more">Show all ${ROWS.length}</button>` : '') +
      `<p class="v-note">Click a row for its graph and command. Ranges are 80%.</p>`);
    // how wide
    const widest = ROWS.filter(r => r.g && num(r.g.lo) && num(r.g.hi) && r.g.hi - r.g.lo > 0.8).length;
    out += sec('d-wide', 'How wide the estimates are', `${widest} of ${ROWS.length} chances span more than 80 points; the pick runs from ${rng(pick.ell, usd) || 'n/a'}`,
      `<div class="v-legend"><span><span class="sw" style="background:var(--accent)"></span>recommended</span><span><span class="dt no"></span>${esc(refShort)}</span><span><span class="sw" style="background:var(--ink2);opacity:.5"></span>others; each range fades to its ends</span></div><div id="forest"></div>`);
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
        V.copyBtn(command(id, 'exploration'), 'copy command');
    }
    return `<div class="v-bet" data-kind="${kind}"><h3>${title}${paused(kind) ? ' (paused)' : ''}</h3><p class="v-note">${note}</p>${body}</div>`;
  }

  if (!pick) return;

  // ------------------------------------------------------------ the options table
  let showAll = false, openRow = null;
  function table() {
    const body = document.querySelector('#tall tbody');
    const list = showAll ? ROWS : ROWS.filter((r, i) => i < 12 || marks(r.id).length);
    const ORIGIN = { catalog: 'catalog', edit: 'edit of the reference', recorded: 'recorded', front: 'found by the search', thompson: 'found in a search draw', polish: 'near the pick' };
    body.innerHTML = list.map(r => {
      const cls = r.id === goalId ? 'pick' : r.id === refId ? 'ref' : '';
      return `<tr class="v-opt ${cls}" data-cfg="${esc(r.id)}" tabindex="0"><td><span class="v-row-name">${D.graphs && D.graphs[r.id] ? V.miniGraph(D.graphs[r.id]) : ''}<span class="lbl">${esc(r.label)}<span class="sub">${markHtml(r.id)}${esc(ORIGIN[r.origin] || r.origin)}</span></span></span></td>` +
        `<td class="nw">${r.g && num(r.g.mean) ? V.chanceBar(r.g, cls) + pct(r.g.mean) : 'n/a'}<span class="sub">${rng(r.g, pct)}</span></td>` +
        `<td class="num">${r.run ? usd(r.run.mean) : 'n/a'}<span class="sub">${median(r)}</span>${tail(r.run)}</td>` +
        `<td class="num">${num(r.rescue) ? usd(r.rescue) : 'n/a'}</td>` +
        `<td class="num"><b>${usd(r.ell.mean)}</b><span class="sub">${rng(r.ell, usd)}</span>${tail(r.ell)}</td>` +
        `<td class="num">${supportText(r.support)}</td></tr>` + (openRow === r.id ? detailRow(r) : '');
    }).join('');
    body.querySelectorAll('.v-opt-graph').forEach(el => { if (D.graphs && D.graphs[el.dataset.cfg]) V.graph(el, D.graphs[el.dataset.cfg]); });
  }
  function detailRow(r) {
    const diff = ((r.c && r.c.diff_vs_usual) || []).filter(Boolean);
    return `<tr class="v-opt-detail" data-for="${esc(r.id)}"><td colspan="6"><div class="v-gfull v-opt-graph" data-cfg="${esc(r.id)}"></div>` +
      (diff.length ? `<p class="v-note">Against ${esc(refShort)}: ${diff.map(esc).join('; ')}.</p>` : '') +
      (num(r.sr) ? `<p class="v-note v-sr">Score estimate: a ${pct(r.sr)} chance to reach ${esc(fmt.x(target.target))}, from too few scores; no total uses it.</p>` : '') +
      V.copyBtn(startCmd(r.id), 'copy command') + `</td></tr>`;
  }
  const toggleRow = tr => { openRow = openRow === tr.dataset.cfg ? null : tr.dataset.cfg; table(); };
  const tall = document.getElementById('tall');
  tall.addEventListener('click', e => { if (e.target.closest('[data-copy]')) return; const tr = e.target.closest('tr.v-opt'); if (tr) toggleRow(tr); });
  tall.addEventListener('keydown', e => { if (e.key !== 'Enter') return; const tr = e.target.closest('tr.v-opt'); if (tr) toggleRow(tr); });
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
    V.ridge(host, THREE.map(t => ({ key: t.r.id, label: t.label, sub: t.r.label, x: t.r.ell, mark: t.mark, tip: tipFor(t.r) })),
      { domain: V.logDomain(THREE.flatMap(t => [t.r.ell.lo, t.r.ell.hi, t.r.ell.mean])), log: true, fmt: V.usd0, rowH: 54, labelW: Math.min(230, Math.max(128, (host.clientWidth || 600) * 0.3)) });
  }
  const draw = {
    'd-wide': () => {
      const list = ROWS.slice(0, 16);
      [pick, ref].forEach(r => { if (r && !list.includes(r)) list.push(r); });
      V.forest(document.getElementById('forest'), list.map(r => ({ key: r.id, label: r.label, x: r.ell, mark: r.id === goalId ? 'pick' : r.id === refId ? 'ref' : '', tip: tipFor(r) })),
        { domain: V.logDomain(list.flatMap(r => [r.ell.lo, r.ell.hi, r.ell.mean])), fmt: V.usd0 });
    },
    'd-try': () => { const host = document.getElementById('pay'), bets = betList().filter(b => !b.none); if (host && bets.length) V.payback(host, bets, { height: 230 }); },
  };
  document.querySelectorAll('details.v-more').forEach(d => d.addEventListener('toggle', () => { if (d.open && draw[d.id]) draw[d.id](); }));
  drawMain();
  if (location.hash === '#open') document.querySelectorAll('details.v-more').forEach(d => { d.open = true; });
  V.onResize(() => { drawMain(); Object.keys(draw).forEach(k => { const d = document.getElementById(k); if (d && d.open) draw[k](); }); });
})();
