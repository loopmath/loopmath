// The results page (lane 2E, 0.2; D119 Z3, direction A): the questions after a fit, in order, in one scroll.
// It draws from the embedded loopmath.view.posterior/1 object on the shared viz.js; estimates.js draws the three
// detail blocks (every estimate by level, one workflow in full, the data behind the fit) into their hosts here.
(() => {
  const V = LM.Viz, { esc, num, fmt, TAIL_NOTE, pulledUp } = LM, { usd, pct, score } = V;
  const D = LM.data(), app = document.getElementById('app');
  const R = D.results || {}, T = R.target || null, RES = R.rescue || null, RW = R.workflows || {};
  const SN = D.score_name || (T && T.score) || null;
  const HEADS = D.heads || {}, task = D.task || {}, fit = D.fit || {}, data = D.data || {};
  const better = T ? T.better : ((HEADS['score:' + SN] || {}).better || 'higher');
  const EFFORTS = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'];
  const RUNS = (D.runs || []).filter(r => r && r.config);
  const mean = (x, d) => x && num(x.mean) ? x.mean : d;
  const rng = (x, f) => x && num(x.lo) && num(x.hi) ? `${f(x.lo)} to ${f(x.hi)}` : '';
  const tail = (...xs) => xs.some(pulledUp) ? ` <span class="v-tail">${esc(TAIL_NOTE)}</span>` : '';
  const tailWord = (...xs) => xs.some(pulledUp) ? '; ' + TAIL_NOTE : '';
  const beats = (a, b) => better === 'lower' ? a < b : a > b;
  const bestWord = better === 'lower' ? 'lowest' : 'highest';  // the best score in the target's direction
  const runsOf = cfgs => RUNS.filter(r => cfgs.indexOf(r.config) >= 0);
  const tgt = T ? score(T.target) : '';
  const scoreWord = SN ? SN.replace(/_/g, ' ') : '';
  const taskCmd = `loopmath posterior --type ${LM.shq(task.type || 'feature')} --repo ${LM.shq(task.repo || 'REPO')}`;

  // ------------------------------------------------------------ the recorded workflows
  const WFS = (D.workflows && D.workflows.length) ? D.workflows : (D.workflow ? [D.workflow] : []);
  const W = WFS.map((w, i) => {
    const r = RW[w.config] || {}, p = w.prediction || {}, sp = (p.scores || {})[SN] || {};
    return { i, key: w.config, name: w.label || w.config, group: w.group || '', graph: w.graph || {}, runs: w.runs || 0,
      origin: w.origin, reach: r.reach || null, perf: r.score || sp.value || null, cost: r.cost_usd || (p.cost || {}).usd || null,
      ell: r.cost_per_accepted_usd || null, rescue: r.expected_rescue_usd, success: p.p_success || null,
      accept: r.p_accepted || null, reachFrom: r.reach_from || null };
  });
  // With too few scores of this name for the task type, the fit prices a miss with the success head's chance of an
  // accepted result, not the score head's chance to reach (recommend's score_backed). The page then ranks and prices
  // by that chance (`p_accepted`), so the totals stay the model's, and names the score estimate apart.
  const FB = !!(T && W.some(w => w.reachFrom === 'success_head'));
  const chanceOf = w => !T ? w.success : w.accept || (w.reachFrom === 'success_head' ? w.success : w.reach);
  const chanceWords = T && !FB ? `chance to reach ${score(T.target)}` : 'chance of an accepted result';
  const fbNote = !FB ? '' : ` The fit has too few ${esc(SN ? SN.replace(/_/g, ' ') : 'score')} scores for this kind of task to predict reaching ${esc(score(T.target))}, so each chance here is of an accepted result, as recommend prices it.` +
    (W.some(w => w.reach && num(w.reach.mean)) ? ` The score estimate's chance to reach it is shown apart and not used.` : '');
  const ranked = W.slice().sort((a, b) => mean(chanceOf(b), -1) - mean(chanceOf(a), -1) || mean(a.cost, Infinity) - mean(b.cost, Infinity));
  const best = ranked[0] || null;
  const U = D.units || [];

  // Shared axes: one score axis and one log cost axis for every figure on the page.
  const span = (vals, pad) => { const v = vals.filter(num); if (!v.length) return null; let lo = Math.min(...v), hi = Math.max(...v); const d = (hi - lo) || Math.abs(hi) || 1; return [lo - d * pad, hi + d * pad]; };
  const PD = span([...W.flatMap(w => w.perf ? [w.perf.lo, w.perf.hi] : []), ...U.flatMap(u => u.perf ? [u.perf.lo, u.perf.hi] : []),
    ...RUNS.map(r => r.score), T ? T.target : null], 0.04) || [0, 1];
  const CD = V.logDomain([...W.flatMap(w => w.cost ? [w.cost.lo, w.cost.hi] : []), ...U.flatMap(u => u.cost ? [u.cost.lo, u.cost.hi] : []),
    ...RUNS.map(r => r.cost_usd)].filter(v => num(v) && v > 0));
  const runTip = r => `${r.run}${r.subtype ? ' (' + r.subtype + ')' : ''}: ${num(r.score) ? score(r.score) + ' ' + scoreWord + ', ' : ''}${usd(r.cost_usd)}${r.reached === true ? ', reached' : r.reached === false ? ', missed' : ''}`;
  const scoreDots = cfgs => runsOf(cfgs).filter(r => num(r.score)).map(r => ({ v: r.score, cls: r.reached === false ? 'no' : '', t: runTip(r), tip: esc(runTip(r)) }));
  const costDots = cfgs => runsOf(cfgs).filter(r => num(r.cost_usd) && r.cost_usd > 0).map(r => ({ v: r.cost_usd, cls: r.reached === false ? 'no' : '', t: runTip(r) }));

  // ------------------------------------------------------------ the top
  const nr = fit.n_runs, yours = nr && typeof nr === 'object' ? +nr.user || 0 : null;
  const total = nr && typeof nr === 'object' ? Object.values(nr).reduce((a, b) => a + (+b || 0), 0) : (num(nr) ? nr : null);
  // Lane 2C's run horizon (`horizon_s`: seconds, or none) reads as a time budget, not as a raw feature.
  const hours = s => s >= 3600 ? `${+(s / 3600).toFixed(1)} h` : s >= 60 ? `${Math.round(s / 60)} min` : `${s} s`;
  const featureWord = (k, v) => k !== 'horizon_s' ? `${k}=${v}` : num(+v) && +v > 0 ? `horizon ${hours(+v)}` : 'open-ended';
  const about = [task.subtype, ...Object.entries(task.features || {}).map(([k, v]) => featureWord(k, v))].filter(Boolean);
  const bits = [`<b class="v-brand">loopmath <span>results</span></b>`, `<span>${esc(task.type || '')}${task.repo ? ' on ' + esc(task.repo) : ''}${about.length ? ' (' + esc(about.join('; ')) + ')' : ''}</span>`,
    T ? `<span>target <b>${esc(T.definition || T.rule)}</b></span>` : '<span>no score target</span>', `<span>fit <b>${esc(fit.id || 'n/a')}</b></span>`];
  const whence = !T ? '' : T.from === 'argument' ? ' (from <code>--target</code>)' : ` (the target of recommendation ${esc(T.rec || '')})`;
  const taskWhy = { arguments: 'as given on the command line', store: 'the most common type and repo in your runs', 'default': 'no task given and no runs yet' }[task.from] || '';
  let lede = T ? `Target <b>${esc(T.definition || T.rule)}</b>${whence}. ` : `No score target: pass <code>--target NAME&gt;=X</code> or run <code>loopmath recommend</code> with one to rank by it. `;
  lede += `Fitted ${esc(fmt.dt(fit.at))}` + (total != null ? ` on ${yours != null ? `<b>${fmt.int(yours)} of your runs</b> and ${fmt.int(total - yours)} shipped runs` : fmt.int(total) + ' runs'}` : '') +
    (num(data.fit_time_s) ? `, in ${data.fit_time_s.toFixed(1)} s` : '') + `. For ${esc(task.type || '')} tasks in ${esc(task.repo || '')}${taskWhy ? ', ' + esc(taskWhy) : ''}. Ranges are 80%.`;

  let h = `<div class="v-wrap"><div class="v-top">${bits.join('')}</div>
  <p class="v-kicker">Results after the fit</p>
  <h1>What your runs say about ${esc(task.type || 'these')} tasks in ${esc(task.repo || 'this repo')}</h1>
  <p class="v-lede" id="lede">${lede}</p>
  <nav class="v-rail" aria-label="questions"><a href="#q1"><b>1</b>Which workflow</a><a href="#q2"><b>2</b>Which model and effort</a><a href="#q3"><b>3</b>Is more spend worth it</a><a href="#q4"><b>4</b>How sure</a><a href="#q5"><b>5</b>What was fitted</a></nav>`;

  // ------------------------------------------------------------ 1. which workflow
  function answer1() {
    if (!W.length) return `No workflow recorded for ${esc(task.type || '')} tasks in ${esc(task.repo || '')} yet. Record runs, or draw one configuration with <code>${esc(taskCmd)} --workflow CFG --html</code>.`;
    const bc = chanceOf(best);
    let s = '';
    if (T && bc) s = `<b>${esc(best.name)}</b> has the best ${esc(chanceWords)}: <b>${pct(bc.mean)}</b>${rng(bc, pct) ? ` (${rng(bc, pct)})` : ''} at <b>${usd(mean(best.cost))}</b> a run${tail(best.cost)}.`;
    else if (T) s = `No workflow has a predicted ${esc(chanceWords)}.`;
    else if (bc) s = `With no score target the list is ranked by the chance of an accepted result: <b>${esc(best.name)}</b> is first at <b>${pct(bc.mean)}</b>${rng(bc, pct) ? ` (${rng(bc, pct)})` : ''}, <b>${usd(mean(best.cost))}</b> a run${tail(best.cost)}.`;
    if (T && RES) {
      const c = W.filter(w => w.ell && num(w.ell.mean)).sort((a, b) => a.ell.mean - b.ell.mean)[0];
      if (c) s += ` The cheapest per accepted result is <b>${esc(c.name)}</b>: <b>${usd(c.ell.mean)}</b> (${rng(c.ell, usd)}${tailWord(c.ell)}), with a ${pct(mean(chanceOf(c)))} ${FB ? 'chance of an accepted result' : 'chance to reach'} and ${usd(mean(c.cost))} a run.`;
    } else if (T) {
      const c = W.filter(w => mean(chanceOf(w), 0) >= 0.5 && w.cost).sort((a, b) => a.cost.mean - b.cost.mean)[0];
      const often = FB ? 'is accepted more often than not' : 'reaches it more often than not';
      s += c ? ` The cheapest that ${often} is <b>${esc(c.name)}</b>: ${pct(chanceOf(c).mean)}, ${usd(c.cost.mean)} a run.` : ` None ${FB ? 'is accepted' : 'reaches it'} more often than not.`;
      s += ` No rescue is priced for this target, so there is no cost per accepted result here; <code>loopmath recommend --type ${esc(LM.shq(task.type || ''))} --repo ${esc(LM.shq(task.repo || ''))} --target ${esc(LM.shq(T.rule || ''))}</code> prices one.`;
    }
    s += fbNote;
    const most = Math.max(0, ...W.map(w => w.runs || 0));
    if (most <= 3) s += ` No workflow has more than ${V.runs(most)} behind it, so the ranges are wide.`;
    return s;
  }
  const legend = `<div class="v-legend"><span><span class="sw" style="background:var(--dist);border:1px solid var(--dist-line)"></span>estimate, 80% range under the shape</span>` +
    (T ? `<span><span class="dt"></span>your run, reached ${esc(tgt)}</span><span><span class="dt no"></span>your run, missed</span><span><span class="ln"></span>target ${esc(tgt)}</span>` : '<span><span class="dt"></span>your run</span>') + '</div>';
  const TOP = 8;
  h += `<section class="v-q" id="q1"><p class="v-qno">1</p><h2>Which workflow for this kind of task?</h2><p class="v-answer" id="a1">${answer1()}</p>`;
  if (W.length) {
    h += `<div class="v-fig">${SN ? legend : ''}<div class="v-wl" id="wl"></div>` +
      (W.length > TOP ? `<button type="button" class="v-btn" id="wlmore">Show all ${W.length} workflows</button>` : '') +
      `<p class="v-figcap"><b>Figure 1.</b> Every workflow recorded for this task type and repo, ranked by the ${esc(chanceWords)}, then by cost. ` +
      `Boxes are workers: a piece of width n is n boxes. ${SN ? 'The shape is the predicted ' + esc(scoreWord) + ' of one run, drawn from its mean and 80% range. ' : ''}` +
      `${RES ? `Per accepted result is the run cost plus the chance of a miss times the rescue (${esc(RES.basis || RES.kind || 'rescue')}${RES.of ? ', ' + esc(RES.of) : ''}: ${usd(RES.usd)}). ` : ''}Click a row to see that workflow in full.</p></div>`;
  }
  h += `<details class="v-more" id="d-wf"><summary><span class="st">One workflow in full</span><span class="sa">Its graph with each piece's cost, rounds and gates, and the estimates behind each setting.</span></summary><div class="v-body est" id="est-graph"></div></details></section>`;

  // ------------------------------------------------------------ 2. which model and effort
  const effortRank = e => { const i = EFFORTS.indexOf(e); return i < 0 ? EFFORTS.length : i; };
  const models = [];
  U.forEach(u => { let m = models.find(x => x.model === u.model); if (!m) models.push(m = { model: u.model, provider: u.provider, harness: u.harness, user: !!u.user_model, units: [] }); m.units.push(u); });
  models.forEach(m => {
    m.units.sort((a, b) => effortRank(a.effort) - effortRank(b.effort));
    const scored = m.units.filter(u => u.perf && num(u.perf.mean));
    m.best = scored.length ? scored.reduce((a, b) => beats(b.perf.mean, a.perf.mean) ? b : a).effort : null;
    const ok = T ? m.units.filter(u => u.perf && num(u.perf.mean) && !beats(T.target, u.perf.mean) && u.cost) : [];
    m.cheapest = ok.length ? ok.reduce((a, b) => b.cost.mean < a.cost.mean ? b : a).effort : null;
  });
  models.sort((a, b) => (b.user - a.user) || String(a.provider).localeCompare(String(b.provider)) || a.model.localeCompare(b.model));
  const mine = models.filter(m => m.user), never = models.filter(m => !m.user);
  function answer2() {
    if (!models.length) return 'No settings to compare: the fit knows no model for this task. Set <code>models.allowed</code> in config, or record runs.';
    const clause = m => {
      const bu = m.units.find(u => u.effort === m.best);
      let s = `<b>${esc(m.model)}</b>` + (bu ? ` scores ${bestWord} at ${esc(m.best)} (${esc(score(bu.perf.mean))} ${esc(scoreWord)}, ${usd(mean(bu.cost))} a run)` : ' has no score estimate');
      if (T) s += m.cheapest ? `, and its cheapest effort whose mean reaches ${esc(tgt)} is ${esc(m.cheapest)}` : `, and no effort reaches ${esc(tgt)} on average`;
      return s;
    };
    return 'One agent working alone at each setting, predicted for this task. Compare efforts within a model. ' +
      (mine.length ? mine.map(clause).join('; ') + '.' : 'None of these models has run on this task.');
  }
  function modelBlocks(list) {
    return list.map(m => {
      const alone = m.units.filter(u => u.runs > 0).length;
      let s = `<div class="v-mb" data-model="${esc(m.model)}"><h3>${esc(m.model)}</h3><div class="prov">${esc(m.provider || '')} &middot; ${esc(m.harness || '')} &middot; ` +
        (m.user ? `${alone} of ${m.units.length} efforts run alone here` : 'never run on this task') + '</div>' +
        `<div class="er h"><span>effort</span><span>${SN ? esc(scoreWord) : 'chance of an accepted result'}</span><span>cost per run</span></div>`;
      m.units.forEach(u => {
        s += `<div class="er${u.effort === m.best ? ' best' : ''}" data-effort="${esc(u.effort)}" title="${esc(`${u.model}/${u.effort}: ${SN && u.perf ? score(u.perf.mean) + ' ' + scoreWord + ' (' + rng(u.perf, score) + '), ' : ''}${usd(mean(u.cost))} a run (${rng(u.cost, usd)}${tailWord(u.cost)})`)}">` +
          `<span class="ef">${esc(u.effort)}<span class="sub">${u.runs ? esc(V.runs(u.runs)) + ' alone' : m.user ? 'not run alone' : ''}</span></span>` +
          (SN ? V.miniDist(u.perf, PD, { w: 150, h: 26, target: T ? T.target : null, dots: scoreDots(u.configs || []), cls: u.effort === m.best ? 'acc' : '' }) : `<span>${V.chanceBar(u.success)} ${pct(mean(u.success))}</span>`) +
          V.miniDist(u.cost, CD, { w: 150, h: 26, log: true, dots: costDots(u.configs || []) }) + '</div>';
      });
      const bu = m.units.find(u => u.effort === m.best);
      if (bu) s += `<p class="v-note">${bestWord === 'lowest' ? 'Lowest' : 'Highest'} mean: <b>${esc(m.best)}</b>, ${esc(score(bu.perf.mean))} at ${usd(mean(bu.cost))}.` +
        (T ? (m.cheapest ? ` Cheapest mean at ${esc(tgt)}: <b>${esc(m.cheapest)}</b>.` : ` No effort reaches ${esc(tgt)} on average.`) : '') + '</p>';
      return s + '</div>';
    }).join('');
  }
  h += `<section class="v-q" id="q2"><p class="v-qno">2</p><h2>Which model and effort?</h2><p class="v-answer" id="a2">${answer2()}</p>`;
  if (mine.length) h += `<div class="v-models" id="models">${modelBlocks(mine)}</div>`;
  if (never.length) h += `<details class="v-more" id="d-never"><summary><span class="st">Models you have not run here (${never.length})</span><span class="sa">Their estimates come from shipped runs and the prior alone.</span></summary><div class="v-body"><div class="v-models" id="never">${modelBlocks(never)}</div></div></details>`;
  if (models.length) h += `<p class="v-figcap"><b>Figure 2.</b> Grouped by model, efforts low to max. ${SN ? 'Left: ' + esc(scoreWord) + ', target dashed. ' : 'Left: the chance of an accepted result. '}Right: cost per run on a log scale. Dots are your runs of one agent alone at that setting. Shapes are drawn from the mean and the 80% range.</p>`;
  h += '</section>';

  // ------------------------------------------------------------ 3. is more spend worth it
  const spendItems = W.filter(w => w.perf && w.cost && num(w.perf.mean) && num(w.cost.mean)).map(w => ({ key: w.key, group: w.group, name: w.name, perf: w.perf, cost: w.cost,
    dots: runsOf([w.key]).map(r => ({ usd: r.cost_usd, score: r.score, cls: r.reached === false ? 'no' : '' })) }));
  const levels = (() => {
    if (!spendItems.length) return [];
    const means = spendItems.map(w => w.perf.mean), hi = Math.max(...means), lo = Math.min(...means);
    let ls = V.niceTicks(lo, hi, 5).filter(t => t >= lo && t <= hi);
    if (T) ls = ls.filter(t => better === 'lower' ? t <= T.target : t >= T.target).concat([T.target]);
    ls = Array.from(new Set(ls)).sort((a, b) => better === 'lower' ? b - a : a - b);
    return ls.slice(0, 4);
  })();
  h += `<section class="v-q" id="q3"><p class="v-qno">3</p><h2>Is more spend worth it?</h2>`;
  h += spendItems.length ? `<p class="v-answer" id="a3"></p><div class="v-fig"><div id="spend"></div><p class="v-figcap"><b>Figure 3.</b> ${esc(scoreWord)} against cost per run, one panel per graph, log cost axis. Line: the 80% ${esc(scoreWord)} range. Small dots: your runs. Filled and labelled with the level: the cheapest workflow whose mean reaches it.</p></div>`
    : `<p class="v-answer" id="a3">No score is recorded for these runs, so there is no score to weigh against cost. Question 1 ranks the workflows by the chance of an accepted result and gives each one's cost.</p>`;
  h += '</section>';

  // ------------------------------------------------------------ 4. how sure
  const rows = data.rows || {};
  const split = k => { const r = rows[k]; if (!r) return null; const y = +r.user || 0, t = Object.values(r).reduce((a, b) => a + (+b || 0), 0); return { yours: y, shipped: t - y }; };
  const barHeads = [SN ? 'score:' + SN : null, 'success', 'cost', 'gate'].filter(k => k && rows[k]);
  const headName = k => k === 'score:' + SN ? scoreWord + (T ? ' (target)' : '') : { success: 'accepted or not', cost: 'cost', gate: 'review gates' }[k] || k;
  function answer4() {
    const sc = SN ? split('score:' + SN) : null, rr = W.map(w => w.runs || 0);
    let s = '';
    if (sc) s += `The ${esc(scoreWord)} estimates rest on <b>${fmt.int(sc.yours)} of your runs</b>` + (sc.shipped ? ` and ${fmt.int(sc.shipped)} shipped rows.` : ` alone: no shipped run has a ${esc(scoreWord)} score.`);
    if (rr.length) s += ` Each workflow has ${Math.min(...rr) === Math.max(...rr) ? V.runs(rr[0]) : V.runs(Math.min(...rr)).replace(/ runs?$/, '') + ' to ' + V.runs(Math.max(...rr))} behind it.`;
    const c = split('cost'), g = split('success');
    if (c && c.shipped) s += ` Cost ${g && g.shipped ? 'and success also lean' : 'also leans'} on shipped runs.`;
    return s || 'The fit recorded no row counts, so the page cannot say how many runs each estimate rests on.';
  }
  const bar = k => { const r = split(k); const t = r.yours + r.shipped || 1; return `<div class="b" data-head="${esc(k)}"><span>${esc(headName(k))}</span><span class="track"><i class="y" style="width:${(r.yours / t * 100).toFixed(1)}%"></i><i class="s" style="width:${(r.shipped / t * 100).toFixed(1)}%"></i></span><span class="num">${fmt.int(r.yours)} yours, ${fmt.int(r.shipped)} shipped</span></div>`; };
  h += `<section class="v-q" id="q4"><p class="v-qno">4</p><h2>How sure is this?</h2><p class="v-answer" id="a4">${answer4()}</p>`;
  if (barHeads.length) h += `<div class="v-fig"><div class="v-legend"><span><span class="sw" style="background:var(--accent)"></span>rows from your runs</span><span><span class="sw" style="background:var(--dot)"></span>rows from shipped runs</span></div><div class="v-bars" id="bars">${barHeads.map(bar).join('')}</div><p class="v-figcap"><b>Figure 4.</b> The rows each estimate learns from, by source.</p></div>`;
  h += `<p class="v-note" id="sens">Your runs alone against this fit with the shipped prior: not computed. It needs a second fit without the shipped prior, which this version does not keep.</p>`;
  h += `<details class="v-more" id="d-levels"><summary><span class="st">Every estimate, by level</span><span class="sa">How each model, effort, role, workflow shape, task type and repo moves cost, success and the scores, with the runs behind each.</span></summary><div class="v-body est" id="est-levels"></div></details></section>`;

  // ------------------------------------------------------------ 5. what was fitted
  const rbs = data.runs_by_source || {};
  const shippedBy = Object.entries(rbs).filter(([k]) => k !== 'user').map(([k, v]) => `${esc(k)} ${fmt.int(v)}`).join(', ');
  const dropped = (data.dropped || []).map(d => d && typeof d === 'object' ? `${fmt.int(d.n)} ${esc(d.reason || '')}` : esc(String(d))).join('; ');
  h += `<section class="v-q" id="q5"><p class="v-qno">5</p><h2>What was fitted, and what was dropped?</h2><dl class="v-kv" id="fitkv">
    <dt>Fit</dt><dd><code>${esc(fit.id || 'n/a')}</code>, ${esc(fmt.dt(fit.at))}${num(data.fit_time_s) ? ', ' + data.fit_time_s.toFixed(1) + ' s' : ''}${data.code_version ? ', loopmath ' + esc(data.code_version) : ''}</dd>
    <dt>Options</dt><dd>${(data.without || []).length ? 'left out: ' + esc(data.without.join(', ')) : 'default: your runs plus the shipped prior'}</dd>
    <dt>Your runs</dt><dd>${yours != null ? fmt.int(yours) : 'not recorded'}${RUNS.length ? `, ${fmt.int(RUNS.length)} of them ${esc(task.type || '')} tasks in ${esc(task.repo || '')}` : ''}</dd>
    <dt>Shipped runs</dt><dd>${shippedBy || 'none'}</dd>
    <dt>Dropped</dt><dd>${dropped || 'nothing'}</dd></dl>`;
  if (RUNS.length) h += `<div class="v-fig"><div id="runsplot"></div><p class="v-figcap"><b>Figure 5.</b> Your ${fmt.int(RUNS.length)} runs ${SN && RUNS.some(r => num(r.score)) ? 'by ' + esc(scoreWord) : 'by cost, log scale'}, one row per ${RUNS.some(r => r.subtype) ? 'subtype' : 'task type'}. Hover a dot for the run.</p></div>`;
  h += `<details class="v-more" id="d-data"><summary><span class="st">Data behind the fit</span><span class="sa">Rows per source and head, dropped rows, the scale of each level, and the sensitivity to the benchmark prior.</span></summary><div class="v-body est" id="est-data"></div></details></section>`;
  h += `<p class="v-foot">loopmath posterior, fit ${esc(fit.id || 'n/a')}${D.generated_at ? ', page written ' + esc(fmt.dt(D.generated_at)) : ''}. Ranges are 80%; shapes are drawn from the mean and the 80% range. The same numbers are in <code>loopmath posterior --json</code>.</p></div>`;
  app.innerHTML = h;

  // ------------------------------------------------------------ figures
  let all = false;
  function listRows() {
    const host = document.getElementById('wl');
    if (!host) return;
    const list = all ? ranked : ranked.slice(0, TOP);
    let s = `<div class="r h"><span>graph</span><span>workflow</span><span>${esc(chanceWords)}</span><span>${SN ? esc(scoreWord) : ''}</span><span class="cs">${RES ? 'per accepted result' : 'cost per run'}</span></div>`;
    list.forEach(w => {
      const c = chanceOf(w), rs = runsOf([w.key]), reached = rs.filter(r => r.reached).length;
      const money = RES && w.ell ? `<b>${usd(w.ell.mean)}</b><span class="sub">${rng(w.ell, usd)}</span><span class="sub">${usd(mean(w.cost))} a run</span>${tail(w.ell, w.cost)}`
        : w.cost ? `<b>${usd(w.cost.mean)}</b><span class="sub">${rng(w.cost, usd)}</span>${tail(w.cost)}` : 'n/a';
      s += `<div class="r${w === best ? ' best' : ''}" data-wf="${w.i}" data-cfg="${esc(w.key)}" tabindex="0" role="button" aria-label="${esc('show ' + w.name + ' in full')}">` +
        `<span class="g">${V.miniGraph(w.graph)}</span>` +
        `<span class="nm">${esc(w.name)}<span class="sub">${esc(V.runs(w.runs))}${T ? ` &middot; ${reached} reached` : ''}${w.origin === 'usual' ? ' &middot; your usual' : ''}</span></span>` +
        `<span class="ch">${c && num(c.mean) ? `<b>${pct(c.mean)}</b> ${V.chanceBar(c, w === best ? 'pick' : '')}<span class="sub">${rng(c, pct) || 'no range'}</span>${FB && w.reach && num(w.reach.mean) ? `<span class="sub">score estimate: ${pct(w.reach.mean)} to reach ${esc(tgt)}, not used</span>` : ''}` : '<span class="sub">not predicted</span>'}</span>` +
        `<span class="dist">${SN ? V.miniDist(w.perf, PD, { w: 300, h: 30, target: T ? T.target : null, dots: scoreDots([w.key]), cls: w === best ? 'acc' : '' }) : ''}</span>` +
        `<span class="cs">${money}</span></div>`;
    });
    host.innerHTML = s;
  }
  function openWorkflow(i) {
    if (window.LMPosterior) window.LMPosterior.showWorkflow(i);
    const d = document.getElementById('d-wf');
    d.open = true;
    d.scrollIntoView({ block: 'start' });
  }
  const wl = document.getElementById('wl');
  if (wl) {
    listRows();
    wl.addEventListener('click', e => { const r = e.target.closest('.r[data-wf]'); if (r) openWorkflow(+r.getAttribute('data-wf')); });
    wl.addEventListener('keydown', e => { const r = e.target.closest('.r[data-wf]'); if (r && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); openWorkflow(+r.getAttribute('data-wf')); } });
  }
  const more = document.getElementById('wlmore');
  if (more) more.addEventListener('click', () => { all = !all; listRows(); more.textContent = all ? `Show the top ${TOP}` : `Show all ${W.length} workflows`; });

  function drawSpend() {
    const host = document.getElementById('spend');
    if (!host) return;
    const res = V.scoreCost(host, spendItems, { xDomain: CD, yDomain: PD, target: T ? T.target : null, levels, better });
    const per = [];  // consecutive levels with the same cheapest workflow share one clause
    levels.forEach(L => {
      const k = Object.keys(res.cheapest).find(key => res.cheapest[key].indexOf(L) >= 0);
      if (!k) return;
      const last = per[per.length - 1];
      if (last && last.w.key === k) last.levels.push(L); else per.push({ levels: [L], w: spendItems.find(w => w.key === k) });
    });
    const a3 = document.getElementById('a3');
    a3.innerHTML = per.length ? 'The cheapest mean cost per ' + esc(scoreWord) + ' level: ' + per.map(({ levels: ls, w }) => `<b>${ls.map(L => esc(score(L))).join(' and ')}</b> ${esc(w.name)} at <b>${usd(w.cost.mean)}</b> a run${tail(w.cost)}`).join('; ') + '.' +
      (levels.length && !Object.values(res.cheapest).some(ls => ls.indexOf(levels[levels.length - 1]) >= 0) ? ` No workflow's mean reaches ${esc(score(levels[levels.length - 1]))}.` : '')
      : `No workflow's mean reaches ${T ? esc(tgt) : 'the levels shown'}.`;
  }
  function drawRuns() {
    const host = document.getElementById('runsplot');
    if (!host) return;
    const byScore = SN && RUNS.some(r => num(r.score));
    const groups = {};
    RUNS.forEach(r => { const k = r.subtype || task.type || 'runs'; (groups[k] = groups[k] || []).push(r); });
    const rowsOut = Object.keys(groups).sort().map(k => ({ key: k, label: k, sub: `${V.runs(groups[k].length)}${T && byScore ? ', ' + groups[k].filter(r => r.reached).length + ' reached' : ''}`,
      dots: groups[k].filter(r => num(byScore ? r.score : r.cost_usd) && (byScore || r.cost_usd > 0)).map(r => ({ v: byScore ? r.score : r.cost_usd, cls: r.reached === false ? 'no' : '', tip: esc(runTip(r)) })) }));
    V.ridge(host, rowsOut, byScore ? { domain: PD, fmt: score, target: T ? T.target : null, targetLabel: 'target ' + tgt, rowH: 40 } : { domain: CD, log: true, fmt: V.usd0, rowH: 40 });
  }
  drawSpend();
  drawRuns();
  V.onResize(() => { drawSpend(); drawRuns(); });
  if (window.LMPosterior) window.LMPosterior.init();
  if (location.hash === '#open') document.querySelectorAll('details.v-more').forEach(d => { d.open = true; });
})();
