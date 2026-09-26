/* loopmath builder (0.2.2, lane 22P): the chance-against-cost chart, the graph canvas with each piece's
   settings edited on the piece, the next steps and the "Your build" sidebar. Numbers come only from
   `loopmath builder` (GET /api/context, POST /api/predict, POST /api/predict_many); when it does not
   answer the page says so. Vanilla JS, no framework, no network beyond the page's own server. */
(() => {
'use strict';
// ------------------------------------------------------------------ helpers
const NS = 'http://www.w3.org/2000/svg';
const $ = id => document.getElementById(id);
function el(tag, attrs, parent, text) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs || {}) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  if (parent) parent.appendChild(n);
  return n;
}
function h(tag, attrs, html) {
  const n = document.createElement(tag);
  for (const k in attrs || {}) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
  if (html != null) n.innerHTML = html;
  return n;
}
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const clone = o => JSON.parse(JSON.stringify(o));
const logit = p => { p = clamp(p, 1e-4, 1 - 1e-4); return Math.log(p / (1 - p)); };
const sig = x => 1 / (1 + Math.exp(-x));
const fin = x => x != null && isFinite(x);
const usd = x => !fin(x) ? 'n/a' : Math.abs(x) >= 100 ? '$' + Math.round(x) : Math.abs(x) >= 10 ? '$' + x.toFixed(1) : '$' + x.toFixed(2);
const pct = p => !fin(p) ? 'n/a' : p < 0.005 ? '<1%' : p > 0.995 ? '>99%' : Math.round(p * 100) + '%';
function signed(v, kind) {
  if (!fin(v)) return '';
  if (kind === 'pts') { const r = Math.round(v * 100); return (r > 0 ? '+' : r < 0 ? '−' : '±') + Math.abs(r) + ' pts'; }
  const a = Math.abs(v); return (a < 0.005 ? '±' : v > 0 ? '+' : '−') + usd(a);
}
const tone = (v, higherIsBetter, eps) => !fin(v) || Math.abs(v) < eps ? 'flat' : (v > 0) === higherIsBetter ? 'good' : 'bad';
const errText = e => typeof e === 'string' ? e : (e && e.message) || String(e);
const errPiece = e => (e && typeof e === 'object' && e.piece) || null;
function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('on'); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove('on'), 1800); }

const ROLES = {
  planner: { name: 'planner', idp: 'plan', color: 'var(--r-planner)', art: 'plan_doc', kind: 'plan', out: 'plan' },
  implementer: { name: 'implementer', idp: 'implement', color: 'var(--r-implementer)', art: 'diff', kind: 'diff', out: 'diff' },
  worker: { name: 'worker', idp: 'work', color: 'var(--r-worker)', art: 'diff', kind: 'diff', out: 'diff' },
  referee: { name: 'selector', idp: 'select', color: 'var(--r-referee)', art: 'diff', kind: 'diff', out: 'pick' },
  reviewer: { name: 'reviewer', idp: 'review', color: 'var(--r-reviewer)', art: 'verdict', kind: 'verdict', out: 'verdict' },
  tester: { name: 'tester', idp: 'test', color: 'var(--r-tester)', art: 'report', kind: 'report', out: 'tests' },
  integrator: { name: 'integrator', idp: 'integrate', color: 'var(--r-tester)', art: 'merged', kind: 'diff', out: 'merged' }
};
const ROLE_ORDER = ['planner', 'implementer', 'worker', 'referee', 'reviewer', 'tester', 'integrator'];
const roleName = r => (ROLES[r] || { name: r }).name;
const roleColor = r => (ROLES[r] || { color: 'var(--r-tester)' }).color;
const isBuilder = r => r === 'implementer' || r === 'worker';
const hasGate = r => r === 'reviewer' || r === 'referee';
const OPTION_NAME = { goal: 'Recommended', pair: 'Explore pick', reference: 'Reference', cheapest_run: 'Cheapest at 50%', most_likely: 'Most likely' };
const Z = { 50: 0.6745, 80: 1.2816, 90: 1.6449, 95: 1.96 };
const TR = { chance: [logit, sig], cost: [v => Math.log(Math.max(v, 1e-4)), Math.exp] };

// ------------------------------------------------------------------ the API (live only, no mock)
const API = { many: null, down: null };
async function call(path, body) {
  let r;
  try {
    r = await fetch(path, body === undefined ? { headers: { accept: 'application/json' } }
      : { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  } catch (e) { throw new Error('loopmath builder is not answering (' + (e.message || e) + ')'); }
  const ct = r.headers.get('content-type') || '';
  if (!/json/.test(ct)) throw Object.assign(new Error(path + ' answered ' + r.status + ' without JSON'), { status: r.status });
  const j = await r.json();
  if (!r.ok && r.status !== 200) throw Object.assign(new Error((j.errors || []).map(errText).join('; ') || path + ' answered ' + r.status), { status: r.status, body: j });
  return j;
}
function setMode(kind, text) { const m = $('mode'); if (!m) return; m.className = 'mode ' + kind; m.textContent = text; }
function apiUp() { if (API.down) { API.down = null; setMode('live', 'live: fitted model'); } }
function apiDown(e) { API.down = e.message || String(e); setMode('down', 'loopmath not answering'); }
async function predictOne(cfg) { const j = await call('/api/predict', { config: cfg }); apiUp(); return j; }
async function predictMany(cfgs) {
  if (!cfgs.length) return [];
  if (API.many !== false) {
    try {
      const j = await call('/api/predict_many', { configs: cfgs });
      if (Array.isArray(j.results) && j.results.length === cfgs.length) { API.many = true; apiUp(); return j.results; }
      throw new Error((j.errors || []).map(errText).join('; ') || 'predict_many gave no results');
    } catch (e) {
      if (e.status === 404) API.many = false;  // a 0.2.1 server: one call per configuration
      else throw e;
    }
  }
  const out = [];
  for (const c of cfgs) out.push(await predictOne(c));
  return out;
}

// ------------------------------------------------------------------ numbers
function bandsFrom(m, lo, hi, kind) {
  const [f, inv] = TR[kind], tm = f(m), dl = Math.max(0, tm - f(lo)), dh = Math.max(0, f(hi) - tm), b = {};
  for (const k of [50, 80, 90, 95]) { const s = Z[k] / Z[80]; b[k] = [inv(tm - dl * s), inv(tm + dh * s)]; }
  b[80] = [lo, hi];
  return b;
}
function iv(o, kind, given) {
  if (!o || !fin(o.mean)) return null;
  const m = o.mean, g80 = given && given['80'];
  let lo = fin(o.lo) ? o.lo : g80 ? g80[0] : m, hi = fin(o.hi) ? o.hi : g80 ? g80[1] : m;
  const d = bandsFrom(m, lo, hi, kind), b = {};
  let derived = false;
  for (const k of [50, 80, 90, 95]) { const g = given && given[String(k)]; if (g && fin(g[0]) && fin(g[1])) b[k] = g; else { b[k] = d[k]; if (k !== 80) derived = true; } }
  // N3 (0.2.3): `typical_first` (the server's rule) leads a run cost with its median, the typical run
  return { mean: m, lo, hi, median: o.median, attempts: o.attempts, b, derived, typical: !!o.typical_first && fin(o.median) };
}
function norm(n) {
  if (!n) return null;
  const bands = n.bands || {};
  let ch = n.chance && fin(n.chance.mean) ? n.chance : n.p_reach && fin(n.p_reach.mean) ? n.p_reach : null;
  if (!ch) {  // candidates carry the chance of an accepted result in bands.chance and the rescue
    const R = D && D.rescue && D.rescue.usd;
    if (fin(n.expected_rescue_usd) && R > 0) ch = { mean: clamp(1 - n.expected_rescue_usd / R, 0, 1) };
    else if (bands.chance && bands.chance['50']) ch = { mean: (bands.chance['50'][0] + bands.chance['50'][1]) / 2 };
  }
  const w = n.p_accepted_within;
  const out = {
    chance: iv(ch, 'chance', bands.chance), run: iv(n.run_cost_usd, 'cost', bands.run_cost_usd),
    cpa: iv(n.cost_per_accepted_usd, 'cost', bands.cost_per_accepted_usd),
    within: w ? iv(w, 'chance', bands.p_accepted_within || w.bands) : null,
    rescue: n.expected_rescue_usd
  };
  return out.chance && out.run && out.cpa ? out : null;
}
function lerpNum(a, b, k) {
  const L = (p, q, kind) => { const [f, inv] = TR[kind]; return inv(f(p) + (f(q) - f(p)) * k); };
  const mix = (o, q, kind) => { if (!o || !q) return q; const out = { ...q, mean: L(o.mean, q.mean, kind), b: {} }; for (const z of [50, 80, 90, 95]) out.b[z] = [L(o.b[z][0], q.b[z][0], kind), L(o.b[z][1], q.b[z][1], kind)]; return out; };
  return { chance: mix(a.chance, b.chance, 'chance'), run: mix(a.run, b.run, 'cost'), cpa: mix(a.cpa, b.cpa, 'cost'), within: mix(a.within, b.within, 'chance'), rescue: b.rescue };
}

// ------------------------------------------------------------------ the context
let D = null;
function runsOf(rb) {
  if (rb == null) return { total: null, byRole: null, byEffort: null };
  if (typeof rb === 'number') return { total: rb, byRole: null, byEffort: null };  // 0.2.1: the fit's count
  return { total: rb.total || 0, byRole: rb.by_role || {}, byEffort: rb.by_effort || {} };
}
function prepare(c) {
  D = { task: c.task || {}, rule: c.rule || {}, fit: c.fit || {}, rescue: c.rescue || {}, catalog: c.catalog || {},
    ownRuns: typeof c.own_runs === 'number' ? c.own_runs : null };
  const cat = D.catalog;
  // the models the builder offers; a workflow on any other model is a retired one: shown, never a start point or a next step
  const offered = Array.isArray(cat.offered) ? new Set(cat.offered.map(m => m.id || m)) : null;
  D.models = (cat.models || []).filter(m => !(m && (m.retired === true || m.offered === false)) && (!offered || offered.has(m.id || m)))
    .map(m => ({ id: m.id || m, family: m.family, harness: m.harness, ...runsOf(m.runs_behind) }));
  D.modelById = {}; D.models.forEach(m => { D.modelById[m.id] = m; });
  D.isRetired = cfg => !!cfg && D.models.length > 0 && Object.values(cfg.settings || {}).some(s => s && s.model && !D.modelById[s.model]);
  D.cands = (c.candidates || []).map(x => ({ id: x.config.id, config: x.config, label: x.label, origin: x.origin, num: norm(x.numbers), retired: x.retired === true || D.isRetired(x.config) })).filter(x => x.num);
  D.byId = {}; D.cands.forEach(x => { D.byId[x.id] = x; });
  const choices = c.choices || [];
  D.goalId = c.goal_config_id || (choices.find(ch => ch.key === 'goal') || {}).config || null;
  D.options = choices.map((ch, i) => {
    const id = ch.key === 'pair' && ch.explore_config ? ch.explore_config : ch.config;
    const cand = D.byId[id];
    const cfg = ch.key === 'pair' ? cand && cand.config : ch.configuration || (cand && cand.config);
    const num = cand ? cand.num : ch.key === 'pair' ? null : norm(ch);
    return { no: ch.option || i + 1, key: ch.key, name: OPTION_NAME[ch.key] || ch.key, title: ch.title, id, cfg, num, label: cand ? cand.label : ch.label,
      retired: ch.retired === true || D.isRetired(cfg) };
  }).filter(o => o.cfg && o.num);
  const goal = D.options.find(o => o.key === 'goal');
  D.recNum = goal ? goal.num : null;
  D.efforts = cat.efforts || ['low', 'medium', 'high', 'xhigh', 'max'];
  D.effortsBy = cat.efforts_by_harness || {};
  D.roleRuns = {};
  (cat.roles || []).forEach(r => { if (typeof r === 'string') D.roleRuns[r] = null; else D.roleRuns[r.id] = r.runs; });
  // efforts with recorded runs per role and model, when the server has no by_effort (0.2.1)
  D.recorded = {};
  D.cands.filter(x => x.origin === 'recorded').forEach(x => x.config.workflow.pieces.forEach(p => {
    const s = x.config.settings[p.id] || {}; D.recorded[p.role + '|' + s.model + '|' + s.effort] = true;
  }));
  D.refStyle = D.cands.some(x => Object.values(x.config.settings).some(s => s && s.model_ref));
  // display names: the model id without its vendor prefix, unless two models (offered or retired) would read the same
  const base = id => String(id).replace(/^gpt-[\d.]+-/, '').replace(/^claude-/, '').replace(/-(\d+)-(\d+)$/, ' $1.$2').replace(/-(\d+)$/, ' $1');
  const ids = new Set(D.models.map(m => m.id));
  [...D.cands.map(x => x.config), ...D.options.map(o => o.cfg)].forEach(cf => Object.values(cf.settings || {}).forEach(s => { if (s && s.model) ids.add(s.model); }));
  const counts = {}; ids.forEach(id => { counts[base(id)] = (counts[base(id)] || 0) + 1; });
  D.short = id => counts[base(id)] > 1 ? String(id).replace(/^gpt-/, '').replace(/^claude-/, '') : base(id);
  // shapes: the catalog's, each dropped as its best searched version when the search has one
  D.shapes = (cat.shapes || []).map(s => {
    const best = D.cands.filter(x => !x.retired && x.config.workflow.id === s.id).sort((a, b) => a.num.cpa.mean - b.num.cpa.mean)[0];
    return { id: s.id, title: s.title || s.id, workflow: s.workflow, best };
  }).filter(s => s.workflow);
  // workflows the page can reuse whole, so an edit that lands on a known structure keeps its id
  D.wfLib = []; const seen = new Set();
  [...D.shapes.map(s => s.workflow), ...D.cands.map(x => x.config.workflow)].forEach(w => { const k = JSON.stringify(w); if (!seen.has(k)) { seen.add(k); D.wfLib.push(w); } });
  // the reference line: a choice when it runs on offered models, else (a retired model) shown on the chart only
  const ref = c.reference;
  D.ref = null;
  if (ref && ref.config && !D.options.some(o => o.key === 'reference')) {
    const num = norm(ref.numbers);
    const retired = (ref.retired_models && ref.retired_models.length > 0) || D.isRetired(ref.config);
    if (num) D.ref = { no: 'R', key: 'reference', name: 'Reference', id: ref.config.id, cfg: ref.config, num, label: ref.label, retired, text: ref.text };
  }
  const live = D.options.filter(o => !o.retired), first = (goal && !goal.retired ? goal : live[0]) || null;
  if (c.start) {  // --start opens even a retired configuration: its pieces show errors until their model changes
    const cand = D.byId[c.start.id];
    D.startOpt = D.options.find(o => o.id === c.start.id) || { no: '\u2022', key: 'start', name: 'Your start', id: c.start.id, cfg: c.start, num: cand && cand.num, label: cand && cand.label };
  } else D.startOpt = first;
  D.start = D.startOpt ? D.startOpt.cfg : ((D.cands.find(x => !x.retired) || {}).config || null);
}
const modelHarness = m => (D.modelById[m] || {}).harness;
const effortsFor = harness => (harness && D.effortsBy[harness]) || D.efforts;
const chanceName = () => D.rule && D.rule.score ? 'Chance to reach the target' : 'Chance of an accepted result';

// ------------------------------------------------------------------ graph state <-> configuration
let S = null;          // {nodes: [{id, role, width, model, effort, harness, set, x, y}], edges: [[a, b]], gates: [{after, on_fail}], rounds, rescue, wf}
let SEL = null;        // {kind: 'node'|'edge'|'gate', id, at}
const past = [], future = [];
let lastAction = null, FLASH = null;

function pieceEdges(w) {
  const pieces = new Set(w.pieces.map(p => p.id)), out = {};
  (w.edges || []).forEach(([a, b]) => { (out[a] = out[a] || []).push(b); });
  const res = [];
  w.pieces.forEach(p => (out[p.id] || []).forEach(a => {
    if (pieces.has(a)) res.push([p.id, a]);
    else (out[a] || []).forEach(b => { if (pieces.has(b) && b !== p.id) res.push([p.id, b]); });
  }));
  const seenE = new Set();
  return res.filter(e => { const k = e.join('>'); if (seenE.has(k)) return false; seenE.add(k); return true; });
}
function fromConfig(cfg) {
  const w = cfg.workflow, control = w.control || {};
  const nodes = w.pieces.map(p => {
    const s = cfg.settings[p.id] || {};
    return { id: p.id, role: p.role, width: p.width || 1, model: s.model, effort: s.effort, harness: s.harness || modelHarness(s.model), set: clone(s), x: 0, y: 0 };
  });
  const st = { nodes, edges: pieceEdges(w), gates: (control.gates || []).filter(g => g.on_fail).map(g => ({ after: g.after, on_fail: g.on_fail })),
    rounds: control.budget_rounds || 1, rescue: control.rescue === undefined ? 'redo_usual' : control.rescue, wf: clone(w) };
  autoLayout(st);
  return st;
}
function settingOf(n) {
  const s = n.set ? clone(n.set) : { harness: n.harness, model: n.model, effort: n.effort, context_policy: 'fresh', options: {} };
  s.model = n.model; s.effort = n.effort; s.harness = n.harness || modelHarness(n.model) || s.harness;
  if (s.model_ref ? s.model_ref.id !== n.model : !n.set && D.refStyle) s.model_ref = { raw: n.model, id: n.model };
  return s;
}
function topo(st) {
  const L = layers(st);
  return st.nodes.map(n => n.id).sort((a, b) => L[a] - L[b] || st.nodes.findIndex(n => n.id === a) - st.nodes.findIndex(n => n.id === b));
}
function layers(st) {
  const L = {}; st.nodes.forEach(n => { L[n.id] = 0; });
  for (let k = 0; k < st.nodes.length; k++) st.edges.forEach(([a, b]) => { if (L[b] < L[a] + 1) L[b] = L[a] + 1; });
  return L;
}
function shapeSig(st) {  // roles, links and gates by piece id, widths left out
  const p = st.nodes.map(n => n.id + ':' + n.role).sort().join(',');
  const e = st.edges.map(x => x.join('>')).sort().join(',');
  const g = st.gates.map(x => x.after + '>' + x.on_fail).sort().join(',');
  return p + '|' + e + '|' + g;
}
function knownWorkflow(st) {
  const s = shapeSig(st);
  for (const w of D.wfLib) {
    const t = { nodes: w.pieces.map(p => ({ id: p.id, role: p.role })), edges: pieceEdges(w), gates: ((w.control || {}).gates || []).filter(g => g.on_fail).map(g => ({ after: g.after, on_fail: g.on_fail })) };
    if (shapeSig(t) === s) return w;
  }
  return null;
}
function generatedWorkflow(st) {
  const ids = topo(st), byId = Object.fromEntries(st.nodes.map(n => [n.id, n]));
  const pieces = ids.map(id => ({ id, role: byId[id].role, setting: null, width: byId[id].width }));
  const into = {}; st.edges.forEach(([a, b]) => (into[b] = into[b] || []).push(a));
  const outOf = {}; st.edges.forEach(([a, b]) => (outOf[a] = outOf[a] || []).push(b));
  const artOf = {}, kinds = { issue: 'issue', repo: 'repo' }, used = new Set(['issue', 'repo']);
  ids.forEach(id => {
    const n = byId[id], R = ROLES[n.role] || ROLES.tester;
    let a = R.art;
    if (isBuilder(n.role) && (outOf[id] || []).some(b => byId[b].role === 'referee')) a = 'candidates';
    if (used.has(a)) a = a + '_' + id;
    used.add(a); artOf[id] = a; kinds[a] = R.kind;
  });
  const edges = [], push = (a, b) => { if (!edges.some(e => e[0] === a && e[1] === b)) edges.push([a, b]); };
  ids.forEach(id => {
    const n = byId[id], ins = into[id] || [];
    if (!ins.length || n.role !== 'referee') push('issue', id);
    if (!ins.length && n.role !== 'referee') push('repo', id);
    if (isBuilder(n.role)) push('repo', id);
    ins.forEach(a => { push(a, artOf[a]); push(artOf[a], id); });
    push(id, artOf[id]);
  });
  const gates = [];
  ids.forEach(id => { if (byId[id].role === 'referee') gates.push({ id: 'g_' + id, after: id, rule: 'referee_pick', on_fail: null }); });
  st.gates.forEach(g => { if (byId[g.after] && byId[g.on_fail]) {
    const i = gates.findIndex(x => x.after === g.after);
    const gg = { id: 'g_' + g.after, after: g.after, rule: byId[g.after].role === 'referee' ? 'referee_pick' : 'review_approve', on_fail: g.on_fail };
    if (i >= 0) gates[i] = gg; else gates.push(gg);
  } });
  return { id: 'custom', version: 1, title: 'Custom workflow', pieces, artifacts: ['issue', 'repo', ...ids.map(id => artOf[id])], edges,
    control: { gates, budget_rounds: st.rounds, rescue: st.rescue }, artifact_kinds: kinds };
}
function toConfig(st) {
  let wf = st.wf ? clone(st.wf) : null;
  if (!wf) { const k = knownWorkflow(st); wf = k ? clone(k) : generatedWorkflow(st); }
  const byId = Object.fromEntries(st.nodes.map(n => [n.id, n]));
  wf.pieces.forEach(p => { if (byId[p.id]) p.width = byId[p.id].width; });
  wf.control = { ...(wf.control || {}), budget_rounds: st.rounds };
  const settings = {};
  wf.pieces.forEach(p => { if (byId[p.id]) settings[p.id] = settingOf(byId[p.id]); });
  return { id: null, workflow: wf, settings };
}
// piece box geometry (world units)
const NW = 176, NH = 70, STK = 7, COLX = 236, ROWY = 112;
function autoLayout(st) {
  const L = layers(st), cols = {};
  st.nodes.forEach(n => (cols[L[n.id]] = cols[L[n.id]] || []).push(n));
  Object.keys(cols).forEach(c => {
    const list = cols[c], tot = list.reduce((s, n) => s + ROWY + (Math.min(n.width, 4) - 1) * STK, 0) - ROWY;
    let y = 200 - tot / 2 - NH / 2;
    list.forEach(n => { n.x = 170 + Number(c) * COLX; n.y = y; y += ROWY + (Math.min(n.width, 4) - 1) * STK; });
  });
}
function shapeOf(st) {
  const w = st.wf || knownWorkflow(st);
  return w ? { id: w.id, title: shapeTitle(w) } : null;
}
const TITLE_MAX = 32;
function shapeTitle(w) {
  const s = D.shapes.find(x => x.id === w.id);
  if (s) return s.title;
  // a long title (a study's own name, say) reads worse on the chips than the workflow's id, when that is shorter
  const t = String(w.title || w.id), id = String(w.id || '').replace(/_/g, ' ');
  return t.length > TITLE_MAX && id && id.length < t.length ? id : t;
}
function labelOf(st) {
  const ids = topo(st), byId = Object.fromEntries(st.nodes.map(n => [n.id, n])), sh = shapeOf(st);
  return (sh ? sh.id : 'custom') + ': ' + ids.map(id => (byId[id].width > 1 ? byId[id].width + ' x ' : '') + byId[id].model + '/' + byId[id].effort).join(', ');
}

// structural edits by role, reusing a known workflow for the new list of roles (next steps: add or remove a piece)
function bySeq(roles) {
  const key = roles.join(',');
  return D.shapes.map(s => s.workflow).concat(D.wfLib).find(w => w.pieces.map(p => p.role).join(',') === key) || null;
}
function defaultSetting(role, st) {
  const pick = n => ({ model: n.model, effort: n.effort, harness: n.harness, set: n.set });
  const same = st.nodes.find(n => n.role === role);
  if (same) return pick(same);
  const goal = D.options.find(o => o.key === 'goal');
  if (goal) { const p = goal.cfg.workflow.pieces.find(q => q.role === role); if (p) { const s = goal.cfg.settings[p.id]; return { model: s.model, effort: s.effort, harness: s.harness, set: s }; } }
  const b = st.nodes.find(n => isBuilder(n.role)) || st.nodes[0];
  if (b) return pick(b);
  const m = D.models[0] || { id: 'gpt-5.6-luna', harness: 'codex' };
  return { model: m.id, effort: 'high', harness: m.harness, set: null };
}
function reshape(st, roles) {
  roles = [...roles].sort((a, b) => ROLE_ORDER.indexOf(a) - ROLE_ORDER.indexOf(b));
  const tpl = bySeq(roles), old = st.nodes.map(n => n);
  const take = role => { const i = old.findIndex(n => n.role === role || (isBuilder(role) && isBuilder(n.role))); return i >= 0 ? old.splice(i, 1)[0] : null; };
  let wf;
  if (tpl) wf = clone(tpl);
  else {  // no known workflow with these pieces: a plain chain, a reviewer sends work back to the last builder
    const pieces = roles.map((r, i) => ({ id: (ROLES[r] || { idp: r }).idp + (roles.filter(x => x === r).length > 1 ? '-' + (i + 1) : ''), role: r, setting: null, width: 1 }));
    const tmp = { nodes: pieces.map(p => ({ id: p.id, role: p.role, width: 1 })), edges: [], gates: [], rounds: st.rounds, rescue: st.rescue };
    for (let i = 1; i < pieces.length; i++) tmp.edges.push([pieces[i - 1].id, pieces[i].id]);
    const rv = pieces.find(p => p.role === 'reviewer'), bl = [...pieces].reverse().find(p => isBuilder(p.role));
    if (rv && bl) tmp.gates.push({ after: rv.id, on_fail: bl.id });
    wf = generatedWorkflow(tmp);
  }
  const settings = {};
  wf.pieces.forEach(p => {
    const o = take(p.role);
    if (o) { settings[p.id] = settingOf(o); if (isBuilder(p.role)) p.width = Math.max(p.width, o.width); }
    else { const d = defaultSetting(p.role, st); settings[p.id] = settingOf({ ...d, id: p.id, role: p.role }); }
  });
  wf.control = { ...(wf.control || {}), budget_rounds: (wf.control.gates || []).some(g => g.on_fail) ? Math.max(st.rounds, 1) : 1 };
  const cfg = { id: null, workflow: wf, settings };
  const out = fromConfig(cfg);
  out.wf = wf.id === 'custom' ? null : out.wf;
  return out;
}

// ------------------------------------------------------------------ predictions, the trail and the glide
let PRED = null, NUM = null, SHOWN = null, LAST_OK = null, PREV_NUM = null, START = null;
let TRAIL = [], predTok = 0, predTimer = null, nudgeTok = 0;
function schedulePredict(delay) {
  clearTimeout(predTimer);
  const tok = ++predTok, st = clone(S), action = lastAction;
  predTimer = setTimeout(async () => {
    let res;
    try { res = await predictOne(toConfig(st)); }
    catch (e) { if (tok !== predTok) return; apiDown(e); PRED = { ok: false, errors: [{ piece: null, message: 'The prediction failed: ' + (e.message || e) }], failed: true }; NUDGES = []; PREVIEW = PINNED = null; nudgeTok++; paintSide(); paintCanvas(); paintPop(); paintNudges(); drawPreview(); return; }
    if (tok !== predTok) return;
    PRED = res;
    if (!res.ok) { NUDGES = []; paintSide(); paintCanvas(); paintPop(); paintNudges(); return; }
    const num = norm(res.numbers);
    if (!num) { PRED = { ok: false, errors: [{ piece: null, message: 'The reply had no numbers to show' }] }; paintSide(); return; }
    const prevId = LAST_OK && LAST_OK.config_id;
    if (prevId && prevId !== res.config_id) {
      const back = TRAIL.length && TRAIL[TRAIL.length - 1].id === res.config_id;
      if (back) TRAIL.pop(); else { TRAIL.push({ s: LAST_OK.snap, num: NUM, id: prevId }); if (TRAIL.length > 14) TRAIL.shift(); }
    }
    PREV_NUM = prevId && prevId !== res.config_id ? NUM : PREV_NUM;
    const from = SHOWN || NUM;
    NUM = num; LAST_OK = { config_id: res.config_id, snap: JSON.stringify(st), action };
    PREVIEW = PINNED = null;
    paintSide(); paintCanvas(); paintPop(); paintBar(); drawChart();
    glide(from, num, from ? 520 : 0);
    computeNudges();
  }, delay == null ? 60 : delay);
}
function glide(from, to, ms) {
  cancelAnimationFrame(glide.raf);
  if (!from || !ms) { SHOWN = to; drawBuild(); return; }
  const t0 = performance.now();
  const step = now => {
    const k = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - k, 3);
    SHOWN = k >= 1 ? to : lerpNum(from, to, e); drawBuild();
    if (k < 1) glide.raf = requestAnimationFrame(step);
  };
  glide.raf = requestAnimationFrame(step);
}

// ------------------------------------------------------------------ history
const snap = () => JSON.stringify(S);
function change(labelText, fn, opts) {
  past.push({ s: snap(), label: labelText }); if (past.length > 300) past.shift();
  future.length = 0; fn(); lastAction = labelText;
  fixSel(); renderAll();
  if (!opts || opts.predict !== false) schedulePredict();
}
function undo() { const e = past.pop(); if (!e) return; future.push({ s: snap(), label: e.label }); S = JSON.parse(e.s); lastAction = 'Undid: ' + e.label; fixSel(); renderAll(); schedulePredict(); }
function redo() { const e = future.pop(); if (!e) return; past.push({ s: snap(), label: e.label }); S = JSON.parse(e.s); lastAction = 'Redid: ' + e.label; fixSel(); renderAll(); schedulePredict(); }
function fixSel() {
  if (SEL && SEL.kind === 'node' && !S.nodes.some(n => n.id === SEL.id)) SEL = null;
  if (SEL && SEL.kind === 'edge' && !S.edges.some(e => e[0] === SEL.id[0] && e[1] === SEL.id[1])) SEL = null;
  if (SEL && SEL.kind === 'gate' && !S.gates.some(g => g.after === SEL.id)) SEL = null;
}

// ------------------------------------------------------------------ edits
function newId(role) { const p = (ROLES[role] || { idp: role }).idp; if (!S.nodes.some(n => n.id === p)) return p; let k = 2; while (S.nodes.some(n => n.id === p + k)) k++; return p + k; }
function reaches(from, to) { const st = [from], seen = new Set(); while (st.length) { const x = st.pop(); if (x === to) return true; if (seen.has(x)) continue; seen.add(x); S.edges.forEach(e => { if (e[0] === x) st.push(e[1]); }); } return false; }
function addPiece(role, at) {
  const id = newId(role), d = defaultSetting(role, S);
  const from = SEL && SEL.kind === 'node' ? S.nodes.find(n => n.id === SEL.id) : null;
  change('Added a ' + roleName(role) + (from ? ' after ' + from.id : ''), () => {
    let x, y;
    if (at) { x = at.x - NW / 2; y = at.y - NH / 2; }
    else if (from) { x = from.x + COLX; y = from.y; while (S.nodes.some(n => Math.abs(n.x - x) < 40 && Math.abs(n.y - y) < 40)) y += ROWY; }
    else { const mx = S.nodes.length ? Math.max(...S.nodes.map(n => n.x)) : -COLX + 170; x = mx + COLX; y = 200 - NH / 2; }
    S.nodes.push({ id, role, width: 1, model: d.model, effort: d.effort, harness: d.harness, set: d.set ? clone(d.set) : null, x, y });
    if (from) S.edges.push([from.id, id]);
    else if (at) {
      const left = S.nodes.filter(n => n.id !== id && n.x + NW < x + 10).sort((a, b) => (Math.abs(a.y - y) + (x - a.x) * 0.3) - (Math.abs(b.y - y) + (x - b.x) * 0.3))[0];
      if (left && Math.abs(left.y - y) < ROWY * 1.2) S.edges.push([left.id, id]);
    }
    if (role === 'reviewer') { const b = [...S.nodes].reverse().find(n => isBuilder(n.role)); if (b) S.gates.push({ after: id, on_fail: b.id }); }
    S.wf = null;
  });
  SEL = { kind: 'node', id }; POP_OPEN = null; renderAll();
}
function removeNode(id) {
  change('Removed ' + id, () => {
    const ins = S.edges.filter(e => e[1] === id).map(e => e[0]), outs = S.edges.filter(e => e[0] === id).map(e => e[1]);
    S.nodes = S.nodes.filter(n => n.id !== id);
    S.edges = S.edges.filter(e => e[0] !== id && e[1] !== id);
    ins.forEach(a => outs.forEach(b => { if (!S.edges.some(e => e[0] === a && e[1] === b)) S.edges.push([a, b]); }));  // keep the chain joined
    S.gates = S.gates.filter(g => g.after !== id && g.on_fail !== id);
    S.wf = null;
  });
  SEL = null; renderAll();
}
function setNode(id, patch, text) {
  const n = S.nodes.find(x => x.id === id); if (!n) return;
  if (Object.keys(patch).every(k => n[k] === patch[k])) return;
  FLASH = id;
  change(text, () => { Object.assign(S.nodes.find(x => x.id === id), patch); });
}
function setGate(after, onFail) {
  change(onFail ? 'Gate after ' + after + ': if rejected, back to ' + onFail : 'Removed the gate after ' + after, () => {
    S.gates = S.gates.filter(g => g.after !== after);
    if (onFail) { S.gates.push({ after, on_fail: onFail }); if (S.rounds < 2) S.rounds = 2; }
    S.wf = null;
  });
}
function setRounds(v) { if (v !== S.rounds) change('Review rounds: ' + S.rounds + ' to ' + v, () => { S.rounds = v; }); }
function loadConfig(cfg, why, opt) {
  change(why, () => { S = fromConfig(cfg); });
  SEL = null;
  if (opt) START = opt;
  renderAll();
}
function startFrom(opt) {
  PANX = 0; TRAIL = []; TAPPED = null; PREVIEW = PINNED = null;
  loadConfig(opt.cfg, 'Started from option ' + opt.no + ' (' + opt.name.toLowerCase() + ')', opt);
}

// ------------------------------------------------------------------ the toolbar
function miniShape(st) {
  const L = layers(st), cols = {}; st.nodes.forEach(n => (cols[L[n.id]] = cols[L[n.id]] || []).push(n));
  const nc = Object.keys(cols).length, W = nc * 13 + 1, svg = el('svg', { width: W, height: 16, viewBox: `0 0 ${W} 16`, class: 'mini' });
  Object.keys(cols).forEach(c => { const list = cols[c]; const hh = list.length * 6; let y = 8 - hh / 2 + 0.5;
    list.forEach(n => { for (let k = Math.min(n.width, 3) - 1; k >= 0; k--) el('rect', { x: Number(c) * 13 + 1 + k * 1.6, y: y + k * 1.6 - (n.width > 1 ? 1 : 0), width: 9, height: 5, rx: 1.2, fill: k ? '#fffdf8' : roleColor(n.role), stroke: roleColor(n.role), 'stroke-width': 0.8 }, svg); y += 6; }); });
  return svg;
}
function paintBar() {
  const starts = $('starts'); if (!starts) return;
  starts.innerHTML = '';
  D.options.filter(o => !o.retired).forEach(o => {
    const b = h('button', { class: 'chip' + (START && START.id === o.id && START.no === o.no ? ' on' : ''), title: o.title || '' }, `<span class="no">${esc(o.no)}</span>${esc(o.name)}`);
    b.onclick = () => startFrom(o);
    starts.appendChild(b);
  });
  const shapes = $('shapes'); shapes.innerHTML = '';
  const cur = shapeOf(S);
  D.shapes.forEach(s => {
    const st = fromConfig({ workflow: s.workflow, settings: {} });
    const b = h('button', { class: 'chip' + (cur && cur.id === s.id ? ' on' : ''), title: s.best ? 'Drops the best searched version: ' + s.best.label : 'Carries your settings over by role' });
    b.appendChild(miniShape(st)); b.appendChild(document.createTextNode(s.title.replace(/:.*$/, '')));
    b.onclick = () => {
      if (s.best) loadConfig(s.best.config, 'Dropped the ' + s.id.replace(/_/g, ' ') + ' shape (its best searched version)');
      else { const roles = s.workflow.pieces.map(p => p.role); change('Dropped the ' + s.id.replace(/_/g, ' ') + ' shape', () => { S = reshape(S, roles); }); SEL = null; renderAll(); }
    };
    shapes.appendChild(b);
  });
  const pal = $('palette'); pal.innerHTML = '';
  const roles = ['planner', 'implementer', 'reviewer', 'referee', 'worker', 'tester'];
  if (D.roleRuns.integrator) roles.push('integrator');
  roles.forEach(r => {
    const runs = D.roleRuns[r], dis = runs === 0;
    const b = h('button', { class: 'chip piece' + (dis ? ' dis' : '') }, `<span class="grip">::</span><span class="sw" style="background:${roleColor(r)}"></span>${roleName(r)}${dis ? ' <small>no runs yet</small>' : r === 'referee' ? ' <small>referee</small>' : ''}`);
    if (!dis) paletteDrag(b, r);
    pal.appendChild(b);
  });
  pal.appendChild(h('span', { class: 'hint' }, 'drag onto the canvas, or tap to add after the selected piece'));
  $('undo').disabled = !past.length; $('redo').disabled = !future.length;
  $('undo').title = past.length ? 'Undo: ' + past[past.length - 1].label : ''; $('redo').title = future.length ? 'Redo: ' + future[future.length - 1].label : '';
}
function paletteDrag(btn, role) {
  let start = null, ghost = null;
  btn.addEventListener('pointerdown', e => { start = { x: e.clientX, y: e.clientY }; try { btn.setPointerCapture(e.pointerId); } catch (err) { /* synthetic */ } });
  btn.addEventListener('pointermove', e => {
    if (!start) return;
    if (!ghost && Math.hypot(e.clientX - start.x, e.clientY - start.y) > 6) { ghost = h('div', { class: 'ghost' }, `<span style="color:${roleColor(role)}">■</span> ${roleName(role)}`); document.body.appendChild(ghost); }
    if (ghost) { ghost.style.left = e.clientX + 'px'; ghost.style.top = e.clientY + 'px'; }
  });
  const end = e => {
    if (!start) return;
    const wasDrag = !!ghost; if (ghost) ghost.remove(); ghost = null; start = null;
    if (e.type === 'pointercancel') return;
    if (!wasDrag) { addPiece(role); return; }
    const cv = $('cv'), r = cv.getBoundingClientRect();
    if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) { SEL = null; addPiece(role, svgPt(cv, e)); }
  };
  btn.addEventListener('pointerup', end); btn.addEventListener('pointercancel', end);
}

// ------------------------------------------------------------------ the chart (chance against cost)
let XM = 'run', TAPPED = null, PREVIEW = null, PINNED = null;  // PINNED: the row tapped or clicked; hover previews others
const CH = { m: { l: 46, r: 18, t: 18, b: 38 } };
const rings = () => D.ref ? [...D.options, D.ref] : D.options;  // the options, and a reference that is not one
// 22P note 2 (0.2.3): a ring within two radii of one already placed moves outward, away from it, until it is clear,
// with a thin line back to its point, so two options on about the same spot keep readable numbers
const RING_R = 9.5;
function ringSpots(sc) {
  const out = [];
  for (const o of rings()) {
    const x0 = sc.sx(o.num[XM].mean), y0 = sc.sy(o.num.chance.mean);
    let x = x0, y = y0;
    for (let i = 0; i < 8; i++) {
      const near = out.find(p => Math.hypot(p.x - x, p.y - y) < 2 * RING_R - 0.01);
      if (!near) break;
      let dx = x - near.x, dy = y - near.y;
      if (Math.hypot(dx, dy) < 0.5) { dx = 0.7; dy = -0.7; }
      const k = 2 * RING_R / Math.hypot(dx, dy);
      x = near.x + dx * k; y = near.y + dy * k;
    }
    out.push({ o, x, y, x0, y0 });
  }
  return out;
}
// 23R2 note 2 (0.2.3): while the run cost metric leads with the typical run, the chart, which places builds by their
// mean, says so on its run cost value and axis, so the two numbers do not read as one contradicting the other
const meanOnChart = () => XM === 'run' && !!(NUM && NUM.run && NUM.run.typical);
function xDomain() {
  const vals = [...D.cands.map(e => e.num[XM].mean), ...rings().map(o => o.num[XM].mean)].filter(v => v > 0);
  if (!vals.length) return [0.1, 100];
  return [Math.max(0.01, Math.min(...vals) / 1.8), Math.max(...vals) * 1.6];
}
function scales() {
  const svg = $('chart'), w = Math.max(320, svg.parentNode.clientWidth - 16), H = Math.round(clamp(w * 0.37, 270, 360));
  const { l, r, t, b } = CH.m, [x0, x1] = xDomain(), L0 = Math.log(x0), L1 = Math.log(x1);
  const sx = v => l + (Math.log(clamp(v, x0, x1)) - L0) / (L1 - L0) * (w - l - r);
  const sy = p => t + (1 - clamp(p, 0, 1)) * (H - t - b);
  return { sx, sy, x0, x1, w, H };
}
function ivLines(sc, n, cls, widths, opa) {
  const x = sc.sx(n[XM].mean), y = sc.sy(n.chance.mean); let s = '';
  [[90, widths[2]], [80, widths[1]], [50, widths[0]]].forEach(([k, sw]) => {
    const cx = n[XM].b[k], cy = n.chance.b[k];
    s += `<line x1="${sc.sx(cx[0]).toFixed(1)}" x2="${sc.sx(cx[1]).toFixed(1)}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${cls}" stroke-width="${sw}" stroke-opacity="${opa}" stroke-linecap="round"/>`;
    s += `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${sc.sy(cy[0]).toFixed(1)}" y2="${sc.sy(cy[1]).toFixed(1)}" stroke="${cls}" stroke-width="${sw}" stroke-opacity="${opa}" stroke-linecap="round"/>`;
  });
  return s;
}
function drawChart() {
  const svg = $('chart'); if (!svg || !D) return;
  const sc = scales(), { l, r, t, b } = CH.m;
  svg.setAttribute('viewBox', `0 0 ${sc.w} ${sc.H}`); svg.setAttribute('height', sc.H);
  let s = '';
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    const y = sc.sy(p);
    s += `<line x1="${l}" x2="${sc.w - r}" y1="${y}" y2="${y}" class="${p === 0 ? 'c-axis' : 'c-grid'}"/>`;
    s += `<text x="${l - 7}" y="${y + 4}" text-anchor="end" class="c-tick">${pct(p)}</text>`;
  }
  let xt = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].filter(v => v >= sc.x0 && v <= sc.x1);
  xt.forEach(v => { const x = sc.sx(v); s += `<line x1="${x}" x2="${x}" y1="${t}" y2="${sc.H - b}" class="c-grid"/><text x="${x}" y="${sc.H - b + 15}" text-anchor="middle" class="c-tick">$${v}</text>`; });
  s += `<text x="${sc.w - r}" y="${sc.H - 5}" text-anchor="end" class="c-axt">${XM === 'run' ? (meanOnChart() ? 'Mean cost per run' : 'Cost per run') : 'Cost per accepted result'}, log scale →</text>`;
  s += `<text x="${l + 6}" y="${t + 11}" class="c-axt c-lbl" style="fill:var(--ink2)">↑ ${esc(D.rule.score ? 'Chance to reach ' + (D.rule.definition || 'the target') : 'Chance of an accepted result')} in one run</text>`;
  // the best trade-offs among the candidates
  const front = [...D.cands].sort((a, c) => a.num[XM].mean - c.num[XM].mean); let best = -1; const fr = [];
  for (const e of front) if (e.num.chance.mean > best) { best = e.num.chance.mean; fr.push(e); }
  if (fr.length > 1) {
    let d = 'M' + sc.sx(fr[0].num[XM].mean).toFixed(1) + ',' + sc.sy(fr[0].num.chance.mean).toFixed(1);
    for (let i = 1; i < fr.length; i++) d += 'H' + sc.sx(fr[i].num[XM].mean).toFixed(1) + 'V' + sc.sy(fr[i].num.chance.mean).toFixed(1);
    s += `<path d="${d}" class="c-front"/>`;
  }
  if (TAPPED) s += ivLines(sc, TAPPED.num, 'var(--ink)', [1.4, 0.8, 0.45], 0.5);
  for (const e of D.cands) {
    const x = sc.sx(e.num[XM].mean), y = sc.sy(e.num.chance.mean);
    s += e.origin === 'recorded' ? `<rect x="${(x - 3.2).toFixed(1)}" y="${(y - 3.2).toFixed(1)}" width="6.4" height="6.4" class="c-rd" transform="rotate(45 ${x.toFixed(1)} ${y.toFixed(1)})"/>`
      : `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.7" class="c-pt"/>`;
  }
  for (const o of rings()) s += ivLines(sc, o.num, START && START.id === o.id ? 'var(--start)' : 'var(--ink2)', [1.2, 0.7, 0.4], START && START.id === o.id ? 0.5 : 0.22);
  // the trail of earlier builds
  const pts = TRAIL.map(tr => tr.num).concat(NUM ? [NUM] : []);
  if (pts.length > 1) s += `<polyline points="${pts.map(n => sc.sx(n[XM].mean).toFixed(1) + ',' + sc.sy(n.chance.mean).toFixed(1)).join(' ')}" class="c-trail"/>`;
  TRAIL.forEach((tr, i) => { s += `<circle cx="${sc.sx(tr.num[XM].mean).toFixed(1)}" cy="${sc.sy(tr.num.chance.mean).toFixed(1)}" r="4.2" class="c-prev" fill-opacity="${(0.14 + 0.26 * (i + 1) / TRAIL.length).toFixed(2)}"/>`; });
  const spots = ringSpots(sc);
  for (const { x, y, x0, y0 } of spots) if (x !== x0 || y !== y0) s += `<line x1="${x0.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" class="c-lead"/><circle cx="${x0.toFixed(1)}" cy="${y0.toFixed(1)}" r="2" class="c-leadpt"/>`;
  for (const { o, x, y } of spots) {
    const on = START && START.id === o.id && START.no === o.no ? ' on' : '';
    s += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="9.5" class="c-opt${on}${o.retired ? ' old' : ''}"/><text x="${x.toFixed(1)}" y="${(y + 3.8).toFixed(1)}" text-anchor="middle" class="c-optn${on}">${esc(o.no)}</text>`;
    if (o.retired) s += `<text x="${x.toFixed(1)}" y="${(y + 22).toFixed(1)}" text-anchor="middle" class="c-lbl c-old">${esc(o.name.toLowerCase())}, retired model</text>`;
  }
  if (TAPPED) s += `<circle cx="${sc.sx(TAPPED.num[XM].mean).toFixed(1)}" cy="${sc.sy(TAPPED.num.chance.mean).toFixed(1)}" r="8" class="c-tap"/>`;
  s += '<g id="pv"></g><g id="bl"></g>';
  let hits = '';
  for (const e of D.cands) hits += `<circle data-c="${esc(e.id)}" cx="${sc.sx(e.num[XM].mean).toFixed(1)}" cy="${sc.sy(e.num.chance.mean).toFixed(1)}" r="9" fill="transparent"/>`;
  for (const { o, x, y } of spots) hits += `<circle data-o="${esc(o.no)}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="13" fill="transparent"/>`;
  TRAIL.forEach((tr, i) => { hits += `<circle data-trail="${i}" cx="${sc.sx(tr.num[XM].mean).toFixed(1)}" cy="${sc.sy(tr.num.chance.mean).toFixed(1)}" r="11" fill="transparent"/>`; });
  s += `<g style="cursor:pointer">${hits}</g>`;
  svg.innerHTML = s;
  CH.sc = sc;
  drawPreview(); drawBuild();
}
function drawPreview() {
  const g = $('pv'), sc = CH.sc; if (!g || !sc) return;
  const n = PREVIEW; if (!n || !NUM) { g.innerHTML = ''; return; }
  const x0 = sc.sx(NUM[XM].mean), y0 = sc.sy(NUM.chance.mean), x1 = sc.sx(n.num[XM].mean), y1 = sc.sy(n.num.chance.mean);
  const len = Math.hypot(x1 - x0, y1 - y0), ux = (x1 - x0) / (len || 1), uy = (y1 - y0) / (len || 1);
  let s = ivLines(sc, n.num, 'var(--accent)', [1.2, 0.7, 0.4], 0.35);
  if (len > 16) {
    const ex = x1 - ux * 9, ey = y1 - uy * 9, ax = ex - ux * 7, ay = ey - uy * 7;
    s += `<line x1="${(x0 + ux * 12).toFixed(1)}" y1="${(y0 + uy * 12).toFixed(1)}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" class="c-arrow"/>`;
    s += `<path d="M${ex.toFixed(1)},${ey.toFixed(1)}L${(ax - uy * 4).toFixed(1)},${(ay + ux * 4).toFixed(1)}L${(ax + uy * 4).toFixed(1)},${(ay - ux * 4).toFixed(1)}Z" fill="var(--accent)"/>`;
  }
  s += `<circle cx="${x1.toFixed(1)}" cy="${y1.toFixed(1)}" r="7.5" class="c-ghost"/>`;
  const right = x1 < sc.w - 190;
  s += `<text x="${(x1 + (right ? 13 : -13)).toFixed(1)}" y="${(y1 + 17).toFixed(1)}" text-anchor="${right ? 'start' : 'end'}" class="c-you-t" style="font-weight:500;font-size:11.5px">${esc(n.text)}</text>`;
  g.innerHTML = s;
}
function drawBuild() {
  const g = $('bl'), sc = CH.sc, n = SHOWN; if (!g || !sc || !n) return;
  const x = sc.sx(n[XM].mean), y = sc.sy(n.chance.mean);
  let s = ivLines(sc, n, 'var(--accent)', [1.8, 0.9, 0.5], 0.6);
  s += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="16" class="c-halo"/><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="8.5" class="c-you"/>`;
  const txt = 'your build  ' + pct(n.chance.mean) + ', ' + (meanOnChart() ? 'mean ' : '') + usd(n[XM].mean);
  const right = x < sc.w - 180, tx = right ? x + 17 : x - 17, ty = y - 15 < 40 ? y + 27 : y - 15;
  s += `<text x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" text-anchor="${right ? 'start' : 'end'}" class="c-you-t">${txt}</text>`;
  g.innerHTML = s;
}
function paintLegend() {
  const dot = svg => `<svg width="14" height="14" viewBox="-7 -7 14 14">${svg}</svg>`;
  $('legend').innerHTML =
    `<span>${dot('<circle r="6" fill="#2a52be" stroke="#fff" stroke-width="1.5"/>')}your build</span>` +
    `<span>${dot('<circle r="3.8" fill="#2a52be" fill-opacity=".35"/>')}earlier builds, tap to go back</span>` +
    `<span>${dot('<circle r="6" fill="#fff" stroke="#231f1a" stroke-width="1.2"/><text y="3" text-anchor="middle" font-size="7.5" font-weight="700">1</text>')}the options</span>` +
    `<span>${dot('<circle r="2.7" fill="#9a9185" fill-opacity=".6"/>')}${D.cands.length} candidates</span>` +
    `<span>${dot('<rect x="-3.2" y="-3.2" width="6.4" height="6.4" fill="#fff" stroke="#6f665a" transform="rotate(45)"/>')}recorded</span>` +
    `<span>${dot('<path d="M-7,3 H-1 V-3 H7" fill="none" stroke="#1d7a4e" stroke-opacity=".7" stroke-width="1.3" stroke-dasharray="3 2"/>')}best trade-offs</span>` +
    '<span style="color:var(--muted)">lines: 50%, 80%, 90% ranges</span>';
}
function paintTap() {
  const e = TAPPED, box = $('tapcard');
  if (!e) { box.className = 'tapcard empty'; box.textContent = 'Tap any point to see what it is and start from it. Tap a faint blue point to go back to an earlier build.'; return; }
  box.className = 'tapcard';
  const rank = 1 + D.cands.filter(x => x.num.cpa.mean < e.num.cpa.mean - 1e-9).length;
  box.innerHTML = `<div style="min-width:0;flex:1"><div class="t">${esc(e.name ? e.name + ': ' : '')}${esc(e.label)}</div><div class="nums">${pct(e.num.chance.mean)} chance, ${e.num.run.typical ? 'mean ' : ''}${usd(e.num.run.mean)} a run, ${usd(e.num.cpa.mean)} per accepted result, #${rank} of ${D.cands.length}${e.origin === 'recorded' ? ', recorded' : ''}</div>` +
    (e.retired ? '<div class="nums" style="color:var(--start)">Retired model: shown for comparison, not offered as a start point.</div>' : '') +
    `</div><div style="display:flex;gap:8px"><button class="btn" id="tapclose">Close</button>${e.retired ? '' : '<button class="btn primary" id="tapstart">Start from this</button>'}</div>`;
  $('tapclose').onclick = () => { TAPPED = null; paintTap(); drawChart(); };
  if (e.retired) return;
  $('tapstart').onclick = () => { const t = TAPPED; TAPPED = null; if (t.opt) startFrom(t.opt); else loadConfig(t.config, 'Started from ' + t.label); paintTap(); };
}

// ------------------------------------------------------------------ the canvas
let VB = null, DRAG = null, PANX = 0, PANMAX = 0;
const MIN_SCALE = 0.86;
function svgPt(svg, e) { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); }
function nodeBox(n) { const k = Math.min(n.width, 4) - 1; return { x: n.x, y: n.y, w: NW + k * STK, h: NH + k * STK }; }
function ioPos() {
  const ins = S.nodes.filter(n => !S.edges.some(e => e[1] === n.id)), outs = S.nodes.filter(n => !S.edges.some(e => e[0] === n.id));
  const avg = (a, f) => a.length ? a.reduce((s, n) => s + f(n), 0) / a.length : 200;
  const minX = S.nodes.length ? Math.min(...S.nodes.map(n => n.x)) : 170, maxX = S.nodes.length ? Math.max(...S.nodes.map(n => nodeBox(n).x + nodeBox(n).w)) : 400;
  return { src: { x: minX - 132, y: avg(ins, n => n.y + NH / 2) }, snk: { x: maxX + 58, y: avg(outs, n => n.y + NH / 2) }, ins, outs };
}
function computeVB() {
  const io = ioPos();
  let x0 = io.src.x - 16, x1 = io.snk.x + 104, y0 = Infinity, y1 = -Infinity;
  S.nodes.forEach(n => { const b = nodeBox(n); y0 = Math.min(y0, b.y - 16); y1 = Math.max(y1, b.y + b.h + 16); });
  if (S.gates.length) y1 += 50;
  const LY = layers(S); if (S.edges.some(([a, b]) => LY[b] - LY[a] > 1)) y0 -= 58;
  if (!isFinite(y0)) { y0 = 80; y1 = 320; }
  y0 = Math.min(y0, io.src.y - 40, io.snk.y - 40); y1 = Math.max(y1, io.src.y + 40, io.snk.y + 40);
  // a readable scale: the canvas grows to fit the graph, and on a narrow screen scrolls sideways by dragging the background
  const cv = $('cv'), cw = cv.clientWidth || 800, w = Math.max(x1 - x0, 560), hh = Math.max(y1 - y0, 220);
  const sc = clamp(cw / w, MIN_SCALE, 1), want = Math.round(clamp(hh * sc, 230, 440));
  if (Math.abs((cv.clientHeight || 0) - want) > 12) cv.style.height = want + 'px';
  const vw = cw / sc, vh = (cv.clientHeight || want) / sc, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  PANMAX = Math.max(0, w - vw); PANX = clamp(PANX, 0, PANMAX);
  return [PANMAX > 0 ? x0 + PANX : cx - vw / 2, cy - vh / 2, vw, vh];
}
function curve(x1, y1, x2, y2) { const mx = Math.max(40, Math.abs(x2 - x1) / 2); return `M${x1},${y1}C${x1 + mx},${y1} ${x2 - mx},${y2} ${x2},${y2}`; }
const pieceOf = id => PRED && PRED.ok && PRED.pieces ? PRED.pieces.find(p => p.piece === id) : null;
const costOf = pc => pc && pc.run_cost_usd != null ? (typeof pc.run_cost_usd === 'object' ? pc.run_cost_usd : { mean: pc.run_cost_usd }) : null;
const passOf = pc => pc && pc.gate_pass != null ? (typeof pc.gate_pass === 'object' ? pc.gate_pass : { mean: pc.gate_pass }) : null;
const badPieces = () => new Set(PRED && !PRED.ok ? (PRED.errors || []).map(errPiece).filter(Boolean) : []);
function paintCanvas() {
  const cv = $('cv'); if (!cv || !S) return;
  if (!DRAG || DRAG.kind === 'pan') VB = computeVB();
  cv.setAttribute('viewBox', VB.join(' '));
  cv.innerHTML = '';
  const defs = el('defs', {}, cv);
  const mk = el('marker', { id: 'arr', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
  el('path', { d: 'M0,1 L9,5 L0,9 z', fill: '#b3a998' }, mk);
  const mk2 = el('marker', { id: 'arrL', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' }, defs);
  el('path', { d: 'M0,1 L9,5 L0,9 z', fill: '#4a433a' }, mk2);
  const flt = el('filter', { id: 'sh', x: '-20%', y: '-20%', width: '140%', height: '160%' }, defs);
  el('feDropShadow', { dx: 0, dy: 2, stdDeviation: 3, 'flood-color': '#231f1a', 'flood-opacity': 0.10 }, flt);
  const flt2 = el('filter', { id: 'shs', x: '-20%', y: '-20%', width: '140%', height: '170%' }, defs);
  el('feDropShadow', { dx: 0, dy: 5, stdDeviation: 7, 'flood-color': '#2a52be', 'flood-opacity': 0.22 }, flt2);
  const gE = el('g', {}, cv), gL = el('g', {}, cv), gN = el('g', {}, cv), gT = el('g', {}, cv);
  const byId = Object.fromEntries(S.nodes.map(n => [n.id, n]));
  const io = ioPos(), bad = badPieces();
  if (!S.nodes.length) el('text', { x: VB[0] + VB[2] / 2, y: VB[1] + VB[3] / 2, 'text-anchor': 'middle', class: 'empty-t' }, gT, 'Drag a piece here, or pick a shape above');
  const term = (p, t, s) => {
    const g = el('g', {}, gN);
    el('rect', { x: p.x, y: p.y - 22, width: 96, height: 44, rx: 22, class: 'io-box' }, g);
    el('text', { x: p.x + 48, y: p.y - 3, 'text-anchor': 'middle', class: 'io-t' }, g, t);
    el('text', { x: p.x + 48, y: p.y + 12, 'text-anchor': 'middle', class: 'io-s' }, g, s);
  };
  if (S.nodes.length) {
    term(io.src, 'task', 'issue + repo');
    term(io.snk, 'result', PRED && PRED.ok && NUM ? pct(NUM.chance.mean) + ' reach' : 'accepted?');
    io.ins.forEach(n => el('path', { d: curve(io.src.x + 96, io.src.y, n.x, n.y + NH / 2), class: 'edge io', 'marker-end': 'url(#arr)' }, gE));
    io.outs.forEach(n => { const b = nodeBox(n); el('path', { d: curve(b.x + NW, n.y + NH / 2, io.snk.x, io.snk.y), class: 'edge io', 'marker-end': 'url(#arr)' }, gE); });
  }
  const LY = layers(S);
  S.edges.forEach(([a, b]) => {
    const A = byId[a], B = byId[b]; if (!A || !B) return;
    let x1 = A.x + NW, y1 = A.y + NH / 2, x2 = B.x, y2 = B.y + NH / 2, d = curve(x1, y1, x2, y2), mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const between = S.nodes.some(n => n !== A && n !== B && n.x > x1 - 10 && n.x + NW < x2 + 10 && Math.abs(n.y - (A.y + B.y) / 2) < NH);
    if (LY[b] - LY[a] > 1 || between) {
      const top = Math.min(A.y, B.y), lift = 58;
      x1 = A.x + NW * 0.72; y1 = A.y; x2 = B.x + NW * 0.28; y2 = B.y;
      d = `M${x1},${y1}C${x1},${top - lift} ${x2},${top - lift} ${x2},${y2 - 3}`;
      mx = (x1 + x2) / 2; my = top - lift * 0.75;
    }
    const selE = SEL && SEL.kind === 'edge' && SEL.id[0] === a && SEL.id[1] === b;
    el('path', { d, class: 'edge' + (selE ? ' sel' : ''), 'marker-end': 'url(#arr)' }, gE);
    const hit = el('path', { d, class: 'edge-hit' }, gE); hit.dataset.edge = a + '>' + b;
    let art = (ROLES[A.role] || {}).out || 'out';
    if (isBuilder(A.role) && B.role === 'referee') art = 'candidates';
    const tw = art.length * 6.6 + 12;
    el('rect', { x: mx - tw / 2, y: my - 9, width: tw, height: 18, rx: 9, class: 'e-artbg' }, gE);
    el('text', { x: mx, y: my + 3.6, 'text-anchor': 'middle', class: 'e-art' }, gE, art);
  });
  S.gates.forEach(g => {
    const A = byId[g.after], B = byId[g.on_fail]; if (!A || !B) return;
    const ba = nodeBox(A), bb = nodeBox(B);
    const x1 = A.x + NW / 2, y1 = ba.y + ba.h, x2 = B.x + NW / 2 + (B === A ? 30 : 0), y2 = bb.y + bb.h;
    const yb = Math.max(y1, y2) + 46;
    const d = `M${x1},${y1 + 8}C${x1},${yb} ${x2},${yb} ${x2},${y2 + 4}`;
    const selG = SEL && SEL.kind === 'gate' && SEL.id === g.after;
    el('path', { d, class: 'loop' + (selG ? ' sel' : ''), 'marker-end': 'url(#arrL)' }, gL);
    const hit = el('path', { d, class: 'edge-hit' }, gL); hit.dataset.gate = g.after;
    const gp = passOf(pieceOf(g.after));
    const t = 'if rejected: back to ' + g.on_fail + (gp ? ', passes ' + pct(gp.mean) : '') + (S.rounds > 1 ? ', up to ' + S.rounds + ' rounds' : '');
    const tw = t.length * 5.9 + 16, mx = (x1 + x2) / 2, my = yb - 11;
    const pg = el('g', { style: 'cursor:pointer' }, gL); pg.dataset.gate = g.after;
    el('rect', { x: mx - tw / 2, y: my - 10, width: tw, height: 20, rx: 10, class: 'g-pill' }, pg);
    el('text', { x: mx, y: my + 4, 'text-anchor': 'middle', class: 'g-t' }, pg, t);
  });
  S.nodes.forEach(n => {
    const isSel = SEL && SEL.kind === 'node' && SEL.id === n.id;
    const g = el('g', { class: 'node' + (isSel ? ' sel' : '') + (bad.has(n.id) ? ' bad' : '') + (FLASH === n.id ? ' flash' : '') + (DRAG && DRAG.over === n.id ? ' drop' : '') }, gN); g.dataset.node = n.id;
    const k = Math.min(n.width, 4) - 1;
    for (let i = k; i >= 1; i--) el('rect', { x: n.x + i * STK, y: n.y + i * STK, width: NW, height: NH, rx: 10, class: 'n-back' }, g);
    el('rect', { x: n.x, y: n.y, width: NW, height: NH, rx: 10, class: 'n-card', filter: isSel ? 'url(#shs)' : 'url(#sh)' }, g);
    el('rect', { x: n.x, y: n.y + 12, width: 4, height: NH - 24, rx: 2, fill: roleColor(n.role) }, g);
    el('text', { x: n.x + 15, y: n.y + 21, class: 'n-role' }, g, roleName(n.role));
    el('text', { x: n.x + 15, y: n.y + 39, class: 'n-set' }, g, D.short(n.model) + ' · ' + n.effort);
    const c = costOf(pieceOf(n.id));
    el('text', { x: n.x + 15, y: n.y + 57, class: 'n-cost' }, g, c ? usd(c.mean) + ' a run' + (n.width > 1 ? ', ' + n.width + ' copies' : '') : n.id);
    if (!isSel) el('text', { x: n.x + NW - 12, y: n.y + 19, 'text-anchor': 'end', class: 'n-edit' }, g, 'edit');
    if (n.width > 1) {
      el('rect', { x: n.x + NW - 40, y: n.y + 27, width: 28, height: 17, rx: 8.5, fill: roleColor(n.role) }, g);
      el('text', { x: n.x + NW - 26, y: n.y + 39.5, 'text-anchor': 'middle', class: 'n-w' }, g, '×' + n.width);
    }
    const px = n.x + NW, py = n.y + NH / 2;
    const ph = el('circle', { cx: px, cy: py, r: 16, class: 'port-hit' }, g); ph.dataset.port = n.id;
    const pp = el('circle', { cx: px, cy: py, r: 6, class: 'port', stroke: roleColor(n.role) }, g); pp.dataset.port = n.id;
    if (hasGate(n.role)) {
      const hx = n.x + NW / 2, hy = n.y + NH + k * STK + 1;
      const gh = el('g', {}, g); gh.dataset.ghandle = n.id;
      el('circle', { cx: hx, cy: hy, r: 16, fill: 'transparent' }, gh);
      el('circle', { cx: hx, cy: hy, r: 8.5, class: 'ghandle' }, gh);
      el('path', { d: `M${hx - 3.6},${hy - 1.4}a3.8,3.8 0 1 0 3.6,-2.6`, fill: 'none', stroke: '#4a433a', 'stroke-width': 1.3, 'stroke-linecap': 'round' }, gh);
      el('path', { d: `M${hx - 0.6},${hy - 5.6}l1.6,1.6-2,1.2`, fill: 'none', stroke: '#4a433a', 'stroke-width': 1.2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, gh);
    }
  });
  if (DRAG && (DRAG.kind === 'link' || DRAG.kind === 'gate') && DRAG.pt) {
    const A = byId[DRAG.from];
    if (DRAG.kind === 'link') el('path', { d: curve(A.x + NW, A.y + NH / 2, DRAG.pt.x, DRAG.pt.y), class: 'tmp' }, gT);
    else { const b = nodeBox(A); el('path', { d: `M${A.x + NW / 2},${b.y + b.h + 8}Q${(A.x + NW / 2 + DRAG.pt.x) / 2},${Math.max(b.y + b.h, DRAG.pt.y) + 50} ${DRAG.pt.x},${DRAG.pt.y}`, class: 'tmp' }, gT); }
  }
  const sh = shapeOf(S);
  $('cvTitle').textContent = sh ? sh.title.replace(/:.*$/, '') : 'Custom workflow';
  $('cvSub').textContent = S.nodes.length + (S.nodes.length === 1 ? ' piece' : ' pieces') + ', ' + S.nodes.reduce((s, n) => s + n.width, 0) + ' agents a round' + (S.gates.length ? ', up to ' + S.rounds + (S.rounds === 1 ? ' round' : ' rounds') : '') + (PANMAX > 0 ? '; drag the background to see the rest' : '');
  if (FLASH) { const f = FLASH; setTimeout(() => { if (FLASH === f) { FLASH = null; const g = document.querySelector(`#cv [data-node="${CSS.escape(f)}"]`); if (g) g.classList.remove('flash'); } }, 700); }
}
function nodeAt(pt, except) {
  for (let i = S.nodes.length - 1; i >= 0; i--) { const n = S.nodes[i]; if (n.id === except) continue; const b = nodeBox(n); if (pt.x >= b.x - 6 && pt.x <= b.x + b.w + 6 && pt.y >= b.y - 6 && pt.y <= b.y + b.h + 6) return n; }
  return null;
}
function canvasEvents() {
  const cv = $('cv');
  cv.addEventListener('pointerdown', e => {
    const t = e.target, pt = svgPt(cv, e);
    const port = t.closest('[data-port]'), gh = t.closest('[data-ghandle]'), nd = t.closest('[data-node]');
    const edgeHit = t.dataset && t.dataset.edge, gateHit = t.closest('[data-gate]');
    try { cv.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
    const box = $('canvasCard').getBoundingClientRect(), at = { x: e.clientX - box.left, y: e.clientY - box.top };
    if (port) { DRAG = { kind: 'link', from: port.dataset.port, pt }; SEL = null; paintPop(); paintCanvas(); return; }
    if (gh) { DRAG = { kind: 'gate', from: gh.dataset.ghandle, pt }; SEL = null; paintPop(); paintCanvas(); return; }
    if (nd) {
      const n = S.nodes.find(x => x.id === nd.dataset.node);
      DRAG = { kind: 'move', id: n.id, dx: pt.x - n.x, dy: pt.y - n.y, moved: false, before: snap(), sx: e.clientX, sy: e.clientY, wasSel: SEL && SEL.kind === 'node' && SEL.id === n.id };
      S.nodes = [...S.nodes.filter(x => x !== n), n];  // bring to front
      return;
    }
    if (edgeHit) { const [a, b] = edgeHit.split('>'); SEL = { kind: 'edge', id: [a, b], at }; renderAll(); return; }
    if (gateHit) { SEL = { kind: 'gate', id: gateHit.dataset.gate, at }; renderAll(); return; }
    DRAG = { kind: 'pan', sx: e.clientX, p0: PANX, moved: false };
  });
  cv.addEventListener('pointermove', e => {
    if (!DRAG) return;
    if (DRAG.kind === 'pan') {
      const dx = e.clientX - DRAG.sx;
      if (!DRAG.moved && Math.abs(dx) < 6) return;
      if (!DRAG.moved) { DRAG.moved = true; if (SEL) { SEL = null; paintPop(); } }
      if (PANMAX > 0) { PANX = clamp(DRAG.p0 - dx * VB[2] / (cv.clientWidth || 800), 0, PANMAX); paintCanvas(); }
      return;
    }
    const pt = svgPt(cv, e);
    if (DRAG.kind === 'move') {
      if (!DRAG.moved && Math.hypot(e.clientX - DRAG.sx, e.clientY - DRAG.sy) < 6) return;
      if (!DRAG.moved) { DRAG.moved = true; SEL = null; paintPop(); }
      const n = S.nodes.find(x => x.id === DRAG.id); n.x = pt.x - DRAG.dx; n.y = pt.y - DRAG.dy;
    } else { DRAG.pt = pt; const o = nodeAt(pt, DRAG.kind === 'link' ? DRAG.from : null); DRAG.over = o ? o.id : null; }
    paintCanvas();
  });
  const end = e => {
    if (!DRAG) return;
    const d = DRAG; DRAG = null;
    if (e.type === 'pointercancel') { renderAll(); return; }
    if (d.kind === 'pan') { if (!d.moved) { SEL = null; renderAll(); } return; }
    if (d.kind === 'move') {
      if (d.moved) { past.push({ s: d.before, label: 'Moved ' + d.id }); future.length = 0; lastAction = 'Moved ' + d.id; renderAll(); return; }
      SEL = d.wasSel ? null : { kind: 'node', id: d.id };  // a tap opens the piece's editor, a second tap closes it
      renderAll(); return;
    }
    const target = d.over ? S.nodes.find(n => n.id === d.over) : null;
    if (d.kind === 'link' && target) {
      if (S.edges.some(x => x[0] === d.from && x[1] === target.id)) { renderAll(); return; }
      if (reaches(target.id, d.from)) { toast('That link would make a cycle: use a review gate to send work back'); renderAll(); return; }
      change('Connected ' + d.from + ' to ' + target.id, () => { S.edges.push([d.from, target.id]); S.wf = null; });
      return;
    }
    if (d.kind === 'gate' && target && target.id !== d.from) { setGate(d.from, target.id); return; }
    renderAll();
  };
  cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
}

// ------------------------------------------------------------------ the piece editor, anchored to the piece
let POP_OPEN = null, SHOW_ALL_MODELS = false;
function stackPreview(n, role) {
  const svg = el('svg', { width: 90, height: 34, viewBox: '0 0 90 34' });
  for (let i = Math.min(n, 4) - 1; i >= 0; i--) el('rect', { x: 2 + i * 8, y: 2 + i * 5, width: 44, height: 16, rx: 4, fill: '#fffdf8', stroke: i ? '#d9cfbd' : roleColor(role), 'stroke-width': i ? 1 : 1.4 }, svg);
  return svg;
}
function effortRuns(n) {
  const m = D.modelById[n.model];
  if (m && m.byEffort) return e => ((m.byEffort[n.role] || {})[e] || 0);
  return e => (D.recorded[n.role + '|' + n.model + '|' + e] ? 1 : 0);
}
function modelButtons(n) {
  const box = h('div', { class: 'models' });
  const list = D.models.length ? [...D.models] : [{ id: n.model, total: null }];
  if (!list.some(m => m.id === n.model)) list.unshift({ id: n.model, total: null, retired: true, harness: n.harness });
  const withRuns = list.length <= 8 ? list : list.filter(m => m.total == null || m.total > 0 || m.id === n.model);
  const shown = SHOW_ALL_MODELS ? list : withRuns;
  shown.forEach(m => {
    const inRole = m.byRole ? (m.byRole[n.role] || 0) : null;
    const hint = m.retired ? 'retired model' : m.total == null ? '' : m.total === 0 ? 'no runs yet' : inRole != null ? inRole + ' as ' + roleName(n.role) + ', ' + m.total + ' in all' : m.total.toLocaleString() + ' runs behind';
    const b = h('button', { class: 'mopt' + (m.id === n.model ? ' on' : '') + (m.total === 0 ? ' zero' : ''), title: m.id }, `<b>${esc(D.short(m.id))}</b><small>${esc(hint)}</small>`);
    b.onclick = () => {
      if (m.id === n.model) return;
      const hv = m.harness || n.harness, efs = effortsFor(hv);
      const effort = efs.includes(n.effort) ? n.effort : efs[Math.min(efs.length - 1, Math.max(0, efs.indexOf('high')))] || n.effort;
      setNode(n.id, { model: m.id, harness: hv, effort }, n.id + ' model: ' + D.short(n.model) + ' to ' + D.short(m.id));
    };
    box.appendChild(b);
  });
  if (list.length > withRuns.length) {
    const more = h('button', { class: 'more-m' }, SHOW_ALL_MODELS ? 'Show only models with runs' : '+' + (list.length - withRuns.length) + ' models with no runs yet');
    more.onclick = () => { SHOW_ALL_MODELS = !SHOW_ALL_MODELS; paintPop(); };
    box.appendChild(more);
  }
  return box;
}
function seg(opts, cur, onPick) {
  const d = h('div', { class: 'seg' });
  opts.forEach(o => { const b = h('button', { class: o.v === cur ? 'on' : '', title: o.title || null }, esc(o.t) + (o.dot ? '<i></i>' : '')); b.onclick = () => onPick(o.v); d.appendChild(b); });
  return d;
}
function paintPop() {
  const host = $('pop'); if (!host) return;
  if (!SEL || DRAG) { host.innerHTML = ''; POP_OPEN = null; return; }
  const key = SEL.kind + ':' + (Array.isArray(SEL.id) ? SEL.id.join('>') : SEL.id);
  const opening = POP_OPEN !== key; POP_OPEN = key;
  const byId = Object.fromEntries(S.nodes.map(n => [n.id, n]));
  const pop = h('div', { class: 'pop', role: 'dialog' });
  if (!opening) pop.style.animation = 'none';
  const head = (title, sw, small) => {
    const ph = h('div', { class: 'ph' }, `<h3>${sw ? `<span class="sw" style="background:${sw}"></span>` : ''}${esc(title)} ${small ? `<small>${esc(small)}</small>` : ''}</h3>`);
    const x = h('button', { class: 'close', 'aria-label': 'Close' }, '×'); x.onclick = () => { SEL = null; renderAll(); };
    ph.appendChild(x); pop.appendChild(ph);
  };
  if (SEL.kind === 'node' && byId[SEL.id]) {
    const n = byId[SEL.id], pc = pieceOf(n.id);
    head(roleName(n.role), roleColor(n.role), n.id);
    (PRED && !PRED.ok ? PRED.errors || [] : []).filter(e => errPiece(e) === n.id).forEach(e => pop.appendChild(h('div', { class: 'perr' }, esc(errText(e)))));
    const mk = h('div'); mk.appendChild(h('div', { class: 'k' }, 'Model <em>runs behind each</em>')); mk.appendChild(modelButtons(n)); pop.appendChild(mk);
    const runs = effortRuns(n), efs = effortsFor(n.harness);
    const ek = h('div'); ek.appendChild(h('div', { class: 'k' }, 'Effort <em>green dot: recorded runs in this role</em>'));
    ek.appendChild(seg(efs.map(x => ({ v: x, t: x, dot: runs(x) > 0, title: runs(x) ? runs(x) + ' recorded' : null })), n.effort, v => setNode(n.id, { effort: v }, n.id + ' effort: ' + n.effort + ' to ' + v)));
    pop.appendChild(ek);
    const wk = h('div'); wk.appendChild(h('div', { class: 'k' }, 'Width <em>copies in parallel</em>'));
    const st = h('div', { class: 'step' });
    const minus = h('button', { 'aria-label': 'Fewer copies' }, '−'), plus = h('button', { 'aria-label': 'More copies' }, '+');
    minus.disabled = n.width <= 1; plus.disabled = n.width >= 8;
    minus.onclick = () => setNode(n.id, { width: n.width - 1 }, n.id + ' width: ' + n.width + ' to ' + (n.width - 1));
    plus.onclick = () => setNode(n.id, { width: n.width + 1 }, n.id + ' width: ' + n.width + ' to ' + (n.width + 1));
    const sp = h('div', { class: 'stack' }); if (n.width > 1) sp.appendChild(stackPreview(n.width, n.role));
    st.append(minus, h('b', {}, String(n.width)), plus, sp); wk.appendChild(st);
    if (n.width > 1 && isBuilder(n.role) && !S.edges.some(e => e[0] === n.id && byId[e[1]].role === 'referee')) wk.appendChild(h('p', { class: 'help', style: 'margin-top:6px' }, 'Add a selector after it to pick the best copy.'));
    pop.appendChild(wk);
    if (hasGate(n.role)) {
      const g = S.gates.find(x => x.after === n.id), gp = passOf(pc);
      const gk = h('div'); gk.appendChild(h('div', { class: 'k' }, 'Gate <em>if rejected, the work goes back to</em>'));
      const gl = h('div', { class: 'gl' });
      const offb = h('button', { class: 'chip' + (!g ? ' on' : '') }, 'no gate'); offb.onclick = () => { if (g) setGate(n.id, null); }; gl.appendChild(offb);
      S.nodes.filter(m => m.id !== n.id && !hasGate(m.role)).forEach(m => {
        const b = h('button', { class: 'chip' + (g && g.on_fail === m.id ? ' on' : '') }, `<span class="sw" style="background:${roleColor(m.role)}"></span>${esc(m.id)}`);
        b.onclick = () => { if (!g || g.on_fail !== m.id) setGate(n.id, m.id); };
        gl.appendChild(b);
      });
      gk.appendChild(gl);
      if (g) {
        const rk = h('div', { style: 'margin-top:9px' }); rk.appendChild(h('div', { class: 'k' }, 'Rounds <em>tries the gate allows</em>'));
        rk.appendChild(seg([1, 2, 3, 4, 5, 6].map(r => ({ v: r, t: String(r) })), S.rounds, setRounds)); gk.appendChild(rk);
      }
      if (gp) gk.appendChild(h('p', { class: 'help', style: 'margin-top:7px' }, `Approves about <b style="color:var(--ink)">${pct(gp.mean)}</b> of the time${fin(gp.lo) ? ' (80%: ' + pct(gp.lo) + ' to ' + pct(gp.hi) + ')' : ''}.` + (PRED && PRED.rounds && fin(PRED.rounds.mean) ? ` Expected rounds: ${PRED.rounds.mean.toFixed(1)}.` : '')));
      pop.appendChild(gk);
    }
    const c = costOf(pc);
    const row = h('div', { class: 'row2' });
    row.appendChild(h('div', { class: 'pcost' }, c ? `About <b>${usd(c.mean)}</b> a run${n.width > 1 ? ' for ' + n.width + ' copies' : ''}${fin(c.lo) ? `<br><span style="color:var(--muted)">80%: ${usd(c.lo)} to ${usd(c.hi)}</span>` : ''}` : 'No cost yet'));
    const rm = h('button', { class: 'danger' }, 'Remove'); rm.onclick = () => removeNode(n.id); row.appendChild(rm);
    pop.appendChild(row);
  } else if (SEL.kind === 'edge') {
    const [a, b] = SEL.id;
    head('Link');
    pop.appendChild(h('p', { class: 'help' }, `<b style="color:var(--ink)">${esc(a)}</b> hands its output to <b style="color:var(--ink)">${esc(b)}</b>.`));
    const rm = h('button', { class: 'danger' }, 'Remove this link');
    rm.onclick = () => { change('Removed the link ' + a + ' to ' + b, () => { S.edges = S.edges.filter(e => !(e[0] === a && e[1] === b)); S.wf = null; }); SEL = null; renderAll(); };
    pop.appendChild(rm);
  } else if (SEL.kind === 'gate') {
    const g = S.gates.find(x => x.after === SEL.id); if (!g) { host.innerHTML = ''; return; }
    head('Review gate');
    pop.appendChild(h('p', { class: 'help' }, `When <b style="color:var(--ink)">${esc(g.after)}</b> rejects the work, <b style="color:var(--ink)">${esc(g.on_fail)}</b> redoes it, up to the round limit.`));
    const rk = h('div'); rk.appendChild(h('div', { class: 'k' }, 'Rounds')); rk.appendChild(seg([1, 2, 3, 4, 5, 6].map(r => ({ v: r, t: String(r) })), S.rounds, setRounds)); pop.appendChild(rk);
    const rm = h('button', { class: 'danger' }, 'Remove this gate'); rm.onclick = () => { setGate(g.after, null); SEL = null; renderAll(); };
    pop.appendChild(rm);
  } else { host.innerHTML = ''; return; }
  host.innerHTML = ''; host.appendChild(pop);
  placePop(pop);
}
function placePop(pop) {
  const card = $('canvasCard'), cv = $('cv'), cr = card.getBoundingClientRect();
  const W = Math.min(356, cr.width - 24); pop.style.width = W + 'px';
  const ph = pop.offsetHeight;
  let anchor;
  if (SEL.kind === 'node') {
    const n = S.nodes.find(x => x.id === SEL.id), b = nodeBox(n), m = cv.getScreenCTM();
    const p0 = new DOMPoint(b.x, b.y).matrixTransform(m), p1 = new DOMPoint(b.x + b.w, b.y + b.h).matrixTransform(m);
    anchor = { l: p0.x - cr.left, t: p0.y - cr.top, r: p1.x - cr.left, b: p1.y - cr.top };
  } else { const a = SEL.at || { x: cr.width / 2, y: 120 }; anchor = { l: a.x - 4, t: a.y - 4, r: a.x + 4, b: a.y + 4 }; }
  const tip = h('span', { class: 'tip' }); pop.appendChild(tip);
  const cy = (anchor.t + anchor.b) / 2, cx = (anchor.l + anchor.r) / 2, gap = 12;
  let x, y;
  if (cr.width - anchor.r >= W + gap + 10) {  // right of the piece
    x = anchor.r + gap; y = clamp(cy - 44, 8, Math.max(8, cr.height - ph + 160)); tip.className = 'tip left'; tip.style.top = clamp(cy - y - 7, 14, ph - 24) + 'px';
  } else if (anchor.l >= W + gap + 10) {  // left of it
    x = anchor.l - gap - W; y = clamp(cy - 44, 8, Math.max(8, cr.height - ph + 160)); tip.className = 'tip right'; tip.style.top = clamp(cy - y - 7, 14, ph - 24) + 'px';
  } else {  // below it, the tip pointing up at the piece
    x = clamp(cx - W / 2, 10, cr.width - W - 10); y = anchor.b + gap; tip.className = 'tip up'; tip.style.left = clamp(cx - x - 7, 16, W - 30) + 'px';
  }
  pop.style.left = x + 'px'; pop.style.top = y + 'px';
}

// ------------------------------------------------------------------ next steps: every one-step change, predicted in one call
let NUDGES = [], RANKBY = 'chance', SHOW_ALL_NUDGES = false;
function otherModels(n) {  // the offered models on the same harness, most runs first (the catalog's order)
  const hv = modelHarness(n.model) || n.harness;
  const same = D.models.filter(m => m.id !== n.model && (m.harness || null) === (hv || null));
  return (same.length ? same : D.models.filter(m => m.id !== n.model).slice(0, 1)).slice(0, 3).map(m => m.id);
}
function neighbours(st) {
  const out = [], multi = r => st.nodes.filter(q => q.role === r).length > 1;
  const nm = n => roleName(n.role) + (multi(n.role) ? ' ' + n.id : '');
  const tweak = (text, kind, piece, fn) => { const t = clone(st); fn(t); out.push({ text, kind, piece, st: t }); };
  for (const n of st.nodes) {
    const efs = effortsFor(n.harness), ei = efs.indexOf(n.effort);
    for (const di of [1, -1]) {
      const ne = efs[ei + di]; if (ei < 0 || !ne) continue;
      tweak(nm(n) + ' effort ' + n.effort + ' to ' + ne, 'effort', n.id, t => { t.nodes.find(q => q.id === n.id).effort = ne; });
    }
    for (const om of otherModels(n)) tweak(nm(n) + ' model ' + D.short(n.model) + ' to ' + D.short(om), 'model', n.id, t => {
      const q = t.nodes.find(x => x.id === n.id), hv = modelHarness(om) || q.harness, e2 = effortsFor(hv);
      q.model = om; q.harness = hv; if (!e2.includes(q.effort)) q.effort = e2[e2.length - 1];
    });
    if (isBuilder(n.role)) {
      if (n.width < 8) tweak(nm(n) + ' width ' + n.width + ' to ' + (n.width + 1), 'width', n.id, t => { t.nodes.find(q => q.id === n.id).width++; });
      if (n.width > 1) tweak(nm(n) + ' width ' + n.width + ' to ' + (n.width - 1), 'width', n.id, t => { t.nodes.find(q => q.id === n.id).width--; });
    }
  }
  const roles = st.nodes.map(n => n.role);
  const restructure = (text, kind, rs) => { const t = reshape(st, rs); out.push({ text, kind, st: t }); };
  const addText = r => { const d = defaultSetting(r, st); return 'add a ' + roleName(r) + ' (' + D.short(d.model) + '/' + d.effort + ')'; };
  if (!roles.includes('planner')) restructure(addText('planner'), 'add', [...roles, 'planner']);
  if (!roles.includes('reviewer')) restructure(addText('reviewer'), 'add', [...roles, 'reviewer']);
  if (!roles.includes('referee') && st.nodes.some(n => isBuilder(n.role) && n.width > 1)) restructure(addText('referee'), 'add', [...roles, 'referee']);
  for (const r of ['planner', 'reviewer', 'referee']) if (roles.includes(r) && roles.length > 1) {
    const rest = [...roles]; rest.splice(rest.indexOf(r), 1); restructure('remove the ' + roleName(r), 'remove', rest);
  }
  if (st.gates.length) {
    if (st.rounds < 6) tweak('review rounds ' + st.rounds + ' to ' + (st.rounds + 1), 'rounds', null, t => { t.rounds++; });
    if (st.rounds > 1) tweak('review rounds ' + st.rounds + ' to ' + (st.rounds - 1), 'rounds', null, t => { t.rounds--; });
  }
  return out;
}
async function computeNudges() {
  const tok = ++nudgeTok, base = NUM, list = neighbours(S);
  $('nnote').innerHTML = ''; if (!NUDGES.length) $('nudges').innerHTML = '<div class="nnote">Predicting every one-step change…</div>';
  let res;
  try { res = await predictMany(list.map(n => toConfig(n.st))); }
  catch (e) { if (tok !== nudgeTok) return; apiDown(e); NUDGES = []; $('nudges').innerHTML = `<div class="nnote" style="color:var(--bad)">The next steps could not be predicted: ${esc(e.message || e)}</div>`; return; }
  if (tok !== nudgeTok) return;
  NUDGES = list.map((n, i) => ({ ...n, res: res[i], num: res[i] && res[i].ok ? norm(res[i].numbers) : null }))
    .filter(n => n.num && n.res.config_id !== (PRED && PRED.config_id))
    .map(n => ({ ...n, dC: n.num.chance.mean - base.chance.mean, dR: n.num.run.mean - base.run.mean, dA: n.num.cpa.mean - base.cpa.mean }));
  NUDGES.skipped = list.length - NUDGES.length;
  sortNudges(); paintNudges();
}
function sortNudges() {
  const k = RANKBY;
  NUDGES.sort((a, b) => k === 'chance' ? b.dC - a.dC || a.dA - b.dA : k === 'run' ? a.dR - b.dR || b.dC - a.dC : a.dA - b.dA || b.dC - a.dC);
}
function paintNudges() {
  const box = $('nudges'), note = $('nnote'); if (!box) return;
  if (PRED && !PRED.ok) { box.innerHTML = ''; note.innerHTML = PRED.failed ? '<div class="nnote" style="color:var(--bad)">No next steps: loopmath builder is not answering.</div>' : '<div class="nnote">Fix the build first: the next steps start from a build that can run.</div>'; $('nfoot').innerHTML = ''; return; }
  const list = SHOW_ALL_NUDGES ? NUDGES : NUDGES.slice(0, 8);
  const maxC = Math.max(0.05, ...NUDGES.map(n => Math.abs(n.dC)));
  const top = NUDGES[0];
  const none = top && ((RANKBY === 'cpa' && top.dA > -0.005) || (RANKBY === 'chance' && top.dC < 0.005) || (RANKBY === 'run' && top.dR > -0.005));
  const what = { chance: 'raises the chance', cpa: 'lowers the cost per accepted result', run: 'lowers the run cost' }[RANKBY];
  note.innerHTML = none ? `<div class="nnote">Nothing one step away ${what}: your build is already the best nearby by this measure. The closest steps follow.</div>` : '';
  box.innerHTML = list.map((n, i) => {
    const bw = Math.round(Math.abs(n.dC) / maxC * 24);
    const bar = `<svg width="52" height="10" viewBox="0 0 52 10"><line x1="26" x2="26" y1="0" y2="10" stroke="#d5cab5"/><rect x="${n.dC >= 0 ? 26 : 26 - bw}" y="2.5" width="${bw}" height="5" rx="2" fill="${n.dC >= 0 ? 'var(--good)' : 'var(--bad)'}" fill-opacity=".75"/></svg>`;
    const piece = n.piece ? S.nodes.find(q => q.id === n.piece) : null;
    const j = NUDGES.indexOf(n);
    return `<div class="nudge${PREVIEW === n ? ' pv' : ''}" data-n="${j}"><span class="no">${i + 1}</span>` +
      `<span class="what">${piece ? `<span class="sw" style="background:${roleColor(piece.role)}"></span>` : ''}${esc(n.text)}<small>${pct(n.num.chance.mean)}, ${usd(n.num.run.mean)} a run, ${usd(n.num.cpa.mean)} per accepted</small></span>` +
      `<span class="dc ${tone(n.dC, true, 0.005)}">${bar}${signed(n.dC, 'pts')}</span>` +
      `<span class="dv ${tone(n.dR, false, 0.005)}">${signed(n.dR)}</span>` +
      `<span class="dv ${tone(n.dA, false, 0.005)}">${signed(n.dA)}</span>` +
      `<button class="btn" data-apply="${j}">Apply</button></div>`;
  }).join('') || '<div class="nnote">No one-step change could be predicted for this build.</div>';
  const foot = $('nfoot');
  foot.innerHTML = `<span>${NUDGES.length} steps predicted${API.many ? ' in one call' : ''}${NUDGES.skipped ? ', ' + NUDGES.skipped + ' left out (same build or cannot run)' : ''}</span>` +
    (NUDGES.length > 8 ? `<button class="btn" id="nmore">${SHOW_ALL_NUDGES ? 'Show the top 8' : 'Show all ' + NUDGES.length}</button>` : '');
  if ($('nmore')) $('nmore').onclick = () => { SHOW_ALL_NUDGES = !SHOW_ALL_NUDGES; paintNudges(); };
}
function applyNudge(n) {
  PREVIEW = PINNED = null;
  if (n.piece) FLASH = n.piece;
  change('Next step: ' + n.text, () => { const keep = Object.fromEntries(S.nodes.map(q => [q.id, q])); S = clone(n.st); if (n.kind !== 'add' && n.kind !== 'remove') S.nodes.forEach(q => { if (keep[q.id]) { q.x = keep[q.id].x; q.y = keep[q.id].y; } }); });
}

// ------------------------------------------------------------------ the sidebar: your build
const OPEN_INFO = {}, ANIM = {};
function tweenText(node, key, to, fmt) {
  const from = ANIM[key] != null ? ANIM[key] : to; ANIM[key] = to;
  if (from === to || !fin(from)) { node.textContent = fmt(to); return; }
  const t0 = performance.now(), dur = 420;
  const step = t => { const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3); node.textContent = fmt(from + (to - from) * e); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
const DOM = {
  chance: { kind: 'lin', lo: 0, hi: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: x => Math.round(x * 100) + '%' },
  within: { kind: 'lin', lo: 0, hi: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: x => Math.round(x * 100) + '%' },
  run: { kind: 'log', lo: 0.05, hi: 200, ticks: [0.1, 1, 10, 100], fmt: x => '$' + x },
  cpa: { kind: 'log', lo: 0.1, hi: 200, ticks: [0.1, 1, 10, 100], fmt: x => '$' + x }
};
function pos(k, x) { const d = DOM[k]; const t = d.kind === 'log' ? (Math.log(clamp(x, d.lo, d.hi)) - Math.log(d.lo)) / (Math.log(d.hi) - Math.log(d.lo)) : (clamp(x, d.lo, d.hi) - d.lo) / (d.hi - d.lo); return clamp(t, 0, 1) * 100; }
function stripHTML(k, m, recMean) {
  const ln = lv => { const b = m.b[lv]; if (!b) return ''; const a = pos(k, b[0]), z = pos(k, b[1]); return `<span class="ln l${lv}" data-a="${k}-${lv}" style="left:${a}%;width:${Math.max(z - a, 0.5)}%"></span>`; };
  return `<div class="strip"><span class="trk"></span>${DOM[k].ticks.map(t => `<span class="tk" style="left:${pos(k, t)}%"></span>`).join('')}${ln(95)}${ln(90)}${ln(80)}${ln(50)}${recMean != null ? `<span class="rm" data-a="${k}-rm" style="left:${pos(k, recMean)}%" title="the recommended option"></span>` : ''}<span class="dt" data-a="${k}-dt" style="left:${pos(k, m.mean)}%"></span></div>` +
    `<div class="ticks">${DOM[k].ticks.map(t => `<span style="left:${pos(k, t)}%">${DOM[k].fmt(t)}</span>`).join('')}</div>`;
}
function snapshotMarks(root) { const o = {}; root.querySelectorAll('[data-a]').forEach(x => { o[x.dataset.a] = [x.style.left, x.style.width]; }); return o; }
function animateMarks(root, old) {
  const to = [];
  root.querySelectorAll('[data-a]').forEach(x => { const o = old[x.dataset.a]; if (!o) return; to.push([x, x.style.left, x.style.width]); x.style.transition = 'none'; x.style.left = o[0]; if (o[1]) x.style.width = o[1]; });
  if (!to.length) return;
  void root.offsetWidth;
  to.forEach(([x, l, w]) => { x.style.transition = ''; x.style.left = l; if (w) x.style.width = w; });
}
function headline(r, rr) {
  const dp = Math.round((r.chance.mean - rr.chance.mean) * 100), dc = r.run.mean - rr.run.mean;
  if (dp === 0 && Math.abs(dc) < 0.005) return 'The same numbers as the recommended option.';
  const a = dp === 0 ? 'The same chance' : `<span class="${dp > 0 ? 'up' : 'dn'}">${dp > 0 ? '+' : '−'}${Math.abs(dp)} points of chance</span>`;
  const c = Math.abs(dc) < 0.005 ? 'at the same run cost' : dc > 0 ? `for <span class="dn">+${usd(dc)}</span> a run` : `and <span class="up">${usd(-dc)} less</span> a run`;
  return `${a} ${c}, against the recommended option.`;
}
function badgeOf() {
  if (!PRED) return '<span class="badge">estimating</span>';
  if (!PRED.ok) return PRED.failed ? '<span class="badge bad">no numbers</span>' : '<span class="badge bad">cannot run</span>';
  const id = PRED.config_id, cand = D.byId[id];
  if (S && D.models.length && S.nodes.some(n => !D.modelById[n.model])) return '<span class="badge bad">retired model</span>';
  if (id && id === D.goalId) return '<span class="badge rec">recommended</span>';
  if ((cand && cand.origin === 'recorded') || PRED.runs_behind > 0) return `<span class="badge rcd">recorded${PRED.runs_behind > 0 ? ', ' + PRED.runs_behind + ' runs' : ''}</span>`;
  if (cand || PRED.known) return '<span class="badge srch">in the search</span>';
  return '<span class="badge new">new</span>';
}
function lastEditLine() {
  if (!lastAction) return '';
  let effect = '';
  if (PREV_NUM && NUM && LAST_OK && LAST_OK.action === lastAction && !/^Started/.test(lastAction)) {
    effect = ': chance ' + pct(PREV_NUM.chance.mean) + ' to ' + pct(NUM.chance.mean) + ', ' + usd(PREV_NUM.run.mean) + ' to ' + usd(NUM.run.mean) + ' a run, ' + usd(PREV_NUM.cpa.mean) + ' to ' + usd(NUM.cpa.mean) + ' per accepted result';
  }
  return `<div class="lastedit"><b>${esc(lastAction)}</b>${esc(effect)}</div>`;
}
function paintSide() {
  const box = $('side'); if (!box || !D) return;
  const ok = PRED && PRED.ok, show = NUM, rr = D.recNum;
  const isRec = ok && PRED.config_id === D.goalId;
  let s = `<div class="bhd"><h2>Your build</h2>${badgeOf()}</div>`;
  s += `<p class="blabel">${esc(ok ? PRED.label : labelOf(S))}${ok && PRED.config_id ? ` <span style="color:var(--muted)">${esc(PRED.config_id)}</span>` : ''}</p>`;
  if (PRED && !PRED.ok) {
    const byPiece = e => errPiece(e) ? `<b>${esc(errPiece(e))}</b>: ` : '';
    s += `<div class="errs"><b>${PRED.failed ? 'No numbers right now.' : 'This build cannot run yet.'}</b><ul>${(PRED.errors || []).map(e => `<li>${byPiece(e)}${esc(errText(e))}</li>`).join('')}</ul>${NUM ? 'The numbers below are the last build that could run.' : ''}</div>`;
  }
  if (show && rr && !isRec && !(PRED && PRED.failed)) s += `<div class="headline">${headline(show, rr)}</div>`;
  s += lastEditLine();
  if (show) {
    const R = D.rescue.usd, att = show.within && show.within.attempts, typ = show.run.typical;
    // N3 (0.2.3): while the user has no runs of their own (or the range is wider than 10x) the run cost leads with the
    // typical run, the median, and its difference from the recommended one; the mean follows on the next line
    const meanLine = !typ ? '' : `<div class="rmean" id="runmean">Mean ${usd(show.run.mean)} a run${show.run.mean > show.run.median ? ': a few runs cost far more' : ''}.` +
      (D.ownRuns === 0 ? ` Onboard to see your own costs: <code>loopmath onboard</code>` : '') + `</div>`;
    const mets = [
      { k: 'chance', t: chanceName(), v: show.chance, rec: rr && rr.chance.mean, f: pct, up: true, pts: true,
        x: `The chance that one run of this build reaches ${esc(D.rule.definition || 'the target')}. The dot is the mean; the lines are the 50%, 80%, 90% and 95% ranges, thinner as they widen. Wide ranges mean few runs sit behind these settings.` },
      { k: 'run', t: typ ? 'Typical run cost' : 'Run cost', v: show.run, big: typ ? show.run.median : show.run.mean,
        rec: rr && (typ && fin(rr.run.median) ? rr.run.median : rr.run.mean), f: usd, up: false, after: meanLine,
        x: typ ? `What the agents cost for a typical run of this build (the median), at list prices (not billed spend). The mean, ${usd(show.run.mean)}, is what sums and the cost per accepted result use.`
          : `What the agents cost for one run of this build, at list prices (not billed spend)${fin(show.run.median) ? '; the median run costs ' + usd(show.run.median) : ''}.` },
      { k: 'cpa', t: 'Cost per accepted result', v: show.cpa, rec: rr && rr.cpa.mean, f: usd, up: false,
        x: `The run cost plus the expected cost of fixing a miss: ${usd(show.run.mean)} + ${pct(1 - show.chance.mean)} chance of a miss × ${usd(R)} = ${usd(show.cpa.mean)}. ${esc(D.rescue.text || '')}` },
      { k: 'within', t: 'Chance within ' + (att || 3) + ' attempts', v: show.within, rec: rr && rr.within && rr.within.mean, f: pct, up: true, pts: true,
        x: `The chance of an accepted result within ${att || 3} attempts: this build first, then, after a miss, the rescue${D.rescue.basis ? ' (' + esc(D.rescue.basis) + ')' : ''}.` }
    ];
    s += `<div class="${ok ? '' : 'dim'}">`;
    mets.forEach(m => {
      if (!m.v) return;
      let dt = '';
      if (m.rec != null && !isRec) {
        const d = (m.big != null ? m.big : m.v.mean) - m.rec;
        dt = `<span class="md ${tone(d, m.up, 0.005)}">${m.pts ? signed(d, 'pts') : signed(d)}<small>vs recommended</small></span>`;
      }
      s += `<div class="metric"><span class="t1"><span class="ml">${esc(m.t)}</span><button class="info" data-info="${m.k}" aria-expanded="${OPEN_INFO[m.k] ? 'true' : 'false'}" aria-label="Explain">?</button></span>` +
        `<div class="bigrow"><span class="big" data-big="${m.k}"></span><span class="rng">80%: ${m.f(m.v.b[80][0])}<br>to ${m.f(m.v.b[80][1])}</span>${dt}</div>` +
        (m.after || '') + stripHTML(m.k, m.v, isRec ? null : (m.big != null ? rr && rr[m.k].mean : m.rec)) + (OPEN_INFO[m.k] ? `<div class="explain">${m.x}</div>` : '') + '</div>';
    });
    s += '</div>';
    s += `<div class="slegend"><span><svg width="58" height="12"><line x1="2" y1="6" x2="56" y2="6" stroke="#2a52be" stroke-opacity=".35" stroke-width="1"/><line x1="9" y1="6" x2="49" y2="6" stroke="#2a52be" stroke-opacity=".45" stroke-width="2"/><line x1="16" y1="6" x2="42" y2="6" stroke="#2a52be" stroke-opacity=".6" stroke-width="4"/><line x1="23" y1="6" x2="35" y2="6" stroke="#2a52be" stroke-opacity=".85" stroke-width="6"/></svg>50, 80, 90, 95%</span>${isRec ? '' : '<span><svg width="4" height="14"><rect x="1" y="0" width="2" height="14" fill="#231f1a"/></svg>recommended</span>'}<span>costs on a log scale</span></div>`;
    if (ok) {
      const rk = PRED.rank || {};
      const behind = PRED.runs_behind > 0 ? `<b>${PRED.runs_behind}</b> recorded runs of exactly this build` : 'No recorded run of exactly this build: the estimate borrows from its pieces';
      s += `<div class="facts">${rk.by_cost_per_accepted ? `<div>Rank <b>${rk.by_cost_per_accepted}</b> of ${rk.of} by cost per accepted result</div>` : ''}<div>${behind}</div>${PRED.rounds && fin(PRED.rounds.mean) && PRED.rounds.max > 1 ? `<div>Expected review rounds <b>${PRED.rounds.mean.toFixed(1)}</b> of up to ${PRED.rounds.max}</div>` : ''}</div>`;
      const ps = (PRED.pieces || []).map(p => ({ p, c: costOf(p) })).filter(x => x.c);
      const tot = ps.reduce((a, x) => a + x.c.mean, 0) || 1;
      if (ps.length) {
        s += '<div class="pshare">' + ps.map(({ p, c }) => `<button class="pr" data-piece="${esc(p.piece)}"><span class="nm"><span class="sw" style="background:${roleColor(p.role)}"></span>${esc(roleName(p.role))}<small>${esc(D.short(p.model) + ' ' + p.effort + (p.width > 1 ? ' ×' + p.width : ''))}</small></span><span class="b"><i style="width:${Math.round(100 * c.mean / tot)}%;background:${roleColor(p.role)}"></i></span><span class="v">${usd(c.mean)}</span></button>`).join('') + '</div>';
      }
      const warns = (PRED.warnings || []).map(errText).filter(w => !/harness .* filled in/.test(w));
      if (warns.length) s += `<div class="warns">${warns.map(esc).join('<br>')}</div>`;
    }
  }
  s += `<div class="bact"><button class="btn primary" id="copybtn"${ok ? '' : ' disabled'}>Copy build</button><button class="btn" id="resetbtn"${START && ok && PRED.config_id === START.id ? ' disabled' : ''}>Reset</button></div>`;
  const old = snapshotMarks(box);
  box.innerHTML = s;
  animateMarks(box, old);
  const BIG = show ? { run: show.run.typical ? show.run.median : show.run.mean } : {};
  box.querySelectorAll('[data-big]').forEach(b => { const k = b.dataset.big, v = show[k]; tweenText(b, k, BIG[k] != null ? BIG[k] : v.mean, k === 'chance' || k === 'within' ? pct : usd); });
  box.querySelectorAll('[data-info]').forEach(b => { b.onclick = () => { OPEN_INFO[b.dataset.info] = !OPEN_INFO[b.dataset.info]; paintSide(); }; });
  box.querySelectorAll('[data-piece]').forEach(b => { b.onclick = () => { SEL = { kind: 'node', id: b.dataset.piece }; renderAll(); $('canvasCard').scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }; });
  $('copybtn').onclick = copyBuild;
  $('resetbtn').onclick = () => { if (START) loadConfig(START.cfg, 'Reset to option ' + START.no + ' (' + START.name.toLowerCase() + ')'); };
}
function diffFrom(cfg) {
  const a = fromConfig(cfg), out = [];
  const byRole = st => { const m = {}; st.nodes.forEach(n => { (m[n.role] = m[n.role] || []).push(n); }); return m; };
  const A = byRole(a), B = byRole(S);
  new Set([...Object.keys(A), ...Object.keys(B)]).forEach(r => {
    const x = A[r] || [], y = B[r] || [];
    if (!x.length) { y.forEach(n => out.push(roleName(r) + ' added: ' + n.model + '/' + n.effort)); return; }
    if (!y.length) { out.push(roleName(r) + ' removed'); return; }
    const p = x[0], q = y[0];
    if (p.model !== q.model) out.push(roleName(r) + ' model ' + p.model + ' to ' + q.model);
    if (p.effort !== q.effort) out.push(roleName(r) + ' effort ' + p.effort + ' to ' + q.effort);
    if (p.width !== q.width) out.push(roleName(r) + ' width ' + p.width + ' to ' + q.width);
  });
  if (a.rounds !== S.rounds) out.push('review rounds ' + a.rounds + ' to ' + S.rounds);
  return out;
}
async function copyBuild() {
  if (!PRED || !PRED.ok) return;
  const ds = START ? diffFrom(START.cfg) : [];
  const from = START ? 'option ' + START.no + ' (' + START.name.toLowerCase() + ')' + (ds.length ? ' with ' + ds.join(', ') : '') : '';
  const txt = 'Use workflow ' + PRED.config_id + ': ' + PRED.label + (from ? ' [' + from + ']' : '');
  try { await navigator.clipboard.writeText(txt); toast('Copied: paste it to your agent'); }
  catch (e) {
    const ta = document.createElement('textarea'); ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); toast('Copied: paste it to your agent'); } catch (err) { toast(txt); }
    ta.remove();
  }
}

// ------------------------------------------------------------------ page
function paintHeader() {
  const t = D.task, r = D.rule, f = D.fit, R = D.rescue;
  $('taskT').textContent = t.title || ((t.type || 'A') + ' task in ' + (t.repo || 'this repo') + ', reach ' + (r.definition || 'the target'));
  const bits = [];
  if (f.id) bits.push(`<b>${esc(f.id)}</b>` + (f.runs ? ' on ' + Number(f.runs).toLocaleString() + ' runs' : ''));
  if (fin(R.usd)) bits.push(`a miss costs <b>${usd(R.usd)}</b> to fix: ${esc(R.basis || R.of || 'the rescue')}`);
  $('taskS').innerHTML = bits.join('  ·  ');
  $('taskS').title = R.text || '';
  $('foot').textContent = 'Numbers from loopmath builder: the fitted model, computed as recommend computes them (fit ' + (f.id || 'n/a') + '). Costs are list prices, not billed spend. Ranges missing from a reply are derived from its 80% range.';
}
function renderAll() { paintBar(); paintCanvas(); paintPop(); paintSide(); }
function wire() {
  $('undo').onclick = undo; $('redo').onclick = redo;
  $('xsel').onclick = e => { const b = e.target.closest('[data-x]'); if (!b) return; XM = b.dataset.x; [...$('xsel').children].forEach(c => c.classList.toggle('on', c === b)); drawChart(); };
  $('rankby').onclick = e => { const b = e.target.closest('[data-r]'); if (!b) return; RANKBY = b.dataset.r; [...$('rankby').children].forEach(c => c.classList.toggle('on', c === b)); sortNudges(); paintNudges(); };
  $('nudges').onclick = e => {
    const ap = e.target.closest('[data-apply]'); if (ap) { applyNudge(NUDGES[+ap.dataset.apply]); return; }
    const row = e.target.closest('[data-n]'); if (!row) return;
    const n = NUDGES[+row.dataset.n]; PINNED = PINNED === n ? null : n; PREVIEW = PINNED; paintNudges(); drawPreview();
  };
  $('nudges').onmouseover = e => {
    if (matchMedia('(hover: none)').matches) return;
    const row = e.target.closest('[data-n]'); if (!row) return; const n = NUDGES[+row.dataset.n];
    if (PREVIEW !== n) { PREVIEW = n; drawPreview(); document.querySelectorAll('.nudge').forEach(r => r.classList.toggle('pv', r === row)); }
  };
  $('nudges').onmouseleave = () => { if (matchMedia('(hover: none)').matches) return; PREVIEW = PINNED; drawPreview(); document.querySelectorAll('.nudge').forEach(r => r.classList.toggle('pv', PINNED != null && NUDGES[+r.dataset.n] === PINNED)); };
  $('chart').onclick = e => {
    const t = e.target;
    if (t.dataset.trail != null) {
      const i = +t.dataset.trail, tr = TRAIL[i]; TRAIL = TRAIL.slice(0, i);
      change('Went back to an earlier build', () => { S = JSON.parse(tr.s); }); SEL = null; renderAll(); return;
    }
    if (t.dataset.o) { const o = rings().find(x => String(x.no) === t.dataset.o); if (o) TAPPED = { opt: o, name: 'Option ' + o.no + ' (' + o.name.toLowerCase() + ')', label: o.label, num: o.num, config: o.cfg, retired: o.retired }; paintTap(); drawChart(); return; }
    if (t.dataset.c) { const c = D.byId[t.dataset.c]; if (c) TAPPED = { name: c.origin === 'recorded' ? 'Recorded' : '', label: c.label, num: c.num, config: c.config, origin: c.origin, retired: c.retired }; paintTap(); drawChart(); return; }
    if (TAPPED) { TAPPED = null; paintTap(); drawChart(); }
  };
  document.addEventListener('keydown', e => {
    if (e.target.closest && e.target.closest('input, textarea, select')) return;
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
    else if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); }
    else if ((e.key === 'Delete' || e.key === 'Backspace') && SEL && SEL.kind === 'node') { e.preventDefault(); removeNode(SEL.id); }
    else if (e.key === 'Escape') { SEL = null; PREVIEW = PINNED = null; renderAll(); paintNudges(); drawPreview(); }
  });
  let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { drawChart(); paintCanvas(); paintPop(); paintSide(); }, 80); });
}
async function boot() {
  let c;
  try { c = await call('/api/context'); }
  catch (e) {
    setMode('down', 'loopmath not answering');
    $('taskT').textContent = 'The builder could not load its recommendation';
    $('body').innerHTML = `<div class="fatal"><h2>No numbers to show</h2><p>The page asked loopmath builder for its recommendation and got no answer: ${esc(e.message || e)}.</p><p>Start it with <code>loopmath builder</code> (the recommend task flags, or <code>--rec REC</code>) and open the address it prints. This page shows only live numbers.</p></div>`;
    return;
  }
  prepare(c);
  setMode('live', 'live: fitted model');
  $('body').innerHTML = ''; $('body').appendChild($('page').content.cloneNode(true));
  paintHeader(); paintLegend(); paintTap(); wire(); canvasEvents();
  if (!D.start) { $('side').innerHTML = '<div class="errs">The recommendation has no configuration to start from.</div>'; return; }
  START = D.startOpt;
  S = fromConfig(D.start);
  lastAction = START ? 'Started from option ' + START.no + ' (' + START.name.toLowerCase() + ')' : 'Started';
  renderAll(); drawChart();
  schedulePredict(0);
}
boot();
})();
