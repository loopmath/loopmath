"""Shared inline JavaScript for the graph visualizer."""

import json

from ..ingest.base import canonical_model
from ..price import load_prices


def model_order(path=None) -> list[str]:
    """Canonical model names in price-table order, then `unknown`.

    The model menu and legend list models in this order; a model the table
    lacks follows them by name. Adding a price row is enough to place a new
    model.
    """
    order: list[str] = []
    for key in load_prices(path).rates:
        name = canonical_model(key) or key
        if name not in order:
            order.append(name)
    return [*order, "unknown"]


COMMON_JS = r"""
'use strict';
const App = (() => {
  const D = DATA, N = D.nodes, E = D.edges, A = D.artifacts, RUN = D.run;
  const TOKEN_STREAMS = RUN.token_streams, TOKEN_TOTAL_STREAMS = RUN.token_total_streams;
  const MODEL_ORDER = __MODEL_ORDER__;
  const rolesFound = [...new Set(N.map(n => n.role))];
  const modelsFound = [...new Set(N.map(n => n.model || 'unknown'))];
  const ROLE_ORDER = [...RUN.role_vocabulary.filter(role => role !== 'external'), ...rolesFound.filter(role => !RUN.role_vocabulary.includes(role) && role !== 'unlabeled').sort(), ...RUN.role_vocabulary.filter(role => role === 'external'), 'unlabeled'];
  const ROLES = [...ROLE_ORDER.filter(r => rolesFound.includes(r)), ...rolesFound.filter(r => !ROLE_ORDER.includes(r)).sort()];
  const MODELS = [...MODEL_ORDER.filter(m => modelsFound.includes(m)), ...modelsFound.filter(m => !MODEL_ORDER.includes(m)).sort()];
  const ROLE_COLOR = { lead: '#2a78d6', dev: '#eb6834', reviewer: '#1baf7a', planner: '#4a3aa7', cli: '#898781', external: '#898781', unlabeled: '#898781' };
  const MODEL_COLOR = { 'opus-5': '#2a78d6', 'gpt-5.6-sol': '#eb6834', 'fable-5': '#1baf7a', unknown: '#898781' };
  const colorFallback = '#898781';

  N.forEach(n => {
    n.written = []; n.read = []; n.out = []; n.in = []; n.children = [];
    n.untimed = n.t0 == null;
    n.durationUnknown = n.dur == null;
    n.t0Plot = n.t0 == null ? 0 : n.t0;
    n.durPlot = n.dur == null ? 0 : n.dur;
    n.t1 = n.untimed ? n.t0Plot : n.t0Plot + n.durPlot;
    const streams = TOKEN_STREAMS.map(key => n.tok[key]);
    n.tokComplete = n.tok_record && streams.every(Number.isFinite);
    n.tokTotal = n.tokComplete ? TOKEN_TOTAL_STREAMS.reduce((sum, key) => sum + n.tok[key], 0) : null;
    n.modelKey = n.model || 'unknown';
  });
  A.forEach(a => { a.w.forEach(i => N[i].written.push(a.i)); a.c.forEach(i => N[i].read.push(a.i)); });
  E.forEach((e, k) => { N[e[0]].out.push(k); N[e[1]].in.push(k); });
  N.forEach(n => { if (Number.isInteger(n.parent) && N[n.parent]) N[n.parent].children.push(n.i); });
  const MAX_USD = Math.max(1, ...N.map(n => n.usd == null ? 0 : n.usd));
  const HANDOFFS = E.filter(e => e[2] === 'artifact').length;

  const state = { roles: new Set(ROLES), model: 'all', sel: null, hov: null, sortKey: 'start', sortDir: 1, cardView: 'swim', views: {} };
  const layouts = new Map();
  let visSet = new Set();
  const $ = id => document.getElementById(id);
  const section = key => document.querySelector(`[data-view="${key}"]`);
  const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const fmt = {
    usd: v => v == null ? 'n/a' : (v < 0.1 ? '$' + v.toFixed(3) : '$' + v.toFixed(2)),
    dur: s => {
      if (s == null) return 'n/a'; s = Math.round(s);
      if (s < 60) return s + 's';
      const m = Math.floor(s / 60), h = Math.floor(m / 60);
      if (h) return h + 'h ' + String(m % 60).padStart(2, '0') + 'm';
      return m + 'm ' + String(s % 60).padStart(2, '0') + 's';
    },
    tok: v => v == null ? 'n/a' : v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(0) + 'k' : String(v),
    int: v => v == null ? 'n/a' : Number(v).toLocaleString('en-US'),
    clock: sec => RUN.start ? new Date(Date.parse(RUN.clock_base) + sec * 1000).toISOString().slice(11, 16) + 'Z' : '+' + fmt.hm(sec),
    hm: sec => { const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60); return h + 'h' + String(m).padStart(2, '0'); },
    ts: iso => iso ? iso.replace('T', ' ').replace(/\.\d+/, '').replace('Z', ' UTC') : 'n/a',
    tier: value => value || 'unknown',
    tokenLabel: key => key.replaceAll('_', ' '),
  };
  const tokenDetail = n => `${n.tokComplete ? 'complete total' : n.tok_record ? 'total unavailable; incomplete' : 'no token record; total unavailable'}; ` + TOKEN_STREAMS.map(key => `${fmt.tokenLabel(key)} ${fmt.int(n.tok[key])}`).join(' / ');
  const attemptValue = (field, value, html) => `<span data-attempt-field="${field}" data-attempt-value="${esc(JSON.stringify(value))}">${html}</span>`;
  const sourceAttempt = n => ({
    id: n.id, harness: n.harness, source: n.src, session_path: n.session,
    model: n.model, model_tier: n.mt, effort: n.effort, workspace: n.ws,
    ts: n.ts, wall_s: n.dur, tokens: n.tok_record ? n.tok : null, usd: n.usd,
    parent: n.parent_source, spawn: n.spawn, launched_by: n.launch,
    role: n.role_source, role_tier: n.rt, role_evidence: n.re,
    phase: n.phase, phase_tier: n.pt,
  });
  const sourceFieldRows = n => {
    const values = sourceAttempt(n);
    return RUN.attempt_source_fields.map(field => {
      const encoded = JSON.stringify(values[field]);
      return `<span class="source-line" data-source-field="${esc(field)}" data-attempt-value="${esc(encoded)}"><b>${esc(field)}</b> <span data-source-visible>${esc(encoded == null ? 'unavailable' : encoded)}</span></span>`;
    }).join('<br>');
  };
  const recordDetail = (field, record) => {
    const label = field.replaceAll('_', ' ');
    if (record == null) return `<span class="evid" data-nested-empty="${field}">${label}: none recorded</span>`;
    const entries = Object.entries(record);
    if (!entries.length) return `<span class="evid" data-nested-map="${field}">${label}: empty mapping</span>`;
    const lines = entries.map(([key, value]) => {
      const shown = value == null ? 'n/a' : typeof value === 'object' ? JSON.stringify(value) : String(value);
      return `<span class="nested-line"><b>${esc(key.replaceAll('_', ' '))}</b> <span data-nested-field="${field}" data-nested-key="${esc(key)}" data-attempt-value="${esc(JSON.stringify(value))}">${esc(shown)}</span></span>`;
    });
    return `<span class="evid">${label}:</span><br>${lines.join('<br>')}`;
  };

  const color = (n, mode) => mode === 'model' ? (MODEL_COLOR[n.modelKey] || colorFallback) : (ROLE_COLOR[n.role] || colorFallback);
  const dashed = n => n.mt !== 'verified';
  const radius = (n, max = 26, min = 4) => n.usd == null ? min : Math.max(min, Math.sqrt(n.usd / MAX_USD) * max);
  const visible = () => N.filter(n => state.roles.has(n.role) && (state.model === 'all' || n.modelKey === state.model)).map(n => n.i);
  const edgeVisible = e => visSet.has(e[0]) && visSet.has(e[1]);
  const artVisible = a => a.c.length > 0 && (a.w.some(i => visSet.has(i)) || a.c.some(i => visSet.has(i)));

  const NS = 'ht' + 'tp:' + '/' + '/www.w3.org/2000/svg';
  const svg = (tag, attrs, parent) => {
    const el = document.createElementNS(NS, tag);
    for (const key in attrs) if (attrs[key] != null) el.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(el);
    return el;
  };
  const text = (parent, x, y, value, cls, attrs) => { const el = svg('text', Object.assign({ x, y, class: cls }, attrs || {}), parent); el.textContent = value; return el; };
  const curve = (x1, y1, x2, y2, bend = 0.5) => { const dx = (x2 - x1) * bend; return `M${x1},${y1}C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`; };
  const vcurve = (x1, y1, x2, y2, bend = 0.5) => { const dy = (y2 - y1) * bend; return `M${x1},${y1}C${x1},${y1 + dy} ${x2},${y2 - dy} ${x2},${y2}`; };
  const arc = (x1, y1, x2, y2, lift = 0.25) => { const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1; const nx = -dy / len, ny = dx / len; return `M${x1},${y1}Q${mx + nx * len * lift},${my + ny * len * lift} ${x2},${y2}`; };
  const timeTicks = (span, target = 8) => { const steps = [300, 600, 900, 1800, 3600, 7200]; const step = steps.find(value => span / value <= target) || 7200; const out = []; for (let at = 0; at <= span; at += step) out.push(at); return out; };
  const edgeClass = e => `edge e-${e[2]} t-${e[3]}`;
  const nodeClass = n => 'node' + (dashed(n) ? ' dashed' : '') + (n.untimed || n.durationUnknown || n.usd == null ? ' hollow' : '');

  function renderStats(vis) {
    const usd = vis.reduce((sum, i) => sum + (N[i].usd == null ? 0 : N[i].usd), 0);
    const unpriced = vis.filter(i => N[i].usd == null).length;
    const consumed = A.filter(artVisible).length;
    $('stats').innerHTML = [
      [`${vis.length}<span class="m">/${N.length}</span>`, 'attempts'],
      [fmt.usd(usd), (vis.length === N.length ? 'known dollars, all attempts' : 'known dollars in view') + (unpriced ? `; ${unpriced} unpriced` : '')],
      [fmt.dur(RUN.span_s), 'run span'],
      [`${consumed}<span class="m">/${RUN.n_artifacts}</span>`, 'artifacts consumed / written'],
      [RUN.n_source_edges === E.length ? E.length : `${E.length}<span class="m">/${RUN.n_source_edges}</span>`, RUN.n_source_edges === E.length ? 'edges' : 'edges shown / supplied'],
    ].map(([value, label]) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`).join('');
  }

  function renderAccounting() {
    const entries = Object.entries(RUN.accounting || {});
    $('accounting').innerHTML = entries.length
      ? '<b>Viewer accounting:</b> ' + entries.map(([key, value]) => `${esc(key.replaceAll('_', ' '))}: ${fmt.int(value)}`).join('; ')
      : '<b>Viewer accounting:</b> no unavailable, excluded, or shortened values.';
  }

  function renderFilters() {
    const host = $('filters');
    let html = `<div class="grp"><span class="gl">role</span>` + ROLES.map(role =>
      `<span class="chip${state.roles.has(role) ? '' : ' off'}" data-role="${esc(role)}"><i class="dot" style="background:${ROLE_COLOR[role] || colorFallback}"></i>${esc(role)} <span class="m" style="color:var(--muted)">${N.filter(n => n.role === role).length}</span></span>`).join('') + `</div>`;
    html += `<div class="grp"><span class="gl">model</span><select data-model-filter><option value="all">all models</option>` + MODELS.map(model => `<option value="${esc(model)}"${state.model === model ? ' selected' : ''}>${esc(model)}</option>`).join('') + `</select></div>`;
    html += `<div class="grp"><button data-reset>reset</button></div>`;
    host.innerHTML = html;
  }

  function filterEvent(ev) {
    const role = ev.target.closest('[data-role]');
    if (role && ev.type === 'click') {
      const value = role.dataset.role;
      if (state.roles.has(value)) state.roles.delete(value); else state.roles.add(value);
      if (state.roles.size === 0) state.roles = new Set(ROLES);
      renderFilters(); renderAll(); return;
    }
    if (ev.target.matches('[data-model-filter]') && ev.type === 'change') {
      state.model = ev.target.value; renderAll(); return;
    }
    if (ev.target.closest('[data-reset]') && ev.type === 'click') {
      state.roles = new Set(ROLES); state.model = 'all'; state.sel = null; state.hov = null;
      layouts.forEach((layout, key) => { state.views[key] = viewDefaults(layout); });
      renderFilters(); layouts.forEach((layout, key) => renderViewControls(key)); renderAll();
    }
  }

  const COLS = [
    { key: 'start', label: 'attempt', cell: n => `<span class="lbl"><i class="dot" style="background:${color(n, state.views.swim.colorBy)}"></i>${esc(n.lbl)}</span><span class="sub" title="${esc(n.aid)}">${esc(n.title)}</span>`, sort: n => n.t0Plot },
    { key: 'role', label: 'role', cell: n => `${esc(n.role)}<span class="tier">${esc(fmt.tier(n.rtd))}</span>`, sort: n => n.role },
    { key: 'model', label: 'model', cell: n => esc(n.model || 'n/a'), sort: n => n.modelKey },
    { key: 'tier', label: 'tier', cell: n => esc(n.mt || 'n/a'), sort: n => n.mt || '' },
    { key: 'usd', label: 'cost', num: true, cell: n => fmt.usd(n.usd), sort: n => n.usd == null ? -1 : n.usd },
    { key: 'tok', label: 'tokens', num: true, cell: n => `<span title="${esc(tokenDetail(n))}">${fmt.tok(n.tokTotal)}</span>`, sort: n => n.tokTotal == null ? -1 : n.tokTotal },
    { key: 'dur', label: 'duration', num: true, cell: n => fmt.dur(n.dur), sort: n => n.dur == null ? -1 : n.dur },
    { key: 'status', label: 'status', cell: n => n.status == null ? 'n/a' : esc(n.status), sort: n => n.status || '' },
    { key: 'nw', label: 'artifacts written', num: true, cell: n => n.written.length, sort: n => n.written.length },
    { key: 'nr', label: 'artifacts read', num: true, cell: n => n.read.length, sort: n => n.read.length },
  ];
  function renderTable(vis) {
    const col = COLS.find(item => item.key === state.sortKey) || COLS[0];
    const rows = vis.slice().sort((a, b) => { const x = col.sort(N[a]), y = col.sort(N[b]); return (x < y ? -1 : x > y ? 1 : 0) * state.sortDir || N[a].t0Plot - N[b].t0Plot; });
    let html = '<thead><tr>' + COLS.map(item => `<th data-key="${item.key}" data-column="${item.key}"${item.num ? ' class="num"' : ''}>${item.label}${item.key === state.sortKey ? `<span class="arrow">${state.sortDir > 0 ? '▲' : '▼'}</span>` : ''}</th>`).join('') + '</tr></thead><tbody>';
    rows.forEach(i => { const n = N[i]; html += `<tr data-i="${i}">` + COLS.map(item => `<td${item.num ? ' class="num"' : ''}>${item.cell(n)}</td>`).join('') + '</tr>'; });
    $('table').innerHTML = html + '</tbody>';
    $('tablecount').textContent = `${rows.length} attempts` + (rows.length < N.length ? ` of ${N.length} (filtered)` : '') + `, sorted by ${col.label}`;
  }
  function tableEvent(ev) {
    const header = ev.target.closest('th[data-key]');
    if (header && ev.type === 'click') {
      const key = header.dataset.key;
      if (state.sortKey === key) state.sortDir = -state.sortDir;
      else { state.sortKey = key; state.sortDir = ['usd', 'tok', 'dur', 'nw', 'nr'].includes(key) ? -1 : 1; }
      renderTable(visible()); applyHighlight(); return;
    }
    const row = ev.target.closest('tr[data-i]');
    if (!row) return;
    if (ev.type === 'click') select({ t: 'n', i: +row.dataset.i }, 'swim', true);
    else if (ev.type === 'mouseover') hover({ t: 'n', i: +row.dataset.i });
  }

  const tip = $('tip');
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.hidden = false;
    const width = tip.offsetWidth, height = tip.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    let left = x + 14, top = y + 14;
    if (left + width > vw - 8) left = x - width - 10;
    if (top + height > vh - 8) top = y - height - 10;
    tip.style.left = Math.max(4, left) + 'px'; tip.style.top = Math.max(4, top) + 'px';
  }
  const hideTip = () => { tip.hidden = true; };
  const nodeTip = n => `<b>${esc(n.lbl)}</b> <span class="m">${esc(n.title)}</span><br>${esc(n.role)} <span class="m">(${esc(fmt.tier(n.rtd))})</span> on ${esc(n.model || 'n/a')} <span class="m">(${esc(fmt.tier(n.mt))})</span><br>${fmt.usd(n.usd)} <span class="m">/</span> ${n.untimed ? 'start unavailable' : 'start ' + fmt.clock(n.t0)} <span class="m">/</span> duration ${fmt.dur(n.dur)} <span class="m">/</span> ${n.tokComplete ? fmt.tok(n.tokTotal) + ' tokens (complete)' : 'token total unavailable'}<br><span class="m">wrote ${n.written.length}, read ${n.read.length} artifacts</span>`;
  const missingRefs = items => items.map(item => `${esc(item.id || 'unavailable id')} (${esc(item.reason.replaceAll('_', ' '))})`).join(', ');
  const artTip = a => `<b>${esc(a.path)}</b><br>${esc(a.kind || 'unknown kind')} <span class="m">(${esc(fmt.tier(a.kt))})</span>${a.lang ? ' · ' + esc(a.lang) : ''}<br>written by ${a.w.map(i => esc(N[i].lbl)).join(', ') || 'nobody in scope'}${a.wm.length ? '; unresolved: ' + missingRefs(a.wm) : ''}<br>read by ${a.c.map(i => esc(N[i].lbl)).join(', ') || 'nobody'}${a.cm.length ? '; unresolved: ' + missingRefs(a.cm) : ''}<br><span class="m">${fmt.int(a.nw)} writes, ${fmt.int(a.nr)} reads${a.t != null ? ', first write ' + fmt.clock(a.t) : ', first write unavailable'}</span>`;
  const edgeTip = e => { const source = N[e[0]], target = N[e[1]]; if (e[2] === 'artifact') { const artifact = A[e[4]]; return `<b>handoff</b> ${esc(source.lbl)} → ${esc(target.lbl)} <span class="m">(${esc(e[3])})</span><br>${esc(artifact ? artifact.path : 'artifact unavailable')}<br><span class="m">${e[5] == null ? 'read lag unavailable' : 'read ' + fmt.dur(e[5]) + ' after the write'}</span>`; } return `<b>${esc(e[2])}</b> ${esc(source.lbl)} → ${esc(target.lbl)} <span class="m">(${esc(e[3])})</span>` + (e[2] === 'launch' ? `<br><span class="m">${e[5] == null ? 'launch lag unavailable' : 'session began ' + e[5] + ' s after the launch command'}</span>` : '') + (e[2] === 'spawn' && target.spawn ? `<br><span class="m">${esc(target.spawn.description || 'description unavailable')}</span>` : ''); };
  const edgeListTip = (keys, el) => {
    const node = N[+el.dataset.connectionNode], artifact = A[+el.dataset.alink], kind = el.dataset.connectionKind;
    const connection = `<b>artifact ${esc(kind)}</b> ${esc(node.lbl)} ${kind === 'write' ? '→' : '←'} ${esc(artifact.path)}`;
    return connection + (keys.length ? '<hr>' + keys.map(key => edgeTip(E[key])).join('<hr>') : '<br><span class="m">no matching handoff edge; counted in viewer accounting</span>');
  };

  function related(focus) {
    const nodes = new Set(), edges = new Set(), artifacts = new Set();
    if (!focus) return { nodes, edges, artifacts };
    if (focus.t === 'n') {
      const n = N[focus.i];
      n.out.forEach(k => { if (edgeVisible(E[k])) { edges.add(k); nodes.add(E[k][1]); } });
      n.in.forEach(k => { if (edgeVisible(E[k])) { edges.add(k); nodes.add(E[k][0]); } });
      n.written.forEach(j => artifacts.add(j)); n.read.forEach(j => artifacts.add(j));
    } else {
      const artifact = A[focus.i];
      artifact.w.forEach(i => nodes.add(i)); artifact.c.forEach(i => nodes.add(i));
      E.forEach((edge, k) => { if (edge[2] === 'artifact' && edge[4] === focus.i) edges.add(k); });
    }
    return { nodes, edges, artifacts };
  }
  function applyHighlight() {
    const focus = state.hov || state.sel, rel = related(focus);
    document.querySelectorAll('.graph [data-node]').forEach(el => { const i = +el.dataset.node, isFocus = focus && focus.t === 'n' && focus.i === i; el.classList.toggle('focus', !!isFocus); el.classList.toggle('rel', !isFocus && rel.nodes.has(i)); el.classList.toggle('dim', !!focus && !isFocus && !rel.nodes.has(i)); });
    document.querySelectorAll('.graph [data-edge]').forEach(el => { const k = +el.dataset.edge; el.classList.toggle('on', rel.edges.has(k)); el.classList.toggle('dim', !!focus && !rel.edges.has(k)); });
    document.querySelectorAll('.graph [data-art]').forEach(el => { const j = +el.dataset.art, isFocus = focus && focus.t === 'a' && focus.i === j; el.classList.toggle('focus', !!isFocus); el.classList.toggle('rel', !isFocus && rel.artifacts.has(j)); el.classList.toggle('dim', !!focus && !isFocus && !rel.artifacts.has(j)); });
    document.querySelectorAll('.graph .alink').forEach(el => { const j = +el.dataset.alink; const on = (focus && focus.t === 'a' && focus.i === j) || rel.artifacts.has(j); el.classList.toggle('on', on); el.classList.toggle('dim', !!focus && !on); });
    $('table').querySelectorAll('tr[data-i]').forEach(row => { const i = +row.dataset.i, isFocus = focus && focus.t === 'n' && focus.i === i; row.classList.toggle('focus', !!isFocus); row.classList.toggle('rel', !isFocus && rel.nodes.has(i)); row.classList.toggle('dim', !!focus && !isFocus && !rel.nodes.has(i)); });
  }
  function hover(focus) { state.hov = focus; applyHighlight(); }

  const artList = ids => ids.length ? '<ul>' + ids.map(j => `<li><span class="link" data-art-link="${j}">${esc(A[j].path)}</span> <span class="evid">${esc(A[j].kind || '')}</span></li>`).join('') + '</ul>' : '<span class="none">none</span>';
  const participantList = (ids, missing, empty) => {
    const linked = ids.map(i => `<span class="link" data-node-link="${i}">${esc(N[i].lbl)}</span>`);
    const unresolved = missing.map(item => `<span class="unresolved">${esc(item.id || 'unavailable id')} (${esc(item.reason.replaceAll('_', ' '))})</span>`);
    return [...linked, ...unresolved].join(', ') || empty;
  };
  function cardHtml(focus) {
    if (focus.t === 'a') {
      const a = A[focus.i];
      return `<button class="x" data-close>×</button><h3><i class="dot" style="background:#f0efec;border:1.5px solid #898781"></i>artifact</h3><div class="ttl mono">${esc(a.path)}</div><dl>
<dt>kind</dt><dd>${esc(a.kind || 'unknown')} <span class="evid">${esc(fmt.tier(a.kt))}</span></dd>
<dt>producer</dt><dd>${Number.isInteger(a.prod) ? `<span class="link" data-node-link="${a.prod}">${esc(N[a.prod].lbl)}</span>` : a.pm ? `${esc(a.pm.id || 'unavailable id')} <span class="evid">(${esc(a.pm.reason.replaceAll('_', ' '))})</span>` : 'none recorded'}</dd>
<dt>writers</dt><dd>${participantList(a.w, a.wm, 'none in scope')}</dd>
<dt>consumers</dt><dd>${participantList(a.c, a.cm, 'none')}</dd>
<dt>first write</dt><dd>${a.t != null ? fmt.clock(a.t) + ' (' + fmt.hm(a.t) + ' into the run)' : 'unavailable'}</dd>
<dt>writes / reads</dt><dd>${fmt.int(a.nw)} / ${fmt.int(a.nr)}</dd>
<dt>language</dt><dd>${esc(a.lang || 'unknown')}</dd>
<dt>size</dt><dd>${a.bytes != null ? fmt.int(a.bytes) + ' bytes' : 'not recorded'}${a.la != null || a.lr != null ? `, +${fmt.int(a.la)}/-${fmt.int(a.lr)} lines` : ''}</dd>
${a.hint ? `<dt>hint</dt><dd class="mono">${esc(a.hint)}</dd>` : ''}</dl>`;
    }
    const n = N[focus.i], parent = Number.isInteger(n.parent) ? N[n.parent] : null;
    const parentText = parent ? `<span class="link" data-node-link="${parent.i}">${esc(parent.lbl)}</span> ` : n.parent_ref ? `unresolved parent ${esc(n.parent_ref)} <span class="evid">(${esc(n.parent_reason.replaceAll('_', ' '))})</span> ` : 'none in scope ';
    return `<button class="x" data-close>×</button><h3><i class="dot" style="background:${color(n, state.views[state.cardView].colorBy)}"></i>${esc(n.lbl)} <span class="evid">${esc(n.role)}</span></h3><div class="ttl">${esc(n.title)}</div><dl>
<dt>attempt id</dt><dd class="mono">${attemptValue('id', n.id, esc(n.aid))}</dd>
<dt>role</dt><dd>${attemptValue('role', n.role_source, esc(n.role))} <span class="evid">${attemptValue('role_tier', n.rt, esc(fmt.tier(n.rt)))}: ${attemptValue('role_evidence', n.re, esc(n.re || 'evidence unavailable'))}</span></dd>
<dt>model</dt><dd>${attemptValue('model', n.model, esc(n.model || 'n/a'))} <span class="evid">${attemptValue('model_tier', n.mt, esc(fmt.tier(n.mt)))}, effort ${attemptValue('effort', n.effort, esc(n.effort || 'n/a'))}</span></dd>
<dt>harness</dt><dd>${attemptValue('harness', n.harness, esc(n.harness || 'n/a'))} <span class="evid">source ${attemptValue('source', n.src, esc(n.src || 'n/a'))}, workspace ${attemptValue('workspace', n.ws, esc(n.ws || 'n/a'))}</span></dd>
<dt>phase</dt><dd>${attemptValue('phase', n.phase, esc(n.phase || 'n/a'))} <span class="evid">${attemptValue('phase_tier', n.pt, esc(fmt.tier(n.pt)))}</span></dd>
<dt>started</dt><dd>${attemptValue('ts', n.ts, n.untimed ? 'no timestamp in the session record' : fmt.ts(n.ts) + ` <span class="evid">(${fmt.hm(n.t0)} into the run)</span>`)}</dd>
<dt>duration</dt><dd>${attemptValue('wall_s', n.dur, n.dur == null ? 'n/a' : fmt.dur(n.dur) + (n.untimed ? ' <span class="evid">(end unavailable without a start)</span>' : ` <span class="evid">ended ${fmt.clock(n.t1)}</span>`))}</dd>
<dt>cost</dt><dd>${attemptValue('usd', n.usd, fmt.usd(n.usd) + (n.usd == null ? ' <span class="evid">(cost unavailable)</span>' : ''))}</dd>
<dt>tokens</dt><dd>${attemptValue('tokens', n.tok_record ? n.tok : null, `${n.tokComplete ? fmt.int(n.tokTotal) + ' <span class="evid">(complete total)</span>' : n.tok_record ? 'total unavailable <span class="evid">(incomplete)</span>' : 'no token record; total unavailable'}<br><span class="evid">${TOKEN_STREAMS.map(key => `<span data-token-stream="${esc(key)}" data-token-value="${esc(JSON.stringify(n.tok[key]))}">${esc(fmt.tokenLabel(key))} ${fmt.int(n.tok[key])}</span>`).join(', ')}</span>`)}</dd>
<dt>status</dt><dd>${n.status == null ? 'n/a <span class="evid">(status unavailable; the graph model carries no acceptance signal)</span>' : esc(n.status)}</dd>
<dt>origin</dt><dd>${attemptValue('parent', n.parent_source, `<span class="evid">parent:</span> ${parentText}`)}<br>${attemptValue('spawn', n.spawn, recordDetail('spawn', n.spawn))}<br>${attemptValue('launched_by', n.launch, recordDetail('launched_by', n.launch))}</dd>
<dt>children</dt><dd>${n.children.length ? n.children.length + ' spawned or launched' : 'none'}</dd>
<dt>written (${n.written.length})</dt><dd data-written-links>${artList(n.written)}</dd>
<dt>read (${n.read.length})</dt><dd data-read-links>${artList(n.read)}</dd>
<dt>source fields (${RUN.attempt_source_fields.length})</dt><dd class="mono" data-source-fields>${sourceFieldRows(n)}</dd>
<dt>session</dt><dd class="mono">${attemptValue('session_path', n.session, esc(n.session || 'n/a'))}</dd></dl>`;
  }
  function positionCard() {
    const card = $('card'), focus = state.sel;
    if (!focus || card.hidden) return;
    const sec = section(state.cardView), wrap = sec.querySelector('[data-wrap]'), graph = sec.querySelector('[data-graph]');
    if (card.parentElement !== wrap) wrap.appendChild(card);
    const el = graph.querySelector(focus.t === 'n' ? `[data-node="${focus.i}"]` : `[data-art="${focus.i}"]`);
    const wr = wrap.getBoundingClientRect(); let left = 12, top = 12;
    if (el) {
      const rect = el.getBoundingClientRect(); left = rect.right - wr.left + wrap.scrollLeft + 10; top = rect.top - wr.top + wrap.scrollTop - 10;
      if (left + card.offsetWidth > wrap.scrollLeft + wrap.clientWidth - 8) left = Math.max(wrap.scrollLeft + 8, rect.left - wr.left + wrap.scrollLeft - card.offsetWidth - 10);
      if (top + card.offsetHeight > wrap.scrollTop + wrap.clientHeight - 8) top = Math.max(wrap.scrollTop + 8, wrap.scrollTop + wrap.clientHeight - card.offsetHeight - 8);
      if (top < wrap.scrollTop + 8) top = wrap.scrollTop + 8;
    }
    card.style.left = left + 'px'; card.style.top = top + 'px';
  }
  function renderCard() {
    const card = $('card');
    if (!state.sel) { card.hidden = true; return; }
    card.innerHTML = cardHtml(state.sel); card.hidden = false; positionCard();
  }
  function select(focus, viewKey = state.cardView, scrollGraph = false) {
    if (focus && state.sel && focus.t === state.sel.t && focus.i === state.sel.i && viewKey === state.cardView) focus = null;
    state.sel = focus; state.hov = null; state.cardView = viewKey;
    if (focus) section(viewKey).open = true;
    applyHighlight(); renderCard();
    if (focus && focus.t === 'n') {
      const row = $('table').querySelector(`tr[data-i="${focus.i}"]`); if (row) row.scrollIntoView({ block: 'nearest' });
      if (scrollGraph) { const el = section(viewKey).querySelector(`[data-node="${focus.i}"]`); if (el) el.scrollIntoView({ block: 'nearest', inline: 'nearest' }); }
    }
  }

  function graphEvent(ev, key) {
    const el = ev.target.closest('[data-node],[data-art],[data-edge],[data-edge-list]');
    if (ev.type === 'mousemove') {
      if (!el) { hideTip(); hover(null); return; }
      if (el.dataset.node != null) hover({ t: 'n', i: +el.dataset.node });
      else if (el.dataset.art != null) hover({ t: 'a', i: +el.dataset.art }); else hover(null);
      const html = el.dataset.node != null ? nodeTip(N[+el.dataset.node]) : el.dataset.art != null ? artTip(A[+el.dataset.art]) : el.dataset.edgeList != null ? edgeListTip(el.dataset.edgeList.split(',').filter(Boolean).map(Number), el) : edgeTip(E[+el.dataset.edge]);
      showTip(html, ev.clientX, ev.clientY); return;
    }
    if (ev.type === 'mouseleave') { hideTip(); hover(null); return; }
    if (ev.type === 'click') {
      hideTip();
      if (!el) { if (state.sel) select(null); return; }
      if (el.dataset.node != null) select({ t: 'n', i: +el.dataset.node }, key);
      else if (el.dataset.art != null) select({ t: 'a', i: +el.dataset.art }, key);
    }
  }

  const line = (dash, col) => `<svg width="26" height="8"><line x1="1" y1="4" x2="25" y2="4" stroke="${col}" stroke-width="1.6"${dash ? ` stroke-dasharray="${dash}"` : ''}/></svg>`;
  // One line style per edge kind. RUN.edge_kinds is the graph model's EDGE_KINDS,
  // so the legend never holds its own vocabulary; each dash pattern here matches
  // the .e-<kind> rule in the stylesheet.
  const EDGE_LEGEND = {
    dep: { dash: '1 3', col: '#7a6a9c', label: 'dep, dotted: the target cannot start before the source settles' },
    fan_in: { dash: '7 2 1 2', col: '#7a6a9c', label: 'fan_in, dash-dot: the source is an acceptance input of the target gate' },
    spawn: { dash: '', col: '#52514e', label: 'spawn' },
    launch: { dash: '4 3', col: '#52514e', label: 'launch' },
    artifact: { dash: '', col: '#b3b1a9', label: 'handoff (darker when verified)' },
  };
  function renderLegend(key, vis) {
    const view = state.views[key], layout = layouts.get(key), items = [];
    if (view.colorBy === 'role') ROLES.filter(role => vis.some(i => N[i].role === role)).forEach(role => items.push(`<span class="it"><i class="sw" style="background:${ROLE_COLOR[role] || colorFallback}"></i>${esc(role)}</span>`));
    else MODELS.filter(model => vis.some(i => N[i].modelKey === model)).forEach(model => items.push(`<span class="it"><i class="sw" style="background:${MODEL_COLOR[model] || colorFallback}"></i>${esc(model)}</span>`));
    RUN.edge_kinds.forEach(kind => { const edge = EDGE_LEGEND[kind] || { dash: '', col: colorFallback, label: kind }; items.push(`<span class="it">${line(edge.dash, edge.col)} ${esc(edge.label)}</span>`); });
    if (layout.artifactMode === 'always' || view.artifacts) items.push(`<span class="it"><i class="sw sq" style="background:#f0efec;border:1px solid #898781"></i>artifact</span>`);
    if (layout.sizeNote) items.push(`<span class="it" style="color:var(--muted)">${esc(layout.sizeNote)}</span>`);
    section(key).querySelector('[data-legend]').innerHTML = items.join('');
  }

  function viewDefaults(layout) {
    const extra = {};
    (layout.controls || []).forEach(control => { extra[control.id] = control.value; });
    return { colorBy: layout.colorBy || 'role', artifacts: false, extra };
  }
  function renderViewControls(key) {
    const layout = layouts.get(key), view = state.views[key], host = section(key).querySelector('[data-controls]');
    let html = `<div class="grp"><span class="gl">color</span><select data-view-color><option value="role"${view.colorBy === 'role' ? ' selected' : ''}>by role</option><option value="model"${view.colorBy === 'model' ? ' selected' : ''}>by model</option></select></div>`;
    if (layout.artifactMode !== 'always') html += `<div class="grp"><label class="cb"><input type="checkbox" data-view-artifacts${view.artifacts ? ' checked' : ''}> artifacts <span class="gl">(consumed ones, off by default)</span></label></div>`;
    (layout.controls || []).forEach(control => {
      const options = control.options.map(option => `<option value="${esc(option.v)}"${view.extra[control.id] === option.v ? ' selected' : ''}>${esc(option.l.replace('%HANDOFFS%', HANDOFFS))}</option>`).join('');
      html += `<div class="grp"><span class="gl">${esc(control.label)}</span><select data-extra="${esc(control.id)}">${options}</select></div>`;
    });
    host.innerHTML = html;
  }
  function viewControlEvent(ev, key) {
    const view = state.views[key];
    if (ev.target.matches('[data-view-color]')) view.colorBy = ev.target.value;
    else if (ev.target.matches('[data-view-artifacts]')) view.artifacts = ev.target.checked;
    else if (ev.target.matches('[data-extra]')) view.extra[ev.target.dataset.extra] = ev.target.value;
    else return;
    renderView(key);
    if (key === 'swim') renderTable(visible());
    applyHighlight(); renderCard();
  }

  function renderView(key) {
    const layout = layouts.get(key), sec = section(key), host = sec.querySelector('[data-graph]'), vis = visible();
    host.innerHTML = '';
    layout.render(host, vis, state.views[key], api);
    renderLegend(key, vis);
  }
  function renderAll() {
    const vis = visible(); visSet = new Set(vis);
    renderStats(vis); renderTable(vis); layouts.forEach((layout, key) => renderView(key));
    applyHighlight(); renderCard();
  }
  function renderFoot() {
    const kinds = RUN.edges_by_kind_tier || {};
    const unpriced = N.filter(n => n.usd == null).length;
    $('foot').innerHTML = `Data: ${esc(RUN.source)}. Workspace ${esc(RUN.workspace)}, ${fmt.ts(RUN.start)} to ${fmt.ts(RUN.end)}. ${N.length} attempts, ${E.length} of ${RUN.n_source_edges} supplied edges shown (${Object.entries(kinds).map(([key, value]) => value + ' ' + key.replace('/', ' ')).join(', ')}), ${RUN.n_artifacts} artifacts of which ${RUN.n_consumed} were read by another attempt. Known cost ${fmt.usd(N.reduce((sum, n) => sum + (n.usd == null ? 0 : n.usd), 0))} with ${unpriced} unpriced attempt. Attempt status is unavailable because the graph model carries no acceptance signal. Attempt labels (dev-07, rev-12) are display names assigned in start order; the attempt id column carries the mapped attempt id. Edge tiers: verified means both ends are present in the logs, heuristic means inferred from timing or text. Self-contained page, no external assets.`;
  }

  const api = { N, E, A, RUN, ROLES, MODELS, ROLE_COLOR, MODEL_COLOR, MAX_USD, fmt, esc, color, dashed, radius, svg, text, curve, vcurve, arc, timeTicks, edgeClass, nodeClass, edgeVisible, artVisible };
  function init(items) {
    items.forEach(layout => { layouts.set(layout.key, layout); state.views[layout.key] = viewDefaults(layout); const sec = section(layout.key); sec.querySelector('[data-idea]').textContent = layout.idea; sec.querySelector('[data-graph-title]').textContent = layout.graphTitle; renderViewControls(layout.key); sec.querySelector('[data-controls]').addEventListener('change', ev => viewControlEvent(ev, layout.key)); const graph = sec.querySelector('[data-graph]'); graph.addEventListener('mousemove', ev => graphEvent(ev, layout.key)); graph.addEventListener('mouseleave', ev => graphEvent(ev, layout.key)); graph.addEventListener('click', ev => graphEvent(ev, layout.key)); });
    $('filters').addEventListener('click', filterEvent); $('filters').addEventListener('change', filterEvent);
    $('table').addEventListener('click', tableEvent); $('table').addEventListener('mouseover', tableEvent); $('table').addEventListener('mouseleave', () => hover(null));
    $('card').addEventListener('click', ev => { if (ev.target.closest('[data-close]')) { select(null); return; } const node = ev.target.closest('[data-node-link]'); if (node) { select({ t: 'n', i: +node.dataset.nodeLink }, state.cardView, true); return; } const artifact = ev.target.closest('[data-art-link]'); if (artifact) { const layout = layouts.get(state.cardView), view = state.views[state.cardView]; if (layout.artifactMode !== 'always' && !view.artifacts) { view.artifacts = true; renderViewControls(state.cardView); renderView(state.cardView); } select({ t: 'a', i: +artifact.dataset.artLink }, state.cardView, true); } });
    renderFilters(); renderAccounting(); renderFoot(); renderAll();
    let timer; window.addEventListener('resize', () => { clearTimeout(timer); timer = setTimeout(renderAll, 150); });
  }
  return { init };
})();
""".replace("__MODEL_ORDER__", json.dumps(model_order()))
