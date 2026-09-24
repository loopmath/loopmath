'use strict';
// loopmath views: shared helpers (lane 12). The page renders from the embedded data object only.
const LM = (() => {
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = v => typeof v === 'number' && Number.isFinite(v);
  const data = () => JSON.parse(document.getElementById('data').textContent);
  const $ = id => document.getElementById(id);

  const fmt = {
    usd: v => !num(v) ? 'n/a' : Math.abs(v) >= 1000 ? '$' + Math.round(v).toLocaleString('en-US') : (v !== 0 && Math.abs(v) < 0.1 ? '$' + v.toFixed(3) : '$' + v.toFixed(2)),
    tok: v => !num(v) ? 'n/a' : Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : Math.abs(v) >= 1e3 ? (v / 1e3).toFixed(0) + 'k' : String(Math.round(v)),
    int: v => !num(v) ? 'n/a' : Math.round(v).toLocaleString('en-US'),
    pct: (p, d = 0) => !num(p) ? 'n/a' : (p * 100).toFixed(d) + '%',
    pp: v => !num(v) ? 'n/a' : (v > 0 ? '+' : v < 0 ? '-' : '') + Math.abs(v).toFixed(Math.abs(v) < 10 ? 1 : 0) + ' pp',
    signed: (v, f) => !num(v) ? 'n/a' : (v > 0 ? '+' : v < 0 ? '-' : '') + f(Math.abs(v)),
    x: v => !num(v) ? 'n/a' : Number(v.toPrecision(3)).toString(),
    rounds: v => !num(v) ? 'n/a' : v.toFixed(1),
    date: iso => iso ? String(iso).slice(0, 10) : 'n/a',
    time: iso => iso && String(iso).length >= 16 ? String(iso).slice(11, 16) : '',
    dt: iso => iso ? String(iso).slice(0, 16).replace('T', ' ') : 'n/a',
    age: s => !num(s) ? 'n/a' : s < 90 ? Math.round(s) + ' s' : s < 5400 ? Math.round(s / 60) + ' min' : s < 172800 ? (s / 3600).toFixed(1) + ' h' : (s / 86400).toFixed(1) + ' days',
    score: (v, unit) => !num(v) ? 'n/a' : (Math.abs(v) >= 100 ? Math.round(v).toLocaleString('en-US') : Number(v.toPrecision(3)).toString()) + (unit ? ' ' + unit : ''),
  };
  // Money is dollars with tokens beside them.
  const money = (usd, tok) => `${fmt.usd(usd)} <span class="m">${fmt.tok(tok)} tok</span>`;
  // D107: a mean above its interval's upper end comes from a heavy tail. The mean stays, with this
  // note in the same words in every view (and in views/common.py TAIL_NOTE); the JSON is unchanged.
  const TAIL_NOTE = 'the average is pulled up by rare very large outcomes';
  // The rule is about the prediction, not what is printed (D107 note 2): a mean shown bare, without
  // its bounds, gets the note too. One note covers the values shown together (dollars and tokens).
  const pulledUp = x => !!x && num(x.mean) && num(x.hi) && x.mean > x.hi;
  const tailHtml = (...xs) => xs.some(pulledUp) ? ` <span class="tail m">${TAIL_NOTE}</span>` : '';
  const tailPlain = (...xs) => xs.some(pulledUp) ? '; ' + TAIL_NOTE : '';
  // Interval text for HTML sinks: formatter output is escaped, since a score unit comes from data.
  const ivText = (x, f) => `${esc(f(x.mean))} <span class="rng-t">(${esc(f(x.lo))} to ${esc(f(x.hi))})</span>`;
  const iv = (x, f = fmt.x) => !x ? 'n/a' : ivText(x, f) + tailHtml(x);
  const ivPlain = (x, f = fmt.x) => !x ? 'n/a' : esc(`${f(x.mean)} (${f(x.lo)} to ${f(x.hi)})${tailPlain(x)}`);
  // Timestamps compare by instant, not text: offsets differ between producers. Unparsed ones sort first.
  const ts = s => { const t = s == null || s === '' ? NaN : Date.parse(s); return Number.isFinite(t) ? t : null; };
  const byTime = (x, y) => { const a = ts(x), b = ts(y); return a === null ? (b === null ? String(x || '').localeCompare(String(y || '')) : -1) : b === null ? 1 : a - b; };
  const moneyIv = m => !m ? 'n/a' : `${m.usd ? ivText(m.usd, fmt.usd) : 'n/a'} <span class="m">${m.tokens ? fmt.tok(m.tokens.mean) + ' tok' : ''}</span>${tailHtml(m.usd, m.tokens)}`;
  const arrow = better => better === 'higher' ? ' <span title="higher is better">&#8593;</span>' : better === 'lower' ? ' <span title="lower is better">&#8595;</span>' : '';

  // A thin interval bar on a shared scale, with an optional realized value as a dot.
  function ivBar(x, o = {}) {
    if (!x || !num(x.lo) || !num(x.hi)) return '';
    const log = !!o.log, lo0 = num(o.min) ? o.min : Math.min(x.lo, num(o.realized) ? o.realized : x.lo), hi0 = num(o.max) ? o.max : Math.max(x.hi, num(o.realized) ? o.realized : x.hi);
    const t = v => log ? Math.log(Math.max(v, 1e-9)) : v, a = t(lo0), b = t(hi0), span = (b - a) || 1;
    const pos = v => Math.max(0, Math.min(100, (t(v) - a) / span * 100));
    let html = `<span class="ivbar${o.wide ? ' wide' : ''}" title="${esc([o.title, pulledUp(x) ? TAIL_NOTE : ''].filter(Boolean).join('; '))}"><span class="track"></span><span class="rng" style="left:${pos(x.lo).toFixed(1)}%;width:${Math.max(1, pos(x.hi) - pos(x.lo)).toFixed(1)}%"></span>`;
    if (num(x.mean)) html += `<span class="mid" style="left:${pos(x.mean).toFixed(1)}%"></span>`;
    if (num(o.realized)) html += `<span class="dot${!o.binary && (o.realized < x.lo || o.realized > x.hi) ? ' out' : ''}" style="left:${pos(o.realized).toFixed(1)}%"></span>`;
    return html + '</span>';
  }
  // Support sits next to every estimate, grey when under 5.
  const support = n => num(n) ? `<span class="sup${n < 5 ? ' thin' : ''}" title="recorded runs behind this estimate">${fmt.int(n)} run${n === 1 ? '' : 's'}</span>` : '';
  const shared = on => on ? '<span class="shared" title="computed from the shared prior only; no runs of yours in this group">from shared data</span>' : '';
  const tier = t => t ? `<span class="badge t-${esc(t)}" title="evidence tier">${esc(t)}</span>` : '';
  const zBadge = z => z === 1 ? '<span class="badge acc">accepted</span>' : z === 0 ? '<span class="badge rej">not accepted</span>' : '<span class="badge unk">unknown</span>';

  let tipEl = null;
  function tip(html, x, y) {
    if (!tipEl) { tipEl = document.createElement('div'); tipEl.className = 'tip'; tipEl.hidden = true; document.body.appendChild(tipEl); }
    if (html == null) { tipEl.hidden = true; return; }
    tipEl.innerHTML = html; tipEl.hidden = false;
    const w = tipEl.offsetWidth, h = tipEl.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    let left = x + 14, top = y + 14;
    if (left + w > vw - 8) left = x - w - 10;
    if (top + h > vh - 8) top = y - h - 10;
    tipEl.style.left = Math.max(4, left) + 'px'; tipEl.style.top = Math.max(4, top) + 'px';
  }
  // Copy only: the page never runs anything.
  function copy(text, button) {
    const done = ok => { if (button) { const old = button.textContent; button.textContent = ok ? 'copied' : 'select and copy'; setTimeout(() => { button.textContent = old; }, 1400); } };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(() => done(true), () => done(fallback(text)));
    else done(fallback(text));
  }
  function fallback(text) {
    const area = document.createElement('textarea'); area.value = text; area.style.position = 'fixed'; area.style.opacity = '0';
    document.body.appendChild(area); area.select();
    let ok = false; try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    document.body.removeChild(area); return ok;
  }
  const shq = s => /^[A-Za-z0-9_./:=@%+,-]+$/.test(String(s)) ? String(s) : "'" + String(s).replace(/'/g, "'\"'\"'") + "'";
  const uniq = xs => [...new Set(xs.filter(x => x != null && x !== ''))].sort();
  const options = (values, current, all) => `<option value="">${esc(all)}</option>` + values.map(v => `<option value="${esc(v)}"${String(v) === String(current) ? ' selected' : ''}>${esc(v)}</option>`).join('');

  return { esc, num, data, $, fmt, money, iv, ivPlain, moneyIv, TAIL_NOTE, pulledUp, tailHtml, tailPlain, arrow, ivBar, support, shared, tier, zBadge, tip, copy, shq, uniq, options, ts, byTime };
})();
