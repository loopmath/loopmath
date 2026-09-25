// The results page's detail blocks (lane 2E, 0.2): the estimates by level, one workflow in full and the data behind
// the fit, moved unchanged from the 0.1 posterior page. results.js draws the page and calls LMPosterior.init().
(function () {
  'use strict';
  var D = JSON.parse(document.getElementById('data').textContent);
  var HEADS = D.heads || {};
  var TITLES = {model: 'Model', effort: 'Effort', role: 'Role', topology: 'Workflow shape (topology)',
    type: 'Task type', repo: 'Repo', feature: 'Task features', harness: 'Harness', source: 'Data source',
    gate: 'Gate rule', task: 'Single task', org: 'Organization'};
  var INTERACTIONS = {family_effort: ['family', 'effort'], role_family: ['role', 'family']};
  var EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];  // fit_pricing.EFFORT_ORDER
  var HEAD_ORDER = ['cost', 'tokens', 'success', 'gate'];
  var LEVEL_WORDS = {family_effort: 'family x effort', role_family: 'role x family', model: 'version', psrc: 'position x source', fsrc: 'family x source'};
  var UNTYPED = 'untyped (no task type recorded)';
  var TAIL_NOTE = 'the average is pulled up by rare very large outcomes';  // D107: same words in every view
  var S = {head: null, wf: 0, open: {}};

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
    });
  }
  function fin(x) { return typeof x === 'number' && isFinite(x); }
  function num(x, d) {
    if (!fin(x)) return 'n/a';
    var t = Number(x).toFixed(d == null ? 1 : d);
    if (t.indexOf('.') >= 0) t = t.replace(/0+$/, '').replace(/\.$/, '');
    return t === '-0' ? '0' : t;
  }
  function signed(x, d) { var t = num(x, d); return (t.charAt(0) === '-' || t === '0' || t === 'n/a') ? t : '+' + t; }
  function usd(x) {
    if (!fin(x)) return 'n/a';
    return Math.abs(x) >= 0.01 || x === 0 ? '$' + x.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})
      : '$' + num(x, 4);
  }
  function tokn(x) {
    if (!fin(x)) return 'n/a';
    if (x >= 1e6) return num(x / 1e6) + 'M';
    if (x >= 1e3) return num(x / 1e3, 0) + 'k';
    return String(Math.round(x));
  }
  function tok(x) { return fin(x) ? tokn(x) + ' tokens' : 'n/a'; }
  function rounds(x) { var t = num(x, 2); return t + (t === '1' ? ' round' : ' rounds'); }
  function pct(x) { return fin(x) ? num(x * 100, 0) + '%' : 'n/a'; }
  // D107: a mean above its interval's upper end keeps the mean and gets the note: inside the brackets in running
  // text and tooltips (`bare` leaves it out), on its own line under the value in table cells (`ivCell`, `tailBlock`).
  function above(d) { return !!d && fin(d.mean) && fin(d.hi) && d.mean > d.hi; }
  function tail(d) { return above(d) ? '; ' + TAIL_NOTE : ''; }
  function tailBlock(d) { return above(d) ? '<span class="why">' + esc(TAIL_NOTE) + '</span>' : ''; }
  function iv(d, f, bare) { return d && fin(d.mean) ? f(d.mean) + ' (' + f(d.lo) + ' to ' + f(d.hi) + (bare ? '' : tail(d)) + ')' : 'n/a'; }
  function ivCell(d, f) { return esc(iv(d, f, true)) + tailBlock(d); }
  function meanTail(d) { return above(d) ? ' (' + TAIL_NOTE + ')' : ''; }  // D107 note 2: a mean shown without its bounds
  function kindOf(head) {
    var m = HEADS[head] || {};
    return m.kind || (head === 'cost' || head === 'tokens' ? 'multiplier' : (head === 'success' || head === 'gate') ? 'pp' : 'shift');
  }
  function headLabel(head) {
    if (head === 'cost') return 'Cost';
    if (head === 'success') return 'Success';
    if (head === 'gate') return 'Gate pass';
    if (head === 'tokens') return 'Tokens';
    if (head.indexOf('score:') === 0) return 'Score: ' + head.slice(6);
    return head;
  }
  function headNote(head) {
    var k = kindOf(head), m = HEADS[head] || {};
    var arrow = m.better === 'lower' ? ' Lower is better (↓).' : m.better === 'higher' ? ' Higher is better (↑).' : '';
    if (head.indexOf('score:') === 0 && k === 'multiplier') return 'How much a node multiplies the score ' + head.slice(6) + ' on one run: x1.00 is no change.' + arrow;
    if (head.indexOf('score:') === 0 && k === 'pp') return 'How a node shifts the score ' + head.slice(6) + ' on one run, in percentage points.' + arrow;
    if (head === 'tokens') return 'How much a node multiplies tokens per round, relative to its parent: x1.00 is no change, below x1 is fewer.';
    if (k === 'multiplier') return 'How much a node multiplies dollars per round, relative to its parent: x1.00 is no change, below x1 is cheaper.';
    if (head === 'gate') return 'How a node shifts the chance that a gate passes on a round, in percentage points at the group\'s base rate.';
    if (k === 'pp') return 'How a node shifts the chance of an accepted result, in percentage points at the group\'s base rate.';
    return 'How a node shifts the score ' + head.slice(6) + (m.unit ? ' (in ' + m.unit + ')' : '') + ' on one run.' + arrow;
  }
  function fmtDisplay(n, bare) {
    var d = n.display || {}, k = kindOf(n.head), m = +d.mean || 0, lo = +d.lo || 0, hi = +d.hi || 0;
    var t = bare ? '' : tail({mean: m, hi: hi});
    if (k === 'multiplier') return 'x' + m.toFixed(2) + ' (' + lo.toFixed(2) + ' to ' + hi.toFixed(2) + t + ')';
    if (k === 'pp') return signed(m) + ' pp (' + signed(lo) + ' to ' + signed(hi) + t + ')';
    var unit = (HEADS[n.head] || {}).unit, dg = Math.max(Math.abs(m), Math.abs(lo), Math.abs(hi)) >= 100 ? 0 : 2;
    return signed(m, dg) + (unit ? ' ' + unit : '') + ' (' + signed(lo, dg) + ' to ' + signed(hi, dg) + t + ')';
  }
  function neutral(head) { return kindOf(head) === 'multiplier' ? 1 : 0; }
  function betterOf(head) {
    if (head === 'cost' || head === 'tokens') return 'lower';
    if (head === 'success' || head === 'gate') return 'higher';
    var b = (HEADS[head] || {}).better;
    return b === 'lower' || b === 'higher' ? b : null;
  }
  function goodness(n) {
    var k = kindOf(n.head), m = +(n.display || {}).mean || 0, b = betterOf(n.head);
    var up = k === 'multiplier' ? (m > 0 ? Math.log(m) : 0) : m;
    return b === 'lower' ? -up : b === 'higher' ? up : 0;
  }
  function hasUser(n) { return ((n.source_mix || {}).user || 0) > 0; }
  function mixText(n) {
    var mix = n.source_mix || {}, keys = Object.keys(mix);
    keys.sort(function (a, b) { return (mix[b] || 0) - (mix[a] || 0); });
    return keys.map(function (k) { return k + ' ' + mix[k]; }).join(', ');
  }
  function parts(level) {
    if (INTERACTIONS[level]) return INTERACTIONS[level].slice();
    var seps = [' x ', '_x_', ' × ', '×'];
    for (var i = 0; i < seps.length; i++) if (level.indexOf(seps[i]) >= 0) return level.split(seps[i]).map(function (s) { return s.trim(); });
    return null;
  }
  function splitKey(key, n) {
    var seps = ['|', ' x ', '×', ',', ':', '/'];
    for (var i = 0; i < seps.length; i++) {
      var p = String(key).split(seps[i]).map(function (s) { return s.trim(); });
      if (p.length === n) return p;
    }
    return [String(key)];
  }
  function tipText(n) {
    var lines = [levelWord(n.level) + ' ' + keyText(n) + ' (' + headLabel(n.head) + ')', fmtDisplay(n),
      'model scale: ' + iv(n.effect, function (v) { return num(v, 3); }),
      (n.support || 0) + ' runs' + (mixText(n) ? ': ' + mixText(n) : '')];
    if (n.parent) lines.push('parent: ' + parentText(n.parent));
    return lines.join('\n');
  }

  // ---------------------------------------------------------------- interval bars
  function axis(nodes, head) {
    var log = kindOf(head) === 'multiplier', z = neutral(head), lo = z, hi = z;
    nodes.forEach(function (n) {
      var d = n.display || {};
      [d.lo, d.hi, d.mean].forEach(function (v) {
        if (!fin(v) || (log && v <= 0)) return;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      });
    });
    var f = log ? Math.log : function (v) { return v; };
    var a = f(lo), b = f(hi);
    if (a === b) { a -= 1; b += 1; }
    var pad = (b - a) * 0.06;
    return {f: f, a: a - pad, b: b + pad, zero: f(z), log: log};
  }
  function bar(n, ax) {
    var W = 180, H = 18, d = n.display || {};
    function x(v) { var t = ax.f(ax.log ? Math.max(v, 1e-9) : v); return Math.max(2, Math.min(W - 2, (t - ax.a) / (ax.b - ax.a) * W)); }
    var s = '<svg class="bar" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(fmtDisplay(n)) + '">';
    var z = (ax.zero - ax.a) / (ax.b - ax.a) * W;
    s += '<line x1="' + z.toFixed(1) + '" x2="' + z.toFixed(1) + '" y1="1" y2="' + (H - 1) + '" stroke="var(--neutral)" stroke-width="1" stroke-dasharray="2 2"/>';
    // P6: a dot for the mean on lines for the 80%, 90% and 95% ranges (95% the thinnest). Node summaries carry
    // no draws in the view, so the 90% and 95% ranges are derived from the 80% one (LM.bands).
    var b = typeof LM !== 'undefined' && LM.bands ? LM.bands(d, {log: ax.log}) : (fin(d.lo) && fin(d.hi) ? {'80': [d.lo, d.hi]} : null);
    var widths = {'95': 1, '90': 2, '80': 4};
    if (b) ['95', '90', '80'].forEach(function (k) {
      if (!b[k]) return;
      s += '<line x1="' + x(b[k][0]).toFixed(1) + '" x2="' + x(b[k][1]).toFixed(1) + '" y1="9" y2="9" stroke="var(--dot)" stroke-width="' + widths[k] +
        '" class="v-band v-b' + k + '"/>';
    });
    if (fin(d.mean)) s += '<circle cx="' + x(d.mean).toFixed(1) + '" cy="9" r="4" fill="var(--ink)" stroke="#fff" stroke-width="1.5" class="v-mdot"/>';
    return s + '</svg>';
  }

  // ---------------------------------------------------------------- estimates by level
  function parentRef(parent) {
    if (parent == null) return null;
    var t = String(parent), i = t.indexOf(':');
    if (i < 0 || !/^[a-z_]+$/.test(t.slice(0, i))) return {level: null, key: t};
    var level = t.slice(0, i), key = t.slice(i + 1);
    if ((level === 'repo' || level === 'subtype') && key.indexOf('/') >= 0) key = key.slice(key.indexOf('/') + 1);
    return {level: level, key: key};
  }
  function parentText(parent) {
    var r = parentRef(parent);
    return r ? (r.level ? levelWord(r.level) + ' ' : '') + keyText({level: r.level || '', key: r.key}) : '';
  }
  function levelWord(level) { return LEVEL_WORDS[level] || level; }
  function keyText(n) {
    if (n.level === 'type' && String(n.key) === 'unknown') return UNTYPED;
    var m = n.level === 'position' ? /^(.*)#(\d+)$/.exec(String(n.key)) : null;
    if (m) return 'piece ' + (+m[2] + 1) + ' of ' + m[1];
    var f = n.level.indexOf('feature:') === 0 ? n.level.slice(8) + '=' : null;
    return f && String(n.key).indexOf(f) === 0 ? String(n.key).slice(f.length) : String(n.key);
  }
  function depthMap(nodes) {
    var byKey = {}, byFull = {}, depth = {};
    nodes.forEach(function (n) {
      if (parts(n.level)) return;
      byKey[n.head + '\u0000' + n.key] = n;
      byFull[n.head + '\u0000' + n.level + '\u0000' + n.key] = n;
    });
    function dep(n, guard) {
      var id = n.head + '\u0000' + n.level + '\u0000' + n.key;
      if (depth[id] != null) return depth[id];
      var r = parentRef(n.parent);
      var p = !r ? null : r.level ? byFull[n.head + '\u0000' + r.level + '\u0000' + r.key] : byKey[n.head + '\u0000' + r.key];
      depth[id] = (p && p !== n && guard < 8) ? dep(p, guard + 1) + 1 : 0;
      return depth[id];
    }
    return function (n) { return dep(n, 0); };
  }
  function pageAxes(nodes) {
    var by = {}, axes = {};
    nodes.forEach(function (n) { if (!parts(n.level)) (by[n.head] = by[n.head] || []).push(n); });
    Object.keys(by).forEach(function (h) { axes[h] = axis(by[h], h); });
    return axes;
  }
  function allNodes() {
    var out = [];
    Object.keys(D.levels || {}).forEach(function (k) { out = out.concat(D.levels[k] || []); });
    return out;
  }
  var SHOWN = 30;
  function nodeRows(nodes, opts) {
    opts = opts || {};
    if (opts.limit && nodes.length > opts.limit + 5) {
      var rest = {showHead: opts.showHead, axes: opts.axes || pageAxes(nodes)};
      return nodeRows(nodes.slice(0, opts.limit), rest) + '<details class="more"><summary>Show ' + (nodes.length - opts.limit) +
        ' more (fewer runs)</summary>' + nodeRows(nodes.slice(opts.limit), rest) + '</details>';
    }
    var heads = [];
    nodes.forEach(function (n) { if (heads.indexOf(n.head) < 0) heads.push(n.head); });
    var axes = opts.axes || pageAxes(nodes);
    var depth = depthMap(nodes), showHead = heads.length > 1 || opts.showHead;
    var s = '<div class="scroll"><table class="nodes"><colgroup><col class="c-node"><col class="c-eff"><col class="c-bar bar"><col class="c-sup">' +
      '<col class="c-mix"></colgroup><thead><tr><th>Estimate</th><th data-first="desc">Effect (80% range)</th><th class="bar" data-nosort>Range</th>' +
      '<th class="num">Runs</th><th>Sources</th></tr></thead><tbody>';
    nodes.forEach(function (n) {
      var pad = depth(n) * 18, notes = '';
      if (showHead) notes += '<span class="tag head">' + esc(headLabel(n.head)) + '</span>';
      if (!hasUser(n)) notes += '<span class="tag">from shared data</span>';
      var why = '';
      if (!(n.support > 0)) why = n.parent != null ? 'No runs here: the parent\'s estimate (' + esc(parentText(n.parent)) + '), widened.'
        : 'No runs here: the prior for this level, widened.';
      s += '<tr data-tip="' + esc(tipText(n)) + '"><td class="node" style="padding-left:' + (8 + pad) + 'px">' +
        '<span class="lvl">' + esc(levelWord(n.level)) + '</span> ' + esc(keyText(n)) + notes + (why ? '<span class="why">' + why + '</span>' : '') + '</td>' +
        '<td class="eff">' + esc(fmtDisplay(n, true)) + tailBlock({mean: +(n.display || {}).mean, hi: +(n.display || {}).hi}) + '</td><td class="bar">' + bar(n, axes[n.head]) + '</td>' +
        '<td class="num"><span class="sup' + ((n.support || 0) < 5 ? ' low' : '') + '">' + esc(n.support || 0) + '</span></td>' +
        '<td class="mix">' + esc(mixText(n)) + '</td></tr>';
    });
    return s + '</tbody></table></div>';
  }
  function heatScale(n, maxg) {
    var k = kindOf(n.head), g = goodness(n);
    if (k === 'multiplier') return g / Math.LN2;
    if (k === 'pp') return g / 15;
    return maxg > 0 ? g / maxg : 0;
  }
  function heatColor(t) {
    var mid = [240, 239, 236], pole = t >= 0 ? [42, 120, 214] : [227, 73, 72], a = Math.min(1, Math.abs(t));
    var c = mid.map(function (m, i) { return Math.round(m + (pole[i] - m) * a); });
    return {bg: 'rgb(' + c.join(',') + ')', fg: a > 0.55 ? '#fff' : 'var(--ink)'};
  }
  function heatNote(head) {
    var k = kindOf(head);
    if (!betterOf(head)) return ' (this score records no better direction, so cells stay uncolored)';
    return k === 'multiplier' ? ' (full color at x0.5 or x2)' : k === 'pp' ? ' (full color at 15 points)' : ' (scaled to the largest shift)';
  }
  function effortOrder(list) {  // low to max; efforts the list does not know keep their order at the end
    var rank = function (x) { var i = EFFORTS.indexOf(x); return i < 0 ? EFFORTS.length : i; };
    return list.map(function (x, i) { return [x, i]; })
      .sort(function (a, b) { return rank(a[0]) - rank(b[0]) || a[1] - b[1]; }).map(function (a) { return a[0]; });
  }
  function heatTable(level, nodes) {
    var p = parts(level), rows = [], cols = [], cell = {}, maxg = 0;
    nodes.forEach(function (n) {
      var k = splitKey(n.key, p.length), r = k[0], c = k.slice(1).join(' | ');
      if (rows.indexOf(r) < 0) rows.push(r);
      if (cols.indexOf(c) < 0) cols.push(c);
      cell[r + '\u0000' + c] = n;
      maxg = Math.max(maxg, Math.abs(goodness(n)));
    });
    if (p[0] === 'effort') rows = effortOrder(rows);
    if (p.length === 2 && p[1] === 'effort') cols = effortOrder(cols);
    var s = '<h3>' + esc(levelWord(level)) + ' <span class="tag head">' + esc(headLabel(nodes[0].head)) + '</span></h3>' +
      '<p class="legend">Each cell: the effect of this ' + esc(p[0]) + ' at this ' + esc(p.slice(1).join(' and ')) +
      '. <span class="sw" style="background:rgb(42,120,214)"></span>better <span class="sw" style="background:rgb(240,239,236)"></span>no change ' +
      '<span class="sw" style="background:rgb(227,73,72)"></span>worse' + heatNote(nodes[0].head) + '. Hover a cell for its range and runs.</p>' +
      '<div class="scroll"><table class="heat"><thead><tr><th class="rowh">' + esc(p[0]) + ' \\ ' + esc(p.slice(1).join(' | ')) + '</th>';
    cols.forEach(function (c) { s += '<th>' + esc(c) + '</th>'; });
    s += '</tr></thead><tbody>';
    rows.forEach(function (r) {
      s += '<tr><th class="rowh">' + esc(r) + '</th>';
      cols.forEach(function (c) {
        var n = cell[r + '\u0000' + c];
        if (!n) { s += '<td class="empty">none</td>'; return; }
        var col = heatColor(heatScale(n, maxg)), d = n.display || {};
        var txt = kindOf(n.head) === 'multiplier' ? 'x' + (+d.mean || 0).toFixed(2) : signed(d.mean) + (kindOf(n.head) === 'pp' ? ' pp' : '');
        s += '<td style="background:' + col.bg + ';color:' + col.fg + '" data-tip="' + esc(tipText(n)) + '">' + esc(txt) +
          ((n.support || 0) < 5 ? '*' : '') + '</td>';
      });
      s += '</tr>';
    });
    var few = nodes.some(function (n) { return (n.support || 0) < 5; });
    var tailed = nodes.filter(function (n) { return above(n.display); }).map(keyText);  // cells show the mean only (D107)
    return s + '</tbody></table></div>' + (few ? '<p class="legend">* fewer than 5 runs.</p>' : '') +
      (tailed.length ? '<p class="legend">' + esc(tailed.join(', ')) + ': ' + esc(TAIL_NOTE) + '.</p>' : '');
  }
  function sectionHtml(name, nodes) {
    var title = TITLES[name] || (name.indexOf('score:') === 0 ? 'Score ' + name.slice(6) : name);
    var shown = S.head === 'all' ? nodes : nodes.filter(function (n) { return n.head === S.head; });
    var s = '<section class="level" data-section="' + esc(name) + '"><h2>' + esc(title) + '</h2>';
    if (!shown.length) {
      s += '<p class="empty">' + (nodes.length ? 'No ' + esc(headLabel(S.head).toLowerCase()) + ' estimates at this level. Other heads have ' + nodes.length + '.'
        : 'No estimates at this level yet.') + '</p>';
      return s + '</section>';
    }
    var plain = shown.filter(function (n) { return !parts(n.level); });
    var inter = shown.filter(function (n) { return parts(n.level); });
    if (name === 'feature') {
      var groups = {}, order = [];
      plain.forEach(function (n) {
        var g = n.level.indexOf('feature:') === 0 ? n.level.slice(8) : n.level;
        if (!groups[g]) { groups[g] = []; order.push(g); }
        groups[g].push(n);
      });
      order.forEach(function (g) { s += '<h3>' + esc(g) + '</h3>' + nodeRows(groups[g], {showHead: S.head === 'all', axes: S.axes, limit: SHOWN}); });
    } else if (plain.length) {
      s += nodeRows(plain, {showHead: S.head === 'all', axes: S.axes, limit: SHOWN});
    }
    var byLevel = {}, lv = [];
    inter.forEach(function (n) {
      var k = n.level + '\u0000' + n.head;
      if (!byLevel[k]) { byLevel[k] = []; lv.push(k); }
      byLevel[k].push(n);
    });
    lv.forEach(function (k) { s += heatTable(k.split('\u0000')[0], byLevel[k]); });
    return s + '</section>';
  }
  function headList() {
    var hs = Object.keys(HEADS);
    Object.keys(D.levels || {}).forEach(function (k) {
      (D.levels[k] || []).forEach(function (n) { if (hs.indexOf(n.head) < 0) hs.push(n.head); });
    });
    var rank = function (h) { var i = HEAD_ORDER.indexOf(h); return i < 0 ? HEAD_ORDER.length : i; };
    return hs.map(function (h, i) { return [h, i]; })
      .sort(function (a, b) { return rank(a[0]) - rank(b[0]) || a[1] - b[1]; }).map(function (a) { return a[0]; });
  }
  function levelsHtml(head) {
    if (head) S.head = head;
    var hs = headList();
    var s = '<div class="heads" role="group" aria-label="Head">';
    hs.concat(hs.length > 1 ? ['all'] : []).forEach(function (h) {
      s += '<button type="button" data-head="' + esc(h) + '" aria-pressed="' + (S.head === h) + '">' + esc(h === 'all' ? 'All heads' : headLabel(h)) + '</button>';
    });
    S.axes = pageAxes(allNodes());
    s += '</div><p class="note">' + (S.head === 'all' ? 'Every head, each row tagged with its head. Bars of one head share an axis across the page.'
      : esc(headNote(S.head))) + ' Grey run counts are under 5. The dashed line on each bar is no change.</p>';
    var names = Object.keys(D.levels || {});
    if (!names.length) return s + '<p class="empty">This fit reports no estimates.</p>';
    names.forEach(function (name) { s += sectionHtml(name, D.levels[name] || []); });
    return s;
  }

  // ---------------------------------------------------------------- one workflow in full
  function layout(g) {
    var nodes = g.nodes || [], ids = nodes.map(function (n) { return n.id; }), out = {}, indeg = {};
    ids.forEach(function (id) { out[id] = []; indeg[id] = 0; });
    (g.edges || []).forEach(function (e) {
      var a = e.from != null ? e.from : e[0], b = e.to != null ? e.to : e[1];
      if (out[a] && out[b]) { out[a].push(b); indeg[b]++; }
    });
    var color = {}, fwd = [], back = [];
    function dfs(u) {
      color[u] = 1;
      out[u].forEach(function (v) {
        if (color[v] === 1) { back.push([u, v]); return; }
        fwd.push([u, v]);
        if (!color[v]) dfs(v);
      });
      color[u] = 2;
    }
    ids.filter(function (id) { return !indeg[id]; }).concat(ids).forEach(function (id) { if (!color[id]) dfs(id); });
    var layer = {}, changed = true, guard = 0;
    ids.forEach(function (id) { layer[id] = 0; });
    while (changed && guard++ < ids.length + 2) {
      changed = false;
      fwd.forEach(function (e) { if (layer[e[1]] < layer[e[0]] + 1) { layer[e[1]] = layer[e[0]] + 1; changed = true; } });
    }
    var cols = [];
    ids.forEach(function (id) { (cols[layer[id]] = cols[layer[id]] || []).push(id); });
    var maxRows = Math.max.apply(null, cols.map(function (c) { return c ? c.length : 0 }).concat([1]));
    var kind = {}, RH = 130, pos = {}, x = 30;
    nodes.forEach(function (n) { kind[n.id] = n.kind || 'piece'; });
    cols.forEach(function (c, i) {
      var cw = (c || []).some(function (id) { return kind[id] === 'piece'; }) ? 262 : 168;
      (c || []).forEach(function (id, j) { pos[id] = {x: x, y: 40 + (j + (maxRows - c.length) / 2) * RH}; });
      x += cw;
    });
    return {pos: pos, fwd: fwd, back: back, width: x + 30, height: 40 + (maxRows - 1) * RH + 86 + 30};
  }
  // I12: a piece of width n is drawn as n worker boxes (up to MAX_WORKERS; above that one box marked xn), each with
  // its setting and its part of the piece's cost. The cost model prices a piece as its width times one worker
  // (belief/compose.py), so one worker's cost and range are the piece's divided by n.
  var MAX_WORKERS = 6;
  function copy(o, extra) { var r = {}; Object.keys(o || {}).forEach(function (k) { r[k] = o[k]; }); Object.keys(extra || {}).forEach(function (k) { r[k] = extra[k]; }); return r; }
  function scaled(d, f) { if (!d) return d; var r = copy(d); ['mean', 'lo', 'hi', 'median'].forEach(function (k) { if (fin(d[k])) r[k] = d[k] * f; }); return r; }
  function perWorker(pr, n) {
    if (!pr) return pr;
    var r = copy(pr);
    ['cost', 'cost_per_round'].forEach(function (k) { if (pr[k]) r[k] = copy(pr[k], {usd: scaled(pr[k].usd, 1 / n), tokens: scaled(pr[k].tokens, 1 / n)}); });
    return r;
  }
  function workers(w) {
    var g = w.graph || {}, wide = {};
    (g.nodes || []).forEach(function (n) { if ((n.kind || 'piece') === 'piece' && n.width > 1 && n.width <= MAX_WORKERS) wide[n.id] = n.width; });
    if (!Object.keys(wide).length) return w;
    function ids(id) { var out = []; for (var k = 1; k <= (wide[id] || 0); k++) out.push(id + '#' + k); return out.length ? out : [id]; }
    function last(id) { var l = ids(id); return l[l.length - 1]; }
    var nodes = [], per = copy(w.per_piece), shares = copy(w.shares);
    (g.nodes || []).forEach(function (n) {
      if (!wide[n.id]) { nodes.push(n); return; }
      var pr = perWorker((w.per_piece || {})[n.id] || n.prediction, wide[n.id]);
      ids(n.id).forEach(function (id, k) {
        nodes.push(copy(n, {id: id, piece: n.id, name: n.id + ' ' + (k + 1) + ' of ' + wide[n.id], worker: k + 1, workers: wide[n.id], width: 1, prediction: pr}));
        per[id] = pr;
        shares[id] = fin((w.shares || {})[n.id]) ? w.shares[n.id] / wide[n.id] : null;
      });
    });
    var edges = [];
    (g.edges || []).forEach(function (e) {
      var a = e.from != null ? e.from : e[0], b = e.to != null ? e.to : e[1];
      ids(a).forEach(function (x) { ids(b).forEach(function (y) { edges.push({from: x, to: y}); }); });
    });
    return copy(w, {graph: copy(g, {nodes: nodes, edges: edges}), per_piece: per, shares: shares,
      gates: (w.gates || []).map(function (gt) { return copy(gt, {after: gt.after && last(gt.after)}); }),
      loops: (w.loops || []).map(function (lp) { return copy(lp, {from: ids(lp.from)[0], to: last(lp.to)}); })});
  }
  function graphSvg(w) {
    w = workers(w);
    var g = w.graph || {}, L = layout(g), byId = {}, shares = w.shares || {}, per = w.per_piece || {};
    (g.nodes || []).forEach(function (n) { byId[n.id] = n; });
    var PW = 214, PH = 86, AW = 120, AH = 28;
    function box(id) {
      var n = byId[id], p = L.pos[id], piece = (n.kind || 'piece') === 'piece';
      return {x: p.x, y: p.y, w: piece ? PW : AW, h: piece ? PH : AH, cy: p.y + (piece ? PH : AH) / 2};
    }
    var loops = w.loops || [], H = L.height + (L.back.length ? 30 : 0) + loops.length * 30, W = L.width;
    (w.gates || []).forEach(function (gt) {
      if (gt.after && L.pos[gt.after]) W = Math.max(W, box(gt.after).x + gateText(gt).length * 7 + 20);
    });
    var s = '<svg class="graph" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" style="min-width:' + Math.round(W * 0.75) + 'px"' +
      ' role="img" aria-label="workflow graph">' +
      '<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">' +
      '<path d="M0,0 L8,4 L0,8 z" fill="#8a8984"/></marker></defs>';
    L.fwd.concat(L.back).forEach(function (e, i) {
      var a = box(e[0]), b = box(e[1]), isBack = i >= L.fwd.length;
      var x1 = a.x + a.w, y1 = a.cy, x2 = b.x, y2 = b.cy, mx = (x1 + x2) / 2;
      var d = isBack ? 'M' + (a.x + a.w / 2) + ',' + (a.y + a.h) + ' C' + (a.x + a.w / 2) + ',' + (a.y + a.h + 50) + ' ' + (b.x + b.w / 2) + ',' + (b.y + b.h + 50) + ' ' + (b.x + b.w / 2) + ',' + (b.y + b.h)
        : 'M' + x1 + ',' + y1 + ' C' + mx + ',' + y1 + ' ' + mx + ',' + y2 + ' ' + x2 + ',' + y2;
      s += '<path d="' + d + '" fill="none" stroke="#8a8984" stroke-width="1.5"' + (isBack ? ' stroke-dasharray="4 3"' : '') + ' marker-end="url(#arr)"/>';
    });
    loops.forEach(function (lp, i) {
      if (!L.pos[lp.from] || !L.pos[lp.to]) return;
      var a = box(lp.to), b = box(lp.from), y = L.height - 4 + (L.back.length ? 30 : 0) + i * 30;
      var self = lp.to === lp.from ? 8 : 0, ax = a.x + a.w / 2 + i * 14 + self, bx = b.x + b.w / 2 - i * 14 - self;
      s += '<path d="M' + ax + ',' + (a.y + a.h) + ' L' + ax + ',' + y + ' L' + bx + ',' + y + ' L' + bx + ',' + (b.y + b.h) + '" fill="none" stroke="#1c5cab" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#arr)"/>';
      var label = 'repair, ' + loopText(lp) + ': ' + (lp.rounds ? rounds(lp.rounds.mean) + ', ' : '') + 'reruns pieces with ' + pct(lp.share) + ' of cost';
      s += '<text x="' + ((self ? Math.max(ax, bx) : Math.min(ax, bx)) + 6) + '" y="' + (y - 5) + '" font-size="12" fill="#184f95" data-tip="' + esc(loopTip(lp)) + '">' + esc(label) + '</text>';
    });
    (g.nodes || []).forEach(function (n) {
      var b = box(n.id);
      if ((n.kind || 'piece') !== 'piece') {
        s += '<g data-tip="' + esc('artifact ' + n.id) + '"><rect x="' + b.x + '" y="' + b.y + '" width="' + b.w + '" height="' + b.h + '" rx="14" fill="#f3f2ef" stroke="#c9c8c3"/>' +
          '<text x="' + (b.x + b.w / 2) + '" y="' + (b.y + 18) + '" font-size="12" text-anchor="middle" fill="#52514e">' + esc(n.id) + '</text></g>';
        return;
      }
      var pr = per[n.id] || n.prediction || {}, cost = pr.cost || {}, st = n.setting || {};
      var lines = [
        [esc(n.name || n.id) + ' <tspan fill="#52514e" font-weight="400">(' + esc(n.role || '') + (n.width > 1 ? ' x' + n.width : '') + ')</tspan>', 13, 600],
        [esc((st.model || 'no setting') + (st.effort ? '/' + st.effort : '')), 12, 400],
        [esc(costLabel(pr)), 12, 400],
        [esc(tok((cost.tokens || {}).mean) + ' per run, ' + pct(shares[n.id]) + ' of cost'), 12, 400]];
      s += '<g data-tip="' + esc(pieceTip(w, n)) + '">';
      for (var k = Math.min(2, (n.width || 1) - 1); k > 0; k--) {
        s += '<rect x="' + (b.x + 4 * k) + '" y="' + (b.y - 4 * k) + '" width="' + b.w + '" height="' + b.h + '" rx="8" fill="#fff" stroke="#9ec5f4" stroke-width="1.5"/>';
      }
      s += '<rect x="' + b.x + '" y="' + b.y + '" width="' + b.w + '" height="' + b.h + '" rx="8" fill="#fff" stroke="#2a78d6" stroke-width="1.5"/>';
      lines.forEach(function (l, i) {
        s += '<text x="' + (b.x + 10) + '" y="' + (b.y + 20 + i * 19) + '" font-size="' + l[1] + '" font-weight="' + l[2] + '" fill="#0b0b0b">' + l[0] + '</text>';
      });
      var sh = Math.max(0, Math.min(1, shares[n.id] || 0));
      s += '<rect x="' + b.x + '" y="' + (b.y + b.h - 4) + '" width="' + (b.w * sh).toFixed(1) + '" height="4" fill="#9ec5f4"/></g>';
    });
    (w.gates || []).forEach(function (gt) {
      if (!gt.after || !L.pos[gt.after]) return;
      var b = box(gt.after), text = gateText(gt);
      s += '<g data-tip="' + esc(gateTip(gt)) + '"><path d="M' + (b.x + b.w + 10) + ',' + (b.cy - 8) + ' l8,8 l-8,8 l-8,-8 z" fill="#1c5cab"/>' +
        '<text x="' + (b.x) + '" y="' + (b.y - 8) + '" font-size="12" fill="#184f95">' + esc(text) + '</text></g>';
    });
    return s + '</svg>';
  }
  function gateText(gt) {
    return 'gate ' + gt.id + ': pass ' + (gt.pass ? pct(gt.pass.mean) : 'n/a') + ', ' + rounds(gt.rounds ? gt.rounds.mean : 1);
  }
  function loopTip(lp) {
    return 'repair loop after gate ' + lp.gate + (lp.rule ? ' (' + lp.rule + ')' : '') + '\npieces: ' + (lp.pieces || []).join(', ') +
      '\nexpected rounds: ' + iv(lp.rounds, function (v) { return num(v, 2); }) + '\nthese pieces\' share of the run\'s cost: ' + pct(lp.share);
  }
  function gateTip(gt) {
    return 'gate ' + gt.id + (gt.rule ? ' (' + gt.rule + ')' : '') + ' after ' + gt.after + (gt.on_fail ? ', on fail back to ' + gt.on_fail : ', on fail stop') +
      '\npass chance per round: ' + iv(gt.pass, pct) + '\nexpected rounds: ' + iv(gt.rounds, function (v) { return num(v, 2); });
  }
  var REFS = null;
  function effNodes(refs) {
    if (!REFS) {
      REFS = {};
      allNodes().forEach(function (nd) { (REFS[nd.level + ':' + nd.key] = REFS[nd.level + ':' + nd.key] || []).push(nd); });
    }
    var out = [];
    (refs || []).forEach(function (r) { (REFS[r] || []).forEach(function (nd) { if (S.head === 'all' || nd.head === S.head) out.push(nd); }); });
    return out;
  }
  function loopText(lp) { return lp.to === lp.from ? lp.to + ' retries itself' : lp.to + ' back to ' + lp.from; }
  // D60: `cost` is the piece's whole-run contribution, `cost_per_round` one execution (absent before lane 5 fills it).
  function perRound(pr) { var c = pr.cost_per_round; return c && c.usd && fin(c.usd.mean) ? c : null; }
  function costLabel(pr) {
    var c = perRound(pr);
    return c ? iv(c.usd, usd, true) + ' per round' : iv((pr.cost || {}).usd, usd, true) + ' per run';
  }
  function costAbove(pr) { var c = perRound(pr); return above(c ? c.usd : (pr.cost || {}).usd); }
  function pieceTip(w, n) {
    var pr = (w.per_piece || {})[n.id] || n.prediction || {}, cost = pr.cost || {}, st = n.setting || {};
    var lines = [(n.name || n.id) + ' (' + (n.role || '') + (n.width > 1 ? ', ' + n.width + ' parallel copies' : '') + ')',
      (st.harness || '') + ' ' + (st.model || '') + (st.effort ? '/' + st.effort : ''),
      'cost per round: ' + (perRound(pr) ? iv(perRound(pr).usd, usd) : 'not given'), 'cost per run: ' + iv(cost.usd, usd),
      'tokens per run: ' + iv(cost.tokens, tokn),
      'expected rounds: ' + iv(pr.rounds, function (v) { return num(v, 2); }), 'share of the run\'s cost: ' + pct((w.shares || {})[n.id])];
    if (pr.gate_pass) lines.push('gate pass per round: ' + iv(pr.gate_pass, pct));
    if (n.workers) lines.push('one of ' + n.workers + ' parallel workers of piece ' + n.piece + '; the figures above are one worker\'s part');
    effNodes(((w.effects || {}).pieces || {})[n.piece || n.id]).forEach(function (e) {
      lines.push(levelWord(e.level) + ' ' + keyText(e) + ' [' + headLabel(e.head) + ']: ' + fmtDisplay(e));
    });
    return lines.join('\n');
  }
  function predictionLine(p) {
    if (!p) return '';
    var s = 'Predicted run: ' + iv((p.cost || {}).usd, usd) + ', ' + tok(((p.cost || {}).tokens || {}).mean) + meanTail((p.cost || {}).tokens) +
      '; chance of an accepted result ' + iv(p.p_success, pct) + '; repair rounds ' + iv(p.rounds, function (v) { return num(v, 2); });
    if (p.support === 0) s += '; no runs yet in any group close to this configuration';
    else if (p.support != null) s += '; ' + p.support + (p.support === 1 ? ' run' : ' runs') + ' in the tightest group with data';
    return '<p class="note">' + esc(s) + '.</p>';
  }
  // I11: every configuration, grouped by workflow graph in the order the view gives; a group opens when the
  // drawn configuration is in it or the reader opened it.
  function groupsOf(list) {
    var order = [], by = {};
    list.forEach(function (x, j) {
      var gname = x.group || ((x.label || '').split(':')[0]) || 'workflow';
      if (!by[gname]) { by[gname] = []; order.push(gname); }
      by[gname].push(j);
    });
    return order.map(function (gname) { return {name: gname, items: by[gname]}; });
  }
  function targetOf(p) {
    var t = D.target;
    if (!t || !p) return null;
    if (t.head === 'success') return {d: p.p_success, f: pct};
    var sc = (p.scores || {})[t.head.split(':').slice(1).join(':')];
    return sc && sc.value ? {d: sc.value, f: function (v) { return num(v, 0) + (sc.unit ? ' ' + sc.unit : ''); }} : null;
  }
  function targetWord() {
    var t = D.target;
    if (!t) return '';
    return t.head === 'success' ? 'chance of an accepted result' : 'expected ' + t.head.split(':').slice(1).join(':');
  }
  function pickerHtml(list) {
    var groups = groupsOf(list), tw = targetWord();
    var s = '<p><label>Configuration: <select id="wfpick">';
    groups.forEach(function (gr) {
      s += '<optgroup label="' + esc(gr.name + ' (' + gr.items.length + ')') + '">';
      gr.items.forEach(function (j) {
        var x = list[j];
        s += '<option value="' + j + '"' + (j === S.wf ? ' selected' : '') + '>' + esc(x.label || x.config) + (x.origin ? ' (' + esc(x.origin) + ')' : '') + '</option>';
      });
      s += '</optgroup>';
    });
    s += '</select></label></p>';
    s += '<p class="note">' + list.length + ' configurations in ' + groups.length + (groups.length === 1 ? ' workflow graph' : ' workflow graphs') +
      ', grouped by graph. In a group: your usual first, then by recorded runs' + (tw ? ', ties broken by ' + esc(tw) : '') + '. Click one to draw it.</p>';
    groups.forEach(function (gr) {
      var open = gr.items.indexOf(S.wf) >= 0 || S.open[gr.name];
      s += '<details class="wfgroup"' + (open ? ' open' : '') + '><summary data-group="' + esc(gr.name) + '"><b>' + esc(gr.name) + '</b> <span class="lvl">' +
        gr.items.length + (gr.items.length === 1 ? ' configuration' : ' configurations') + '</span></summary>' +
        '<div class="scroll"><table class="compact wflist"><thead><tr><th>Configuration</th><th class="num">Runs</th>' + (tw ? '<th class="num">' + esc(tw) + '</th>' : '') +
        '<th class="num">Cost per run</th></tr></thead><tbody>';
      gr.items.forEach(function (j) {
        var x = list[j], p = x.prediction || {}, t = targetOf(p), cost = (p.cost || {}).usd;
        s += '<tr data-wf="' + j + '"' + (j === S.wf ? ' class="on"' : '') + '><td><a href="#graph" data-wf="' + j + '">' + esc(x.label || x.config) + '</a>' +
          (x.origin === 'usual' ? ' <span class="tag">usual</span>' : '') + '</td><td class="num">' + (fin(x.runs) ? x.runs : '') + '</td>' +
          (tw ? '<td class="num">' + (t && t.d && fin(t.d.mean) ? esc(t.f(t.d.mean)) : 'n/a') + '</td>' : '') +
          '<td class="num">' + (cost && fin(cost.mean) ? esc(usd(cost.mean)) : 'n/a') + '</td></tr>';
      });
      s += '</tbody></table></div></details>';
    });
    return s;
  }
  function graphHtml(i) {
    var list = (D.workflows && D.workflows.length) ? D.workflows : (D.workflow ? [D.workflow] : []);
    if (!list.length) {
      return '<p class="empty">No configuration to draw. Pass <code>--workflow CFG</code> or <code>--workflow FILE.toml</code>, ' +
        'or record runs for this task type and repo, then run <code>loopmath posterior --html</code> again.</p>';
    }
    if (i != null) S.wf = i;
    if (S.wf >= list.length) S.wf = 0;
    var w = list[S.wf], s = '';
    if (list.length > 1) s += pickerHtml(list);
    s += '<h2>' + esc(w.label || w.config) + '</h2><p class="note">Configuration <code>' + esc(w.config) + '</code>. ' +
      'Each piece shows its predicted cost per round with an 80% range (per run when the fit gives no per-round figure), ' +
      'its tokens per run and its share of the run\'s cost. ' +
      'Gates show the chance of passing per round and the expected rounds; dashed arcs are repair loops. Hover for details.</p>';
    s += predictionLine(w.prediction);
    s += '<div class="graphwrap">' + graphSvg(w) + '</div>';
    var per = w.per_piece || {}, shares = w.shares || {}, pieces = ((w.graph || {}).nodes || []).filter(function (n) { return (n.kind || 'piece') === 'piece'; });
    var tailed = [];  // what the graph's boxes and labels show without the note (D107 and its note 2)
    pieces.forEach(function (n) {
      var pr = per[n.id] || n.prediction || {};
      if (costAbove(pr)) tailed.push('cost of ' + n.id);
      if (above((pr.cost || {}).tokens)) tailed.push('tokens of ' + n.id);
    });
    (w.gates || []).forEach(function (gt) {
      if (above(gt.pass)) tailed.push('pass chance of gate ' + gt.id);
      if (above(gt.rounds)) tailed.push('rounds of gate ' + gt.id);
    });
    (w.loops || []).forEach(function (lp) { if (above(lp.rounds)) tailed.push('rounds of the repair ' + loopText(lp)); });
    if (tailed.length) s += '<p class="note">In the graph, ' + esc(tailed.join(', ')) + ': ' + esc(TAIL_NOTE) + '.</p>';
    s += '<h3>Pieces</h3><div class="scroll"><table><thead><tr><th>Piece</th><th>Setting</th><th class="num">Cost per round</th>' +
      '<th class="num">Cost per run</th><th class="num">Tokens per run</th><th class="num">Expected rounds</th><th class="num">Share of cost</th><th class="num">Gate pass per round</th></tr></thead><tbody>';
    pieces.forEach(function (n) {
      var pr = per[n.id] || n.prediction || {}, cost = pr.cost || {}, st = n.setting || {};
      s += '<tr><td>' + esc(n.id) + ' <span class="lvl">' + esc(n.role || '') + (n.width > 1 ? ' x' + n.width : '') + '</span></td>' +
        '<td>' + esc((st.harness || '') + ' ' + (st.model || '') + (st.effort ? '/' + st.effort : '')) + '</td>' +
        '<td class="num">' + (perRound(pr) ? ivCell(perRound(pr).usd, usd) : 'n/a') + '</td>' +
        '<td class="num">' + ivCell(cost.usd, usd) + '</td><td class="num">' + ivCell(cost.tokens, tokn) + '</td>' +
        '<td class="num">' + ivCell(pr.rounds, function (v) { return num(v, 2); }) + '</td><td class="num">' + esc(pct(shares[n.id])) + '</td>' +
        '<td class="num">' + (pr.gate_pass ? ivCell(pr.gate_pass, pct) : '') + '</td></tr>';
    });
    s += '</tbody></table></div>';
    if ((w.gates || []).length) {
      s += '<h3>Gates</h3><div class="scroll"><table><thead><tr><th>Gate</th><th>After</th><th>On fail</th>' +
        '<th class="num">Pass chance per round</th><th class="num">Expected rounds</th></tr></thead><tbody>';
      w.gates.forEach(function (gt) {
        var byRound = (gt.pass_by_round || []).map(function (p, k) { return 'round ' + (k + 1) + ': ' + pct(p.mean) + meanTail(p); }).join(', ');
        s += '<tr><td>' + esc(gt.id) + (gt.rule ? ' <span class="lvl">' + esc(gt.rule) + '</span>' : '') + '</td><td>' + esc(gt.after || '') + '</td>' +
          '<td>' + esc(gt.on_fail || 'stop') + '</td><td class="num">' + ivCell(gt.pass, pct) + (byRound ? '<span class="why">' + esc(byRound) + '</span>' : '') + '</td>' +
          '<td class="num">' + ivCell(gt.rounds, function (v) { return num(v, 2); }) + '</td></tr>';
      });
      s += '</tbody></table></div>';
    }
    if ((w.loops || []).length) {
      s += '<h3>Repair loops</h3><div class="scroll"><table><thead><tr><th>Loop</th><th>Pieces</th><th class="num">Expected rounds</th>' +
        '<th class="num">Pieces\' share of cost</th></tr></thead><tbody>';
      w.loops.forEach(function (lp) {
        s += '<tr><td>' + esc(loopText(lp)) + ' <span class="lvl">gate ' + esc(lp.gate) + '</span></td><td>' + esc((lp.pieces || []).join(', ')) + '</td>' +
          '<td class="num">' + ivCell(lp.rounds, function (v) { return num(v, 2); }) + '</td><td class="num">' + esc(pct(lp.share)) + '</td></tr>';
      });
      s += '</tbody></table></div>';
    }
    var eff = w.effects || {}, any = false;
    var axes = pageAxes(effNodes([].concat.apply(eff.workflow || [], Object.keys(eff.pieces || {}).map(function (k) { return eff.pieces[k]; }))));
    var block = '<h3>Estimates behind each piece\'s setting</h3><p class="note">The level estimates that apply to each piece, for the head chosen under every estimate by level (' +
      esc(S.head === 'all' ? 'all heads' : headLabel(S.head)) + ').</p>';
    pieces.forEach(function (n) {
      var rows = effNodes((eff.pieces || {})[n.id]);
      if (!rows.length) return;
      any = true;
      block += '<h3>' + esc(n.id) + '</h3>' + nodeRows(rows, {showHead: S.head === 'all', axes: axes});
    });
    var topo = effNodes(eff.workflow);
    if (topo.length) { any = true; block += '<h3>Workflow shape</h3>' + nodeRows(topo, {showHead: S.head === 'all', axes: axes}); }
    if (any) s += block;
    return s;
  }

  // ---------------------------------------------------------------- data behind the fit
  var COL = {reason: 'Reason', n: 'Rows', run: 'Run', level: 'Level', source: 'Source', head: 'Head', runs: 'Runs'};
  function colName(k) { return COL[k] || (HEADS[k] || /^(cost|success|gate|score:)/.test(k) ? headLabel(k) : k); }
  function matrix(obj, rowName, rowLabel) {
    var cols = Object.keys(obj || {}), rows = [];
    cols.forEach(function (c) { Object.keys(obj[c] || {}).forEach(function (r) { if (rows.indexOf(r) < 0) rows.push(r); }); });
    if (!cols.length) return '<p class="empty">Not recorded by this fit.</p>';
    var s = '<div class="scroll"><table class="compact"><thead><tr><th>' + esc(colName(rowName)) + '</th>';
    cols.forEach(function (c) { s += '<th class="num">' + esc(colName(c)) + '</th>'; });
    s += '</tr></thead><tbody>';
    rows.forEach(function (r) {
      s += '<tr><td>' + esc(rowLabel ? rowLabel(r) : r) + '</td>';
      cols.forEach(function (c) { var v = (obj[c] || {})[r]; s += '<td class="num">' + esc(v == null ? '' : (typeof v === 'number' ? num(v, 3) : generic(v, true))) + '</td>'; });
      s += '</tr>';
    });
    return s + '</tbody></table></div>';
  }
  function bySource(rows) {  // {head: {source: n}} to {source: {head: n}}: a few sources across, one head per row
    var out = {};
    Object.keys(rows || {}).forEach(function (h) {
      Object.keys(rows[h] || {}).forEach(function (src) { (out[src] = out[src] || {})[h] = rows[h][src]; });
    });
    return out;
  }
  function generic(v, flat) {
    if (v == null) return '';
    if (typeof v !== 'object') return typeof v === 'number' ? num(v, 3) : String(v);
    if (flat) return JSON.stringify(v);
    if (Array.isArray(v)) {
      if (!v.length) return '<p class="empty">None.</p>';
      var keys = [];
      v.forEach(function (x) { if (x && typeof x === 'object') Object.keys(x).forEach(function (k) { if (keys.indexOf(k) < 0) keys.push(k); }); });
      if (!keys.length) return '<p>' + esc(v.join(', ')) + '</p>';
      var numeric = {};
      keys.forEach(function (k) { numeric[k] = v.every(function (x) { var y = (x || {})[k]; return y == null || typeof y === 'number'; }); });
      var s = '<div class="scroll"><table class="compact"><thead><tr>';
      keys.forEach(function (k) { s += '<th' + (numeric[k] ? ' class="num"' : '') + '>' + esc(colName(k)) + '</th>'; });
      s += '</tr></thead><tbody>';
      v.forEach(function (x) {
        s += '<tr>';
        keys.forEach(function (k) { s += '<td' + (numeric[k] ? ' class="num"' : '') + '>' + esc(generic((x || {})[k], true)) + '</td>'; });
        s += '</tr>';
      });
      return s + '</tbody></table></div>';
    }
    var t = '<table class="kv"><tbody>';
    Object.keys(v).forEach(function (k) {
      var x = v[k];
      t += '<tr><td>' + esc(k) + '</td><td>' + (x && typeof x === 'object' ? generic(x) : esc(generic(x))) + '</td></tr>';
    });
    return t + '</tbody></table>';
  }
  function dataHtml() {
    var d = D.data || {}, fit = D.fit || {}, n = fit.n_runs, runs;
    if (n && typeof n === 'object') runs = Object.keys(n).map(function (k) { return k + ' ' + n[k]; }).join(', ');
    else runs = n == null ? 'not recorded' : String(n);
    var s = '<h2>The fit</h2><table class="kv"><tbody>' +
      '<tr><td>Fit</td><td><code>' + esc(fit.id) + '</code></td></tr><tr><td>Fitted at</td><td>' + esc(fit.at) + '</td></tr>' +
      '<tr><td>Runs</td><td>' + esc(runs) + '</td></tr>' +
      '<tr><td>Time to fit</td><td>' + esc(fin(d.fit_time_s) ? num(d.fit_time_s, 1) + ' s' : 'not recorded') + '</td></tr>' +
      (d.code_version ? '<tr><td>Code version</td><td>' + esc(d.code_version) + '</td></tr>' : '') +
      '<tr><td>Page generated</td><td>' + esc(D.generated_at) + '</td></tr></tbody></table>';
    s += '<h2>Rows per source and head</h2><p class="note">How many rows each data source gave each head. The prior is data too: ' +
      'our sweep, E0, RQ1 and benchmark rows enter as their own sources.</p>' + matrix(bySource(d.rows), 'head', headLabel);
    var rbs = d.runs_by_source || {};
    if (Object.keys(rbs).length) s += '<h3>Runs per source</h3>' + matrix({runs: rbs}, 'source');
    s += '<h2>Dropped rows</h2>' + (d.dropped && d.dropped.length ? generic(d.dropped) : '<p class="empty">No rows were dropped.</p>');
    s += '<h2>Scales per level (phi)</h2><p class="note">The empirical Bayes scale of each level: how far a child node may move from its parent. ' +
      'Larger means the data showed more spread at that level.</p>' + matrix(d.scales, 'level', levelWord);
    s += '<h2>Sensitivity to the benchmark prior</h2>' + ((d.without || []).length ? '<p class="note">This fit left out: ' +
      esc(d.without.join(', ')) + '. Compare it with a fit on everything to see what those sources move.</p>' : '') + (d.sensitivity ? generic(d.sensitivity)
      : '<p class="empty">No comparison yet. Run <code>loopmath fit --without benchmark</code> to see how much the benchmark priors move the estimates.</p>');
    return s;
  }

  // ---------------------------------------------------------------- page
  function lede() {
    var fit = D.fit || {}, n = fit.n_runs, total = null, user = null;
    if (n && typeof n === 'object') { total = 0; Object.keys(n).forEach(function (k) { total += +n[k] || 0; }); user = +n.user || 0; }
    else if (fin(n)) total = n;
    return 'What loopmath currently estimates about how each model, effort level, role, workflow shape, task type and repo changes cost, ' +
      'the chance of success and the chance a gate passes. Every estimate has an 80% range and the number of runs it rests on. ' +
      'These are the posterior estimates of fit ' + esc(fit.id) + ' (' + esc(fit.at) + ')' +
      (total != null ? ', from ' + total.toLocaleString('en-US') + ' runs' + (user != null ? ', ' + user.toLocaleString('en-US') + ' of them yours' : '') : '') + '.';
  }
  function taskLine() {
    var t = D.task;
    if (!t) return '';
    var why = {arguments: 'from the command line', store: 'the most common type and repo in your runs',
      'default': 'no task given and no runs yet: a task in a repo with no runs'}[t.from] || t.from;
    var feats = Object.keys(t.features || {}).map(function (k) { return k + '=' + t.features[k]; }).join(', ');
    return 'The workflows are computed for a <b>' + esc(t.type) + '</b> task in <b>' + esc(t.repo) + '</b>' + (t.subtype ? ' (' + esc(t.subtype) + ')' : '') +
      (feats ? ' with ' + esc(feats) : '') + ' (' + esc(why) + '). For another task: <code>loopmath posterior --type T --repo R --html</code>.';
  }
  function host(id) { return document.getElementById(id); }
  function drawGraph() { var g = host('est-graph'); if (g) g.innerHTML = graphHtml(); }
  function draw() {
    var l = host('est-levels'), d = host('est-data');
    if (l) l.innerHTML = levelsHtml();
    drawGraph();
    if (d) d.innerHTML = dataHtml();
  }
  var started = false;
  function init() {  // results.js calls this once the hosts are on the page
    if (started) return;
    started = true;
    var hs = headList();
    S.head = D.head && hs.indexOf(D.head) >= 0 ? D.head : (hs[0] || 'all');
    draw();
    document.addEventListener('click', function (ev) {
      var sum = ev.target && ev.target.closest ? ev.target.closest('summary[data-group]') : null;
      if (sum && sum.getAttribute('data-group') != null) { S.open[sum.getAttribute('data-group')] = !(sum.parentNode && sum.parentNode.open); return; }  // before it toggles
      var t = ev.target && ev.target.closest ? ev.target.closest('[data-head],a[data-wf]') : null;
      if (!t) return;
      if (t.getAttribute('data-wf') != null) {
        ev.preventDefault();
        S.wf = +t.getAttribute('data-wf') || 0;
        drawGraph();
        return;
      }
      S.head = t.getAttribute('data-head');
      draw();
    });
    document.addEventListener('change', function (ev) {
      if (ev.target && ev.target.id === 'wfpick') { S.wf = +ev.target.value || 0; drawGraph(); }
    });
    var tip = host('tip');
    if (!tip) { tip = document.createElement('div'); tip.id = 'tip'; tip.className = 'tip est-tip'; tip.hidden = true; document.body.appendChild(tip); }
    document.addEventListener('mousemove', function (ev) {
      var t = ev.target && ev.target.closest ? ev.target.closest('[data-tip]') : null;
      if (!t) { tip.hidden = true; return; }
      tip.textContent = t.getAttribute('data-tip');
      tip.hidden = false;
      var x = ev.clientX + 14, y = ev.clientY + 14;
      if (x + 370 > window.innerWidth) x = Math.max(4, ev.clientX - 370);
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
    });
  }
  function showWorkflow(i) { S.wf = i; drawGraph(); }
  window.LMPosterior = {levelsHtml: levelsHtml, graphHtml: graphHtml, dataHtml: dataHtml, layout: layout,
    fmtDisplay: fmtDisplay, heads: headList, state: S, init: init, lede: lede, taskLine: taskLine, showWorkflow: showWorkflow};
})();
