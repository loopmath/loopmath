// A walk of the builder page in headless Chrome over the DevTools protocol, for test_builder_walk.py (opt-in).
// It answers /api/predict_many with a 404, as a 0.2.1 server would, so the page falls back to one /api/predict
// per configuration; then it drags a piece, taps a point on the chart for its card, and copies the build.
// usage: node walk_builder.mjs <chrome> <page url> <profile dir>
// It prints the Chrome pid and its parent, stops that pid only, and prints one JSON line last.
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const [chromePath, url, profile] = process.argv.slice(2);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const chrome = spawn(chromePath, ['--headless=new', '--use-mock-keychain', `--user-data-dir=${profile}`,
  '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0', '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--disable-component-update', '--disable-sync', '--disable-extensions', 'about:blank'],
{ stdio: 'ignore' });
console.log(`chrome pid ${chrome.pid} parent ${process.pid}`);
const out = { steps: {}, requests: [], api: [], hosts: [], exceptions: [], console: [] };
let ws = null;
const stop = async () => {
  if (ws) ws.close();
  console.log(`stopping chrome pid ${chrome.pid}`);
  try { process.kill(chrome.pid, 'SIGTERM'); } catch (e) { /* gone */ }
  for (let i = 0; i < 20; i++) { await sleep(100); try { process.kill(chrome.pid, 0); } catch (e) { return; } }
  try { process.kill(chrome.pid, 'SIGKILL'); } catch (e) { /* gone */ }
};
try {
  let port = null;
  for (let i = 0; i < 150 && !port; i++) {
    await sleep(100);
    const f = join(profile, 'DevToolsActivePort');
    if (existsSync(f)) port = readFileSync(f, 'utf8').split('\n')[0];
  }
  if (!port) throw new Error('Chrome gave no DevTools port');
  const ver = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  ws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener('open', r, { once: true }));
  let seq = 0; const waiting = new Map(), handlers = [];
  ws.addEventListener('message', ev => {
    const m = JSON.parse(ev.data);
    if (m.id && waiting.has(m.id)) { const w = waiting.get(m.id); waiting.delete(m.id); m.error ? w.rej(new Error(JSON.stringify(m.error))) : w.res(m.result); }
    else handlers.forEach(h => h(m));
  });
  const send = (method, params = {}, sessionId) => new Promise((res, rej) => { const id = ++seq; waiting.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params, sessionId })); });
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  const S = (m, p) => send(m, p, sessionId);
  handlers.push(m => {
    if (m.sessionId !== sessionId) return;
    const p = m.params || {};
    if (m.method === 'Network.requestWillBeSent') {
      out.requests.push(p.request.url);
      if (/\/api\//.test(p.request.url)) out.api.push(p.request.method + ' ' + new URL(p.request.url).pathname);
    }
    if (m.method === 'Fetch.requestPaused') {
      S('Fetch.fulfillRequest', { requestId: p.requestId, responseCode: 404, responseHeaders: [{ name: 'content-type', value: 'application/json' }],
        body: Buffer.from(JSON.stringify({ ok: false, errors: [{ piece: null, message: 'no such path: /api/predict_many' }] })).toString('base64') });
    }
    if (m.method === 'Runtime.exceptionThrown') out.exceptions.push(JSON.stringify(p.exceptionDetails).slice(0, 600));
    if (m.method === 'Runtime.consoleAPICalled' && p.type === 'error') out.console.push(p.args.map(a => a.value || a.description).join(' '));
    if (m.method === 'Log.entryAdded' && p.entry.level === 'error' && !/predict_many/.test(p.entry.url || '')) out.console.push(p.entry.text);
  });
  await S('Runtime.enable'); await S('Network.enable'); await S('Log.enable'); await S('Page.enable');
  await S('Fetch.enable', { patterns: [{ urlPattern: '*/api/predict_many*', requestStage: 'Request' }] });
  await S('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  const ev = async expr => { const r = await S('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error('eval: ' + JSON.stringify(r.exceptionDetails).slice(0, 400)); return r.result.value; };
  const until = async (expr, ms = 20000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { try { if (await ev(expr)) return true; } catch (e) { /* not yet */ } await sleep(100); } throw new Error('timed out waiting for ' + expr); };
  const mouse = async (type, x, y) => S('Input.dispatchMouseEvent', { type, x, y, button: 'left', buttons: type === 'mouseReleased' ? 0 : 1, clickCount: 1 });
  const centerOf = async expr => ev(`(() => { const e = ${expr}; if (!e) return null; e.scrollIntoView({block: 'center'}); const r = e.getBoundingClientRect(); return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })()`);
  const click = async expr => {
    const p = await centerOf(expr); if (!p) throw new Error('nothing at ' + expr);
    await sleep(100);
    await S('Input.dispatchMouseEvent', { type: 'mouseMoved', x: p.x, y: p.y });
    await mouse('mousePressed', p.x, p.y); await mouse('mouseReleased', p.x, p.y);
    await sleep(250);
  };
  const lastEdit = `(document.querySelector('.lastedit') || {}).textContent || ''`;

  // 1. load: predict_many answers 404, so the numbers come from one /api/predict per configuration
  await S('Page.navigate', { url });
  await until(`document.querySelectorAll('[data-big]').length >= 3 && document.querySelectorAll('.nudge').length > 0`, 45000);
  await sleep(500);
  out.steps.loaded = {
    big: await ev(`[...document.querySelectorAll('[data-big]')].map(b => b.textContent).join(' | ')`),
    mode: await ev(`(document.getElementById('mode') || {}).textContent || ''`),
    many: out.api.filter(a => a.endsWith('/api/predict_many')).length,
    single: out.api.filter(a => a.endsWith('/api/predict')).length,
  };
  // the run cost metric's title and mean line, and the words the chart puts on the build's run cost and its axis
  out.steps.chart = await ev(`(() => { const t = [...document.querySelectorAll('.metric .ml')].map(e => e.textContent).find(x => /run cost/i.test(x)) || null;
    const m = document.getElementById('runmean'), ax = [...document.querySelectorAll('#chart .c-axt')].map(e => e.textContent).find(x => /log scale/.test(x)) || null;
    const you = document.querySelector('#bl text'); return { title: t, meanLine: m ? m.textContent : null, axis: ax, label: you ? you.textContent : null }; })()`);

  // 2. drag a piece on the canvas: it moves, and the last edit says so
  const node = await ev(`[...document.querySelectorAll('#cv [data-node]')].map(g => g.dataset.node)[0]`);
  const from = await centerOf(`document.querySelector('#cv [data-node="${node}"] .n-card')`);
  const cardAt = `(() => { const r = document.querySelector('#cv [data-node="${node}"] .n-card'); return {x: +r.getAttribute('x'), y: +r.getAttribute('y')}; })()`;  // canvas units: the canvas re-fits around the graph after a move
  const fromIn = await ev(cardAt);
  await S('Input.dispatchMouseEvent', { type: 'mouseMoved', x: from.x, y: from.y });
  await mouse('mousePressed', from.x, from.y);
  for (let i = 1; i <= 8; i++) { await S('Input.dispatchMouseEvent', { type: 'mouseMoved', x: from.x + 7 * i, y: from.y + 4 * i, button: 'left', buttons: 1 }); await sleep(30); }
  await mouse('mouseReleased', from.x + 56, from.y + 32);
  await until(`/Moved/.test(${lastEdit})`, 5000);
  const to = await ev(cardAt);
  out.steps.drag = { node, last: await ev(lastEdit), from: fromIn, to, moved: Math.hypot(to.x - fromIn.x, to.y - fromIn.y) > 20 };

  // 3. tap a candidate's point on the chart: its card opens with its numbers
  await ev(`(() => { const e = [...document.querySelectorAll('#chart [data-c]')].find(c => { c.scrollIntoView({block: 'center'}); const r = c.getBoundingClientRect(); return document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2) === c; }); if (e) e.id = 'walk-point'; return !!e; })()`);
  await click(`document.getElementById('walk-point')`);
  await until(`!document.getElementById('tapcard').classList.contains('empty')`, 5000);
  out.steps.tap = { card: await ev(`document.getElementById('tapcard').innerText`), start: await ev(`!!document.getElementById('tapstart')`) };
  await click(`document.getElementById('tapclose')`);

  // 4. Copy build: the line for the agent goes to the clipboard and the page says so
  await ev(`(() => { window.__copied = null; Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: t => { window.__copied = t; return Promise.resolve(); } } }); return true; })()`);
  await until(`!!document.getElementById('copybtn') && !document.getElementById('copybtn').disabled`, 10000);
  await click(`document.getElementById('copybtn')`);
  await until(`!!window.__copied`, 5000);
  out.steps.copy = { text: await ev('window.__copied'), toast: await ev(`document.getElementById('toast').textContent`) };
  await send('Target.closeTarget', { targetId });
} catch (e) {
  out.error = String(e && e.message || e);
}
out.hosts = [...new Set(out.requests.map(u => { try { const x = new URL(u); return /^https?:$/.test(x.protocol) ? x.hostname : x.protocol; } catch (e) { return u; } }))];
delete out.requests;
await stop();
console.log(JSON.stringify(out));
process.exit(0);
