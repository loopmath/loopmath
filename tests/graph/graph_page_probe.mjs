#!/usr/bin/env node
// Headless check of one `graph --format html` page: load it in headless Chrome with name
// resolution disabled at a given width, switch every view to color by model, and print one JSON
// object: {chrome_pid, parent_pid, exceptions, console_errors, network, result}.
// usage: node tests/graph/graph_page_probe.mjs PAGE.html [--width 1440] [--shot FILE.png] [--timezone America/Los_Angeles]
// Chrome runs with a temporary profile and a mock keychain; it is stopped by its own pid.

import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const CHROME = process.env.LOOPMATH_PROBE_CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const args = process.argv.slice(2);
const option = name => { const at = args.indexOf(name); return at >= 0 ? args[at + 1] : null; };
const input = args[0];
const width = Number(option('--width')) || 1440;
const shot = option('--shot');
const timezone = option('--timezone');
if (!input || input.startsWith('--')) { process.stderr.write('usage: node graph_page_probe.mjs PAGE.html [--width N] [--shot FILE]\n'); process.exit(2); }

// Runs in the page after load: color every view by model, then measure.
const CHECK = `(async () => {
  const grey = 'rgb(137, 135, 129)', greyHex = '#898781';
  document.querySelectorAll('[data-view-color]').forEach(select => { select.value = 'model'; select.dispatchEvent(new Event('change', { bubbles: true })); });
  await new Promise(done => setTimeout(done, 80));
  const swim = document.querySelector('[data-view="swim"]'), svg = swim.querySelector('[data-graph] svg');
  const lanes = [...svg.querySelectorAll('rect.lanebg')];
  const plotLeft = Math.min(...lanes.map(r => +r.getAttribute('x'))), plotRight = Math.max(...lanes.map(r => +r.getAttribute('x') + +r.getAttribute('width')));
  const boxes = [...svg.querySelectorAll('rect.node, rect.hit')].map(r => ({ i: +r.dataset.node, cls: r.getAttribute('class'), left: +r.getAttribute('x'), right: +r.getAttribute('x') + +r.getAttribute('width'), fill: r.getAttribute('fill') }));
  const bars = boxes.filter(b => b.cls.startsWith('node'));
  const legend = view => [...document.querySelectorAll('[data-view="' + view + '"] [data-legend] .it')].filter(it => it.querySelector('.sw:not(.sq)')).map(it => ({ label: it.textContent.trim(), color: getComputedStyle(it.querySelector('.sw')).backgroundColor }));
  const models = [...new Set(DATA.nodes.map(n => n.model || 'unknown'))];
  const accounting = document.getElementById('accounting');
  const shown = document.body.innerText;
  const counters = Object.keys(DATA.run.accounting || {}).map(key => key.replaceAll('_', ' ')).filter(words => shown.includes(words + ':'));
  const overflow = [...document.querySelectorAll('[data-wrap], .tablescroll')].map(w => w.scrollWidth - w.clientWidth);
  const textPastEdge = [...document.querySelectorAll('[data-graph] svg')].flatMap(root => { const w = +root.getAttribute('width'); return [...root.querySelectorAll('text')].filter(t => { const box = t.getBBox(); return box.width && (box.x < -0.5 || box.x + box.width > w + 0.5); }).map(t => t.closest('[data-view]').dataset.view + ': ' + t.textContent); });
  const rows = [...document.querySelectorAll('#table tbody tr')].map(tr => [...tr.cells].map(td => td.innerText.trim()));
  return {
    models,
    bars: bars.map(b => ({ name: DATA.nodes[b.i].lbl, model: DATA.nodes[b.i].model || 'unknown', fill: b.fill, left: b.left, right: b.right })),
    plot: { left: plotLeft, right: plotRight, svgWidth: +svg.getAttribute('width') },
    pastEdge: boxes.filter(b => b.left < plotLeft - 0.01 || b.right > plotRight + 0.01).map(b => ({ name: DATA.nodes[b.i].lbl, left: b.left, right: b.right })),
    textPastEdge,
    greyBarsWithAModel: bars.filter(b => DATA.nodes[b.i].model && (b.fill || '').toLowerCase() === greyHex).map(b => DATA.nodes[b.i].lbl),
    legend: { swim: legend('swim'), force: legend('force'), cost: legend('cost') },
    greyLegendModels: ['swim', 'force', 'cost'].flatMap(view => legend(view).filter(item => item.label !== 'unknown' && item.color === grey).map(item => view + ':' + item.label)),
    pageOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    graphOverflow: overflow,
    accounting: { tag: accounting.tagName.toLowerCase(), open: accounting.open, summary: accounting.querySelector('summary')?.textContent || null },
    countersVisible: counters,
    laneNames: [...svg.querySelectorAll('text.lanelbl')].map(t => t.textContent),
    table: rows.map(row => ({ attempt: row[0], role: row[1], model: row[2], cost: row[4], tokens: row[5] })),
    times: { foot: document.getElementById('foot').innerText, titles: [...document.querySelectorAll('[data-graph-title]')].map(t => t.textContent),
             ticks: [...document.querySelectorAll('[data-graph] svg text.axlbl')].map(t => t.textContent).filter(t => t.includes('  +')) },  // clock time, two spaces, time into the run
  };
})()`;

const profile = await mkdtemp(join(tmpdir(), 'lm-graph-probe-'));
const chrome = spawn(CHROME, ['--headless=new', '--use-mock-keychain', '--password-store=basic', `--user-data-dir=${profile}`,
  '--disable-gpu', '--disable-background-networking', '--disable-component-update', '--disable-default-apps', '--disable-sync',
  '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', '--host-resolver-rules=MAP * ~NOTFOUND',
  'about:blank'], { stdio: ['ignore', 'ignore', 'ignore'] });
process.stderr.write(`chrome pid ${chrome.pid}, parent ${process.pid}\n`);
const delay = ms => new Promise(done => setTimeout(done, ms));
const exited = new Promise(done => chrome.once('exit', done));
const stopChrome = async () => {  // by its own pid only: SIGTERM, then SIGKILL after 3 s
  if (chrome.exitCode !== null || chrome.signalCode !== null) return;
  chrome.kill('SIGTERM');
  if (await Promise.race([exited.then(() => true), delay(3000).then(() => false)])) return;
  chrome.kill('SIGKILL');
  await Promise.race([exited, delay(2000)]);
};
for (const sig of ['SIGTERM', 'SIGINT', 'SIGHUP']) process.once(sig, () => { chrome.kill('SIGKILL'); process.exit(1); });

let socket, nextId = 0;
const pending = new Map(), listeners = new Map();
const on = (method, fn) => { if (!listeners.has(method)) listeners.set(method, []); listeners.get(method).push(fn); };
const call = (method, params = {}, sessionId) => {
  const id = ++nextId;
  socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  return new Promise((ok, bad) => pending.set(id, { ok, bad }));
};

let code = 0;
try {
  let port = null, path = null;
  for (let i = 0; i < 200 && !port; i++) {
    try { [port, path] = (await readFile(profile + '/DevToolsActivePort', 'utf8')).trim().split('\n'); } catch { await delay(50); }
  }
  if (!port) throw new Error('Chrome did not publish its DevTools endpoint');
  socket = new WebSocket(`ws://127.0.0.1:${port}${path}`);
  await new Promise((ok, bad) => { socket.onopen = ok; socket.onerror = () => bad(new Error('DevTools socket failed')); });
  socket.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) { const p = pending.get(msg.id); pending.delete(msg.id); if (msg.error) p.bad(new Error(msg.error.message)); else p.ok(msg.result); return; }
    (listeners.get(msg.method) || []).forEach(fn => fn(msg));
  };
  const target = await call('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await call('Target.attachToTarget', { targetId: target.targetId, flatten: true });
  const exceptions = [], consoleErrors = [], network = [];
  on('Runtime.exceptionThrown', m => { if (m.sessionId === sessionId) exceptions.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text); });
  on('Runtime.consoleAPICalled', m => { if (m.sessionId === sessionId && m.params.type === 'error') consoleErrors.push(m.params.args.map(a => a.value ?? a.description).join(' ')); });
  on('Log.entryAdded', m => { if (m.sessionId === sessionId && m.params.entry.level === 'error') consoleErrors.push(m.params.entry.text); });
  on('Network.requestWillBeSent', m => { if (m.sessionId === sessionId && !/^(file|data|about|blob):/i.test(m.params.request.url)) network.push(m.params.request.url); });
  for (const domain of ['Runtime', 'Log', 'Network', 'Page']) await call(`${domain}.enable`, {}, sessionId);
  await call('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile: false }, sessionId);
  if (timezone) await call('Emulation.setTimezoneOverride', { timezoneId: timezone }, sessionId);
  const loaded = new Promise(ok => on('Page.loadEventFired', m => { if (m.sessionId === sessionId) ok(); }));
  await call('Page.navigate', { url: pathToFileURL(resolve(input)).href }, sessionId);
  await loaded;
  await delay(250);  // the resize handler re-renders 150 ms after the metrics change
  const answer = await call('Runtime.evaluate', { expression: CHECK, returnByValue: true, awaitPromise: true }, sessionId);
  let result = null;
  if (answer.exceptionDetails) exceptions.push('probe script: ' + (answer.exceptionDetails.exception?.description || answer.exceptionDetails.text));
  else result = answer.result.value;
  if (shot) {
    const h = await call('Runtime.evaluate', { expression: 'Math.min(8000, document.documentElement.scrollHeight)', returnByValue: true }, sessionId);
    await call('Emulation.setDeviceMetricsOverride', { width, height: h.result.value, deviceScaleFactor: 1, mobile: false }, sessionId);
    await delay(250);
    const image = await call('Page.captureScreenshot', { format: 'png' }, sessionId);
    await writeFile(shot, Buffer.from(image.data, 'base64'));
  }
  process.stdout.write(JSON.stringify({ chrome_pid: chrome.pid, parent_pid: process.pid, exceptions, console_errors: consoleErrors, network, result }) + '\n');
} catch (err) {
  process.stderr.write(String(err && err.stack || err) + '\n');
  code = 1;
} finally {
  try { socket && socket.close(); } catch {}
  await stopChrome();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exit(code);
