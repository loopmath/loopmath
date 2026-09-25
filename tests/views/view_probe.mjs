#!/usr/bin/env node
// Headless check of one view page (lane 12): load it in headless Chrome with name resolution
// disabled, run an optional script in the page, and print one JSON object:
// {load_ms, exceptions, console_errors, network, result}.
// usage: node tests/views/view_probe.mjs PAGE.html [SCRIPT.js]
// The script is an expression whose value (awaited) becomes `result`.

import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const CHROME = process.env.LOOPMATH_PROBE_CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const WIDTH = Number(process.env.LOOPMATH_PROBE_WIDTH) || 1366;  // 820 or 1180 for an iPad-sized viewport
const [input, scriptFile] = process.argv.slice(2);
if (!input) { process.stderr.write('usage: node view_probe.mjs PAGE.html [SCRIPT.js]\n'); process.exit(2); }
const script = scriptFile ? await readFile(scriptFile, 'utf8') : null;
const profile = await mkdtemp(join(tmpdir(), 'lm-view-probe-'));
const chrome = spawn(CHROME, ['--headless=new', '--disable-gpu', '--disable-background-networking', '--disable-component-update',
  '--disable-default-apps', '--disable-sync', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0',
  // A mock keychain: a fresh profile must not ask macOS for a keychain (a sandboxed caller has none,
  // and each probe then raises a system dialog).
  '--use-mock-keychain', '--password-store=basic',
  '--host-resolver-rules=MAP * ~NOTFOUND', `--user-data-dir=${profile}`, 'about:blank'], { stdio: ['ignore', 'ignore', 'ignore'] });
const delay = ms => new Promise(done => setTimeout(done, ms));
const exited = new Promise(done => chrome.once('exit', done));
// Stop Chrome and wait until it has gone: SIGTERM, then SIGKILL after 3 s.
const stopChrome = async () => {
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
  await call('Emulation.setDeviceMetricsOverride', { width: WIDTH, height: 900, deviceScaleFactor: 1, mobile: false }, sessionId);
  const loaded = new Promise(ok => on('Page.loadEventFired', m => { if (m.sessionId === sessionId) ok(); }));
  const t0 = Date.now();
  await call('Page.navigate', { url: pathToFileURL(resolve(input)).href }, sessionId);
  await loaded;
  const load_ms = Date.now() - t0;
  await delay(100);
  let result = null;
  if (script) {
    const answer = await call('Runtime.evaluate', { expression: script, returnByValue: true, awaitPromise: true }, sessionId);
    if (answer.exceptionDetails) exceptions.push('probe script: ' + (answer.exceptionDetails.exception?.description || answer.exceptionDetails.text));
    else result = answer.result.value;
  }
  await delay(50);
  if (process.env.LOOPMATH_PROBE_SHOT) {  // optional full-page PNG, for a person to look at
    const { writeFile } = await import('node:fs/promises');
    const h = await call('Runtime.evaluate', { expression: 'Math.min(6000, document.documentElement.scrollHeight)', returnByValue: true }, sessionId);
    await call('Emulation.setDeviceMetricsOverride', { width: WIDTH, height: h.result.value, deviceScaleFactor: 1, mobile: false }, sessionId);
    const shot = await call('Page.captureScreenshot', { format: 'png' }, sessionId);
    await writeFile(process.env.LOOPMATH_PROBE_SHOT, Buffer.from(shot.data, 'base64'));
  }
  process.stdout.write(JSON.stringify({ load_ms, exceptions, console_errors: consoleErrors, network, result }) + '\n');
} catch (err) {
  process.stderr.write(String(err && err.stack || err) + '\n');
  code = 1;
} finally {
  try { socket && socket.close(); } catch {}
  await stopChrome();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exit(code);
