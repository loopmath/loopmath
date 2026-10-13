#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { attemptAuditSource } from './graph_html_probe_common.mjs';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const EXPECTED_COLUMNS = ['attempt', 'role', 'model', 'tier', 'cost', 'tokens', 'duration', 'status', 'artifacts written', 'artifacts read'];
const input = process.argv[2];
const smoke = process.argv.includes('--smoke');
const attemptOnly = process.argv.includes('--attempt');
const sourceAt = process.argv.indexOf('--source');
const sourceFile = sourceAt >= 0 ? process.argv[sourceAt + 1] : null;
if (!input || (!smoke && !sourceFile)) {
  process.stderr.write('usage: node tests/graph_html_browser_probe.mjs FILE [--smoke] --source SOURCE_JSON\n');
  process.exit(2);
}
const sourceOracle = sourceFile ? JSON.parse(await readFile(sourceFile, 'utf8')) : null;
const attemptAudit = sourceOracle ? attemptAuditSource(sourceOracle) : null;

const outDir = resolve('out');
await mkdir(outDir, { recursive: true });
const profile = await mkdtemp(outDir + '/chrome-probe-');
const chrome = spawn(CHROME, [
  '--headless=new',
  '--disable-gpu',
  '--disable-background-networking',
  '--disable-component-update',
  '--disable-default-apps',
  '--disable-sync',
  '--metrics-recording-only',
  '--no-first-run',
  '--no-default-browser-check',
  '--remote-debugging-port=0',
  '--host-resolver-rules=MAP * ~NOTFOUND',
  `--user-data-dir=${profile}`,
  'about:blank',
], { stdio: ['ignore', 'ignore', 'ignore'] });

const delay = ms => new Promise(resolveDelay => setTimeout(resolveDelay, ms));
async function activePort() {
  const path = profile + '/DevToolsActivePort';
  for (let attempt = 0; attempt < 200; attempt++) {
    try {
      const [port, browserPath] = (await readFile(path, 'utf8')).trim().split('\n');
      if (port && browserPath) return { port, browserPath };
    } catch {}
    await delay(50);
  }
  throw new Error('Chrome did not publish its DevTools endpoint');
}

let socket;
let nextId = 0;
const pending = new Map();
const listeners = new Map();
function on(method, listener) {
  if (!listeners.has(method)) listeners.set(method, []);
  listeners.get(method).push(listener);
}
function call(method, params = {}, sessionId = undefined) {
  const id = ++nextId;
  socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  return new Promise((resolveCall, rejectCall) => pending.set(id, { resolveCall, rejectCall }));
}
async function evaluate(sessionId, expression) {
  const answer = await call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }, sessionId);
  if (answer.exceptionDetails) throw new Error(answer.exceptionDetails.text || 'evaluation failed');
  return answer.result.value;
}

let failed = 0;
let passed = 0;
function check(name, condition, detail) {
  if (condition) {
    passed++;
    process.stdout.write(`PASS ${name}${detail ? ': ' + detail : ''}\n`);
  } else {
    failed++;
    process.stdout.write(`FAIL ${name}${detail ? ': ' + detail : ''}\n`);
  }
}

try {
  const endpoint = await activePort();
  socket = new WebSocket(`ws://127.0.0.1:${endpoint.port}${endpoint.browserPath}`);
  await new Promise((resolveSocket, rejectSocket) => {
    socket.onopen = resolveSocket;
    socket.onerror = () => rejectSocket(new Error('DevTools socket failed'));
  });
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const item = pending.get(message.id); pending.delete(message.id);
      if (message.error) item.rejectCall(new Error(message.error.message)); else item.resolveCall(message.result);
      return;
    }
    (listeners.get(message.method) || []).forEach(listener => listener(message));
  };

  const target = await call('Target.createTarget', { url: 'about:blank' });
  const attached = await call('Target.attachToTarget', { targetId: target.targetId, flatten: true });
  const session = attached.sessionId;
  const consoleErrors = [];
  const exceptions = [];
  const network = [];
  on('Runtime.consoleAPICalled', message => {
    if (message.sessionId === session && message.params.type === 'error') consoleErrors.push(message.params);
  });
  on('Runtime.exceptionThrown', message => {
    if (message.sessionId === session) exceptions.push(message.params);
  });
  on('Log.entryAdded', message => {
    if (message.sessionId === session && message.params.entry.level === 'error') consoleErrors.push(message.params.entry);
  });
  on('Network.requestWillBeSent', message => {
    if (message.sessionId === session && /^https?:/i.test(message.params.request.url)) network.push(message.params.request.url);
  });
  await call('Runtime.enable', {}, session);
  await call('Log.enable', {}, session);
  await call('Network.enable', {}, session);
  await call('Page.enable', {}, session);
  await call('Emulation.setDeviceMetricsOverride', {
    width: 1366,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  }, session);
  const loaded = new Promise(resolveLoad => on('Page.loadEventFired', message => {
    if (message.sessionId === session) resolveLoad();
  }));
  await call('Page.navigate', { url: pathToFileURL(resolve(input)).href }, session);
  await loaded;
  await evaluate(session, 'new Promise(done => setTimeout(done, 120))');

  check('console errors', consoleErrors.length === 0, String(consoleErrors.length));
  check('uncaught exceptions', exceptions.length === 0, exceptions.length ? exceptions.map(item => item.exceptionDetails.exception?.description || item.exceptionDetails.text).join(' | ') : '0');
  check('external network requests', network.length === 0, String(network.length));

  const structure = await evaluate(session, `(() => {
    const table = document.querySelector('[data-attempt-table]');
    const views = [...document.querySelectorAll('details[data-view]')];
    return {
      headers: [...document.querySelectorAll('#table th')].map(el => el.textContent.trim().replace(/[▲▼]/g, '')),
      views: views.map(el => el.dataset.view),
      open: views.map(el => el.open),
      tableFirst: table.compareDocumentPosition(views[0]) === Node.DOCUMENT_POSITION_FOLLOWING,
    };
  })()`);
  check('ten table columns', JSON.stringify(structure.headers) === JSON.stringify(EXPECTED_COLUMNS), structure.headers.join(', '));
  check('table before pictures', structure.tableFirst, String(structure.tableFirst));
  check('view order', JSON.stringify(structure.views) === JSON.stringify(['swim', 'force', 'cost']), structure.views.join(', '));
  check('views start unfolded', structure.open.length === 3 && structure.open.every(Boolean), structure.open.join(', '));

  if (attemptOnly) {
    const audit = await evaluate(session, attemptAudit);
    check('attempt source payload and rendered card', audit.errorCount === 0, audit.errorCount ? audit.errors.join('; ') : `${audit.nodes} attempts, ${audit.completeTokens} complete and ${audit.incompleteTokens} incomplete token sets, ${audit.omissions.length} omissions, ${audit.forceConnections} source-checked force connections`);
  }

  if (!smoke && !attemptOnly) {
  const defaults = await evaluate(session, `(() => {
    const count = (view, selector) => document.querySelectorAll('[data-view="' + view + '"] ' + selector).length;
    const artifact = document.querySelector('[data-view="force"] [data-art]');
    if (artifact) artifact.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    return { swim: count('swim', '[data-art]'), force: count('force', '[data-art]'), cost: count('cost', '[data-art]'), forceToggle: count('force', '[data-view-artifacts]'), clickable: !!artifact && !document.querySelector('#card').hidden && document.querySelector('#card h3').textContent.includes('artifact') };
  })()`);
  check('artifact defaults', defaults.swim === 0 && defaults.cost === 0 && defaults.force > 0 && defaults.forceToggle === 0 && defaults.clickable, JSON.stringify(defaults));

  const allSorting = await evaluate(session, `(() => {
    const result = {};
    const keys = [...document.querySelectorAll('#table th[data-key]')].map(header => header.dataset.key);
    for (const key of keys) {
      document.querySelector('#table th[data-key="' + key + '"]').click();
      const first = document.querySelector('#table th[data-key="' + key + '"] .arrow')?.textContent;
      document.querySelector('#table th[data-key="' + key + '"]').click();
      const second = document.querySelector('#table th[data-key="' + key + '"] .arrow')?.textContent;
      result[key] = first && second && first !== second;
    }
    return result;
  })()`);
  check('every column sorting control', Object.keys(allSorting).length === 10 && Object.values(allSorting).every(Boolean), JSON.stringify(allSorting));

  const sorting = await evaluate(session, `(() => {
    const values = column => [...document.querySelectorAll('#table tbody tr')].map(row => row.cells[column].textContent.trim());
    document.querySelector('th[data-key="usd"]').click();
    const money = values(4).map(value => value === 'n/a' ? -1 : Number(value.replace('$', '')));
    const moneyOk = money.every((value, i) => i === 0 || money[i - 1] >= value);
    document.querySelector('th[data-key="role"]').click();
    const roles = values(1).map(value => value.split(/verified|reported|heuristic|unknown/)[0]);
    const roleOk = roles.every((value, i) => i === 0 || roles[i - 1].localeCompare(value) <= 0);
    return { moneyOk, roleOk, firstMoney: money.slice(0, 3), firstRoles: roles.slice(0, 3) };
  })()`);
  check('cost column sorting', sorting.moneyOk, sorting.firstMoney.join(', '));
  check('text column sorting', sorting.roleOk, sorting.firstRoles.join(', '));

  const audit = await evaluate(session, attemptAudit);
  check('attempt source payload and rendered card', audit.errorCount === 0, audit.errorCount ? audit.errors.join('; ') : `${audit.nodes} attempts, ${audit.sourceFields.length} source fields, ${audit.tokenStreams.length} token streams, ${audit.omissions.length} omissions, ${audit.forceConnections} source-checked force connections`);

  const linkedCard = await evaluate(session, `(() => {
    const row = [...document.querySelectorAll('#table tbody tr')].find(item => Number(item.cells[8].textContent) + Number(item.cells[9].textContent) > 0); row.click();
    document.querySelector('#card [data-art-link]').click();
    const artifact = !document.querySelector('#card').hidden && document.querySelector('#card h3').textContent.includes('artifact');
    const node = document.querySelector('[data-view="swim"] [data-node]');
    node.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    const attempt = !document.querySelector('#card').hidden && [...document.querySelectorAll('#card dt')].some(el => el.textContent === 'attempt id');
    return { artifact, attempt };
  })()`);
  check('artifact links and node card', linkedCard.artifact && linkedCard.attempt, JSON.stringify(linkedCard));

  const tooltips = await evaluate(session, `(() => {
    const move = el => { if (!el) return false; el.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 30, clientY: 30 })); return true; };
    const tip = document.querySelector('#tip');
    const node = move(document.querySelector('[data-view="swim"] [data-node]')) && !tip.hidden && tip.textContent.length > 0;
    const edges = {};
    for (const [view, selector] of [['swim', '[data-edge]'], ['force', '.alink[data-edge-list]'], ['cost', '[data-edge]']]) {
      const edge = document.querySelector('[data-view="' + view + '"] ' + selector); edges[view] = move(edge) && !tip.hidden && tip.textContent.length > 0;
    }
    const artifactNode = document.querySelector('[data-view="force"] [data-art]');
    const artifact = move(artifactNode) && !tip.hidden && tip.textContent.length > 0;
    const forceConnections = [...document.querySelectorAll('[data-view="force"] .alink')];
    const forceData = forceConnections.every(el => el.hasAttribute('data-edge-list') && el.hasAttribute('data-connection-node') && el.hasAttribute('data-connection-kind'));
    return { node, edges, artifact, forceConnections: forceConnections.length, forceData };
  })()`);
  check('node edge artifact tooltips in every view', tooltips.node && Object.values(tooltips.edges).every(Boolean) && tooltips.artifact && tooltips.forceConnections > 0 && tooltips.forceData, JSON.stringify(tooltips));

  const missingValues = await evaluate(session, `(() => {
    document.querySelector('[data-reset]').click();
    const node = DATA.nodes.find(item => item.dur == null && !item.tokComplete) || DATA.nodes.find(item => item.untimed) || DATA.nodes[0];
    const row = document.querySelector('#table tr[data-i="' + node.i + '"]'); row.click();
    const fields = [...document.querySelectorAll('#card dt')].map(el => el.textContent.trim());
    const values = [...document.querySelectorAll('#card dd')].map(el => el.textContent.trim());
    const value = name => values[fields.indexOf(name)];
    const accounting = document.querySelector('#accounting')?.textContent || '';
    const cardTokens = value('tokens');
    return { tableDuration: row.cells[6].textContent.trim(), tableTokens: row.cells[5].textContent.trim(), cardDuration: value('duration'), cardTokens, allTokenStreamsUnavailable: typeof cardTokens === 'string' && DATA.run.token_streams.every(stream => cardTokens.includes(stream.replaceAll('_', ' ') + ' n/a')), accounting, noNegativeArtifactIndexes: DATA.artifacts.every(item => item.prod == null || item.prod >= 0) && DATA.edges.every(edge => edge[4] == null || edge[4] >= 0) };
  })()`);
  check('unknown values and viewer accounting', missingValues.tableDuration === 'n/a' && missingValues.tableTokens === 'n/a' && missingValues.cardDuration === 'n/a' && typeof missingValues.cardTokens === 'string' && missingValues.cardTokens.includes('no token record; total unavailable') && missingValues.allTokenStreamsUnavailable && missingValues.accounting.includes('attempt durations unavailable: 1') && missingValues.accounting.includes('token stream values unavailable: 226') && missingValues.noNegativeArtifactIndexes, JSON.stringify(missingValues));

  const controls = await evaluate(session, `(() => {
    document.querySelector('[data-reset]').click();
    const attempt = DATA.nodes.find(node => node.role === 'dev' && node.model === 'opus-5').i;
    const shape = {
      swim: () => document.querySelector('[data-view="swim"] [data-node="' + attempt + '"]').getAttribute('fill'),
      force: () => document.querySelector('[data-view="force"] [data-node="' + attempt + '"]').getAttribute('fill'),
      cost: () => document.querySelector('[data-view="cost"] .seg[data-node="' + attempt + '"]').getAttribute('stroke'),
    };
    const before = Object.fromEntries(Object.entries(shape).map(([key, value]) => [key, value()]));
    for (const key of ['swim', 'force', 'cost']) {
      const select = document.querySelector('[data-view="' + key + '"] [data-view-color]');
      select.value = select.value === 'role' ? 'model' : 'role';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    }
    const after = Object.fromEntries(Object.entries(shape).map(([key, value]) => [key, value()]));
    document.querySelector('[data-reset]').click();
    const count = (view, selector) => document.querySelectorAll('[data-view="' + view + '"] ' + selector).length;
    const handoffBefore = { swim: count('swim', '.e-artifact'), force: count('force', '[data-art]'), cost: count('cost', '.e-artifact') };
    for (const key of ['swim', 'force', 'cost']) {
      const select = document.querySelector('[data-view="' + key + '"] [data-extra="handoffs"]');
      select.value = 'none'; select.dispatchEvent(new Event('change', { bubbles: true }));
    }
    const handoffAfter = { swim: count('swim', '.e-artifact'), force: count('force', '[data-art]'), cost: count('cost', '.e-artifact') };
    document.querySelector('[data-reset]').click();
    for (const key of ['swim', 'cost']) {
      const box = document.querySelector('[data-view="' + key + '"] [data-view-artifacts]');
      box.checked = true; box.dispatchEvent(new Event('change', { bubbles: true }));
    }
    const toggled = { swim: count('swim', '[data-art]'), cost: count('cost', '[data-art]') };
    return { before, after, handoffBefore, handoffAfter, toggled };
  })()`);
  check('color controls', ['swim', 'force', 'cost'].every(key => controls.before[key] !== controls.after[key]), JSON.stringify({ before: controls.before, after: controls.after }));
  check('handoff controls', controls.handoffBefore.swim > 0 && controls.handoffBefore.force > 0 && controls.handoffBefore.cost > 0 && Object.values(controls.handoffAfter).every(value => value === 0), JSON.stringify({ before: controls.handoffBefore, after: controls.handoffAfter }));
  check('artifact toggles', controls.toggled.swim > 0 && controls.toggled.cost > 0, JSON.stringify(controls.toggled));

  const filters = await evaluate(session, `(() => {
    const counts = () => Object.fromEntries(['swim', 'force', 'cost'].map(view => [view, new Set([...document.querySelectorAll('[data-view="' + view + '"] [data-node]')].map(el => el.dataset.node)).size]));
    document.querySelector('[data-reset]').click();
    const base = counts();
    document.querySelector('[data-role="dev"]').click();
    const role = counts();
    document.querySelector('[data-reset]').click();
    const modelSelect = document.querySelector('[data-model-filter]');
    modelSelect.value = 'gpt-5.6-sol'; modelSelect.dispatchEvent(new Event('change', { bubbles: true }));
    const model = counts();
    const rows = document.querySelectorAll('#table tbody tr').length;
    return { base, role, model, rows };
  })()`);
  const everyChanged = result => ['swim', 'force', 'cost'].every(key => result[key] < filters.base[key]);
  check('role filter reaches every view', everyChanged(filters.role), JSON.stringify(filters.role));
  check('model filter reaches every view', everyChanged(filters.model) && filters.rows === filters.model.swim, JSON.stringify({ model: filters.model, rows: filters.rows }));

  const emptyAndUntimed = await evaluate(session, `(() => {
    document.querySelector('[data-reset]').click();
    const model = document.querySelector('[data-model-filter]'); model.value = 'fable-5'; model.dispatchEvent(new Event('change', { bubbles: true }));
    document.querySelector('[data-role="planner"]').click();
    const empty = {
      rows: document.querySelectorAll('#table tbody tr').length,
      nodes: ['swim', 'force', 'cost'].map(view => document.querySelectorAll('[data-view="' + view + '"] [data-node]').length),
      svgs: document.querySelectorAll('details[data-view] [data-graph] > svg').length,
    };
    document.querySelector('[data-reset]').click();
    const untimed = DATA.nodes.filter(node => node.untimed).map(node => node.i);
    const hollow = untimed.map(i => ['swim', 'force', 'cost'].every(view => document.querySelector('[data-view="' + view + '"] .node.hollow[data-node="' + i + '"]')));
    const costLabel = [...document.querySelectorAll('[data-view="cost"] text')].some(el => el.textContent.includes('(untimed)'));
    return { empty, untimed: untimed.length, hollow, costLabel };
  })()`);
  check('empty filter intersection', emptyAndUntimed.empty.rows === 0 && emptyAndUntimed.empty.nodes.every(value => value === 0) && emptyAndUntimed.empty.svgs === 3, JSON.stringify(emptyAndUntimed.empty));
  check('untimed attempts', emptyAndUntimed.untimed > 0 && emptyAndUntimed.hollow.every(Boolean) && emptyAndUntimed.costLabel, JSON.stringify(emptyAndUntimed));

  const folds = await evaluate(session, `(() => {
    document.querySelector('[data-reset]').click();
    const views = [...document.querySelectorAll('details[data-view]')];
    views.forEach(view => { view.open = true; });
    views[0].querySelector('summary').click(); const first = views.map(view => view.open);
    views[1].querySelector('summary').click(); const second = views.map(view => view.open);
    return { first, second };
  })()`);
  check('independent folds', JSON.stringify(folds.first) === JSON.stringify([false, true, true]) && JSON.stringify(folds.second) === JSON.stringify([false, false, true]), JSON.stringify(folds));
  check('console remains clean', consoleErrors.length === 0 && exceptions.length === 0, `console=${consoleErrors.length} uncaught=${exceptions.length}`);
  }

  process.stdout.write(`SUMMARY passed=${passed} failed=${failed}\n`);
  await call('Browser.close');
  await Promise.race([new Promise(resolveExit => chrome.once('exit', resolveExit)), delay(3000)]);
} catch (error) {
  failed++;
  process.stdout.write(`FAIL browser probe: ${error.stack || error}\n`);
  process.stdout.write(`SUMMARY passed=${passed} failed=${failed}\n`);
} finally {
  if (chrome.exitCode == null) { chrome.kill('SIGTERM'); await Promise.race([new Promise(resolveExit => chrome.once('exit', resolveExit)), delay(3000)]); }
  if (socket && socket.readyState === WebSocket.OPEN) socket.close();
  for (let attempt = 0; attempt < 5; attempt++) {
    try { await rm(profile, { recursive: true, force: true }); break; }
    catch (error) { if (attempt === 4) throw error; await delay(100); }
  }
}

process.exit(failed ? 1 : 0);
