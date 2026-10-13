#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { attemptAuditSource } from './graph_html_probe_common.mjs';

const files = process.argv.slice(2);
if (files.length !== 5) {
  process.stderr.write('usage: node tests/graph_html_parity_probe.mjs OUTPUT PROTO1 PROTO5 PROTO3 SOURCE_JSON\n');
  process.exit(2);
}
const sourceOracle = JSON.parse(await readFile(files[4], 'utf8'));
const attemptAudit = attemptAuditSource(sourceOracle);

const outDir = resolve('out');
await mkdir(outDir, { recursive: true });
const profile = await mkdtemp(outDir + '/chrome-parity-');
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
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

const delay = ms => new Promise(done => setTimeout(done, ms));
async function endpoint() {
  for (let attempt = 0; attempt < 200; attempt++) {
    try {
      const [port, path] = (await readFile(profile + '/DevToolsActivePort', 'utf8')).trim().split('\n');
      if (port && path) return `ws://127.0.0.1:${port}${path}`;
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
  return new Promise((done, fail) => pending.set(id, { done, fail }));
}
async function evaluate(session, expression) {
  const answer = await call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }, session);
  if (answer.exceptionDetails) throw new Error(answer.exceptionDetails.text || 'evaluation failed');
  return answer.result.value;
}

let failures = 0;
function check(name, condition, detail = '') {
  process.stdout.write(`${condition ? 'PASS' : 'FAIL'} ${name}${detail ? ': ' + detail : ''}\n`);
  if (!condition) failures++;
}

const snapshotSource = selector => `(() => {
  const host = document.querySelector(${JSON.stringify(selector)});
  const root = host.querySelector('svg');
  const attrs = el => Object.fromEntries([...el.attributes].filter(item => !['data-edge', 'data-edge-list', 'data-connection-node', 'data-connection-kind'].includes(item.name)).map(item => [item.name, item.value]));
  const records = query => [...host.querySelectorAll(query)].map(el => ({ tag: el.tagName, attrs: attrs(el), text: el.textContent })).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
  const claimed = new Set([...host.querySelectorAll('[data-node],[data-edge],[data-art],.alink')]);
  const plain = [...host.querySelectorAll('rect,line,path,circle')].filter(el => !claimed.has(el)).map(el => ({ tag: el.tagName, attrs: attrs(el) })).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
  return {
    frame: root ? { width: root.getAttribute('width'), height: root.getAttribute('height'), viewBox: root.getAttribute('viewBox') } : null,
    nodes: records('[data-node]'),
    edges: records('[data-edge]'),
    artifacts: records('[data-art]'),
    artifactLinks: records('.alink'),
    labels: records('text'),
    plain,
  };
})()`;
const tableSource = `(() => ({
  headers: [...document.querySelectorAll('#table th')].map(el => el.textContent.trim().replace(/[▲▼]/g, '')),
  rows: [...document.querySelectorAll('#table tbody tr')].map(row => [...row.cells].map(cell => cell.textContent.trim())),
}))()`;
const sortSource = `(() => {
  const result = {};
  const keys = [...document.querySelectorAll('#table th[data-key]')].map(header => header.dataset.key);
  for (const key of keys) {
    document.querySelector('#table th[data-key="' + key + '"]').click();
    const firstArrow = document.querySelector('#table th[data-key="' + key + '"] .arrow')?.textContent;
    const first = [...document.querySelectorAll('#table tbody tr')].map(row => row.dataset.i);
    document.querySelector('#table th[data-key="' + key + '"]').click();
    const secondArrow = document.querySelector('#table th[data-key="' + key + '"] .arrow')?.textContent;
    const second = [...document.querySelectorAll('#table tbody tr')].map(row => row.dataset.i);
    result[key] = { firstArrow, secondArrow, first, second };
  }
  return result;
})()`;
const cardSource = `(() => {
  const row = [...document.querySelectorAll('#table tbody tr')].sort((left, right) => Number(right.cells[8].textContent) + Number(right.cells[9].textContent) - Number(left.cells[8].textContent) - Number(left.cells[9].textContent))[0];
  const written = Number(row.cells[8].textContent), read = Number(row.cells[9].textContent);
  row.click();
  const card = document.querySelector('#card'), terms = [...card.querySelectorAll('dt')], fields = terms.map(el => el.textContent.trim());
  const links = prefix => [...(terms.find(el => el.textContent.trim().startsWith(prefix))?.nextElementSibling.querySelectorAll('[data-art-link]') || [])].map(el => Number(el.dataset.artLink));
  const idMarker = card.querySelector('[data-attempt-field="id"]');
  const attempt = { fields, id: idMarker ? JSON.parse(idMarker.dataset.attemptValue) : null, written, read, writtenLinks: links('written ('), readLinks: links('read (') };
  const artifactLink = card.querySelector('[data-art-link]'), artifactIndex = Number(artifactLink?.dataset.artLink); if (artifactLink) artifactLink.click();
  const artifactTerms = [...card.querySelectorAll('dt')];
  const artifact = {
    visible: !card.hidden && card.querySelector('h3')?.textContent.includes('artifact'),
    index: artifactIndex,
    fields: artifactTerms.map(el => el.textContent.trim()),
    entries: artifactTerms.map(el => [el.textContent.trim(), el.nextElementSibling.textContent.trim()]),
  };
  return { attempt, artifact };
})()`;
const interactionSource = (view, prototype = false) => {
  const hostSelector = prototype ? '#graph' : `[data-view="${view}"] [data-graph]`;
  const resetSelector = prototype ? '#f-reset' : '[data-reset]';
  const modelSelector = prototype ? '#f-model' : '[data-model-filter]';
  const colorSelector = prototype ? '#f-color' : `[data-view="${view}"] [data-view-color]`;
  const handoffSelector = prototype ? '[data-extra="handoffs"]' : `[data-view="${view}"] [data-extra="handoffs"]`;
  const stackSelector = prototype ? '[data-extra="stack"]' : `[data-view="${view}"] [data-extra="stack"]`;
  const artifactSelector = prototype ? '#f-art' : `[data-view="${view}"] [data-view-artifacts]`;
  const alignForce = prototype && view === 'force';
  return `(() => {
    const host = document.querySelector(${JSON.stringify(hostSelector)});
    const attrs = el => Object.fromEntries([...el.attributes].filter(item => !['data-edge', 'data-edge-list', 'data-connection-node', 'data-connection-kind'].includes(item.name)).map(item => [item.name, item.value]));
    const records = query => [...host.querySelectorAll(query)].map(el => ({ tag: el.tagName, attrs: attrs(el), text: el.textContent })).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
    const snapshot = () => {
      const root = host.querySelector('svg'), claimed = new Set([...host.querySelectorAll('[data-node],[data-edge],[data-art],.alink')]);
      return {
        rows: [...document.querySelectorAll('#table tbody tr')].map(row => row.dataset.i),
        frame: root ? { width: root.getAttribute('width'), height: root.getAttribute('height'), viewBox: root.getAttribute('viewBox') } : null,
        nodes: records('[data-node]'), edges: records('[data-edge]'), artifacts: records('[data-art]'), artifactLinks: records('.alink'), labels: records('text'),
        plain: [...host.querySelectorAll('rect,line,path,circle')].filter(el => !claimed.has(el)).map(el => ({ tag: el.tagName, attrs: attrs(el) })).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))),
      };
    };
    const change = (selector, value) => { const el = document.querySelector(selector); el.value = value; el.dispatchEvent(new Event('change', { bubbles: true })); };
    const reset = () => {
      document.querySelector(${JSON.stringify(resetSelector)}).click();
      ${alignForce ? `const artifact = document.querySelector(${JSON.stringify(artifactSelector)}); artifact.checked = true; artifact.dispatchEvent(new Event('change', { bubbles: true }));` : ''}
    };
    reset(); document.querySelector('[data-role="dev"]').click(); const role = snapshot();
    reset(); change(${JSON.stringify(modelSelector)}, 'gpt-5.6-sol'); const model = snapshot();
    reset(); const colorControl = document.querySelector(${JSON.stringify(colorSelector)}); change(${JSON.stringify(colorSelector)}, colorControl.value === 'role' ? 'model' : 'role'); const color = snapshot();
    const handoffs = {}; for (const mode of ['all', 'verified', 'none']) { reset(); change(${JSON.stringify(handoffSelector)}, mode); handoffs[mode] = snapshot(); }
    const stacks = {}; if (${JSON.stringify(view)} === 'cost') for (const mode of ['start', 'role', 'cost']) { reset(); change(${JSON.stringify(stackSelector)}, mode); stacks[mode] = snapshot(); }
    let artifacts = null; reset(); const artifactControl = document.querySelector(${JSON.stringify(artifactSelector)}); if (artifactControl) { artifactControl.checked = true; artifactControl.dispatchEvent(new Event('change', { bubbles: true })); artifacts = snapshot(); }
    reset(); change(${JSON.stringify(modelSelector)}, 'fable-5'); document.querySelector('[data-role="planner"]').click(); const empty = snapshot();
    reset(); const untimed = DATA.nodes.filter(node => node.untimed).map(node => ({ i: node.i, shapes: [...host.querySelectorAll('[data-node="' + node.i + '"]')].map(attrs) }));
    const styled = selector => { const el = host.querySelector(selector); if (!el) return null; const value = getComputedStyle(el); return { stroke: value.stroke, fill: value.fill, width: value.strokeWidth, dash: value.strokeDasharray, opacity: value.opacity, pointer: value.pointerEvents }; };
    const styles = Object.fromEntries([['spawn', '.e-spawn'], ['launch', '.e-launch'], ['verified handoff', '.e-artifact.t-verified'], ['heuristic handoff', '.e-artifact.t-heuristic'], ['node', '.node,.seg'], ['hollow', '.node.hollow,.seg.unavailable']].map(([name, selector]) => [name, styled(selector)]));
    return { role, model, color, handoffs, stacks, artifacts, empty, untimed, styles };
  })()`;
};
const requirementSource = `(() => {
  const tip = document.querySelector('#tip');
  const move = el => { if (el) el.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 40, clientY: 40 })); return !!el && !tip.hidden && tip.textContent.length > 0; };
  const edgeTips = Object.fromEntries([
    ['swim', '[data-view="swim"] [data-edge]'],
    ['force', '[data-view="force"] .alink[data-edge-list]'],
    ['cost', '[data-view="cost"] [data-edge]'],
  ].map(([key, selector]) => [key, move(document.querySelector(selector))]));
  const forceConnections = [...document.querySelectorAll('[data-view="force"] .alink')];
  const views = [...document.querySelectorAll('details[data-view]')];
  const nodeCards = Object.fromEntries(['swim', 'force', 'cost'].map(key => { document.querySelector('[data-view="' + key + '"] [data-node]').dispatchEvent(new MouseEvent('click', { bubbles: true })); return [key, !document.querySelector('#card').hidden && [...document.querySelectorAll('#card dt')].some(el => el.textContent === 'attempt id')]; }));
  document.querySelector('#table tbody tr').click(); const tableCard = !document.querySelector('#card').hidden && [...document.querySelectorAll('#card dt')].some(el => el.textContent === 'attempt id');
  document.querySelector('[data-view="force"] [data-art]').dispatchEvent(new MouseEvent('click', { bubbles: true })); const artifactCard = !document.querySelector('#card').hidden && document.querySelector('#card h3')?.textContent.includes('artifact');
  return {
    edgeTips,
    nodeTips: ['swim', 'force', 'cost'].every(key => move(document.querySelector('[data-view="' + key + '"] [data-node]'))),
    artifactTip: move(document.querySelector('[data-view="force"] [data-art]')),
    forceConnectionData: forceConnections.length > 0 && forceConnections.every(el => el.hasAttribute('data-edge-list') && el.hasAttribute('data-connection-node') && el.hasAttribute('data-connection-kind')),
    forceToggleCount: document.querySelectorAll('[data-view="force"] [data-view-artifacts]').length,
    viewOrder: views.map(view => view.dataset.view),
    viewsOpen: views.map(view => view.open),
    tableFirst: document.querySelector('[data-attempt-table]').compareDocumentPosition(views[0]) === Node.DOCUMENT_POSITION_FOLLOWING,
    accounting: document.querySelector('#accounting')?.textContent || '',
    nodeCards, tableCard, artifactCard,
  };
})()`;

try {
  socket = new WebSocket(await endpoint());
  await new Promise((done, fail) => { socket.onopen = done; socket.onerror = () => fail(new Error('DevTools socket failed')); });
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const item = pending.get(message.id); pending.delete(message.id);
      if (message.error) item.fail(new Error(message.error.message)); else item.done(message.result);
      return;
    }
    (listeners.get(message.method) || []).forEach(listener => listener(message));
  };
  const target = await call('Target.createTarget', { url: 'about:blank' });
  const attached = await call('Target.attachToTarget', { targetId: target.targetId, flatten: true });
  const session = attached.sessionId;
  await call('Runtime.enable', {}, session);
  await call('Page.enable', {}, session);
  await call('Emulation.setDeviceMetricsOverride', { width: 1366, height: 900, deviceScaleFactor: 1, mobile: false }, session);

  async function load(file) {
    const loaded = new Promise(done => {
      const listener = message => {
        if (message.sessionId === session) done();
      };
      on('Page.loadEventFired', listener);
    });
    await call('Page.navigate', { url: pathToFileURL(resolve(file)).href }, session);
    await loaded;
    await evaluate(session, 'new Promise(done => setTimeout(done, 80))');
  }

  await load(files[0]);
  const output = {
    table: await evaluate(session, tableSource),
    swim: await evaluate(session, snapshotSource('[data-view="swim"] [data-graph]')),
    force: await evaluate(session, snapshotSource('[data-view="force"] [data-graph]')),
    cost: await evaluate(session, snapshotSource('[data-view="cost"] [data-graph]')),
    sorts: await evaluate(session, sortSource),
    card: await evaluate(session, cardSource),
    attemptAudit: await evaluate(session, attemptAudit),
    requirements: await evaluate(session, requirementSource),
    interactions: {},
  };
  for (const view of ['swim', 'force', 'cost']) output.interactions[view] = await evaluate(session, interactionSource(view));
  await load(files[1]);
  const proto1 = { table: await evaluate(session, tableSource), graph: await evaluate(session, snapshotSource('#graph')), sorts: await evaluate(session, sortSource), card: await evaluate(session, cardSource), interactions: await evaluate(session, interactionSource('swim', true)) };
  await load(files[2]);
  await evaluate(session, `(() => { const box = document.querySelector('#f-art'); box.checked = true; box.dispatchEvent(new Event('change', { bubbles: true })); })()`);
  const proto5 = { table: await evaluate(session, tableSource), graph: await evaluate(session, snapshotSource('#graph')), sorts: await evaluate(session, sortSource), card: await evaluate(session, cardSource), interactions: await evaluate(session, interactionSource('force', true)) };
  await load(files[3]);
  const proto3 = { table: await evaluate(session, tableSource), graph: await evaluate(session, snapshotSource('#graph')), sorts: await evaluate(session, sortSource), card: await evaluate(session, cardSource), interactions: await evaluate(session, interactionSource('cost', true)) };

  const withoutTokenCells = table => ({ headers: table.headers, rows: table.rows.map(row => row.filter((_, index) => index !== 5)) });
  check('attempt table non-token cells against prototype 1', JSON.stringify(withoutTokenCells(output.table)) === JSON.stringify(withoutTokenCells(proto1.table)), `${output.table.rows.length} rows`);
  for (const [name, prototype] of [['prototype 1', proto1], ['prototype 5', proto5], ['prototype 3', proto3]]) {
    for (const key of Object.keys(output.sorts)) {
      const actual = output.sorts[key], expected = prototype.sorts[key];
      if (key === 'tok') {
        check(`table sort tok marks incomplete data against ${name}`, actual.firstArrow === '▼' && actual.secondArrow === '▲' && output.table.rows.every(row => row[5] === 'n/a'), `${actual.firstArrow}/${actual.secondArrow}; ${output.table.rows.filter(row => row[5] === 'n/a').length} unavailable`);
        continue;
      }
      const same = JSON.stringify(actual) === JSON.stringify(expected);
      const differing = same ? -1 : actual.first.findIndex((value, index) => value !== expected.first[index]);
      check(`table sort ${key} against ${name}`, same, same ? `${actual.firstArrow}/${actual.secondArrow}` : `first difference at ${differing}: output=${actual.first[differing]}, prototype=${expected.first[differing]}`);
    }
  }
  for (const [name, actual, expected] of [
    ['swimlanes', output.swim, proto1.graph],
    ['force with artifacts', output.force, proto5.graph],
    ['cost curve', output.cost, proto3.graph],
  ]) {
    for (const part of ['frame', 'nodes', 'edges', 'artifacts', 'artifactLinks', 'labels', 'plain']) {
      const same = JSON.stringify(actual[part]) === JSON.stringify(expected[part]);
      check(`${name} ${part}`, same, `output=${actual[part]?.length ?? JSON.stringify(actual[part])} prototype=${expected[part]?.length ?? JSON.stringify(expected[part])}`);
    }
  }
  check('prototype 5 non-token table remains shared', JSON.stringify(withoutTokenCells(output.table)) === JSON.stringify(withoutTokenCells(proto5.table)), `${proto5.table.rows.length} rows`);
  check('prototype 3 non-token table remains shared', JSON.stringify(withoutTokenCells(output.table)) === JSON.stringify(withoutTokenCells(proto3.table)), `${proto3.table.rows.length} rows`);
  check('complete source attempt payload and rendered card', output.attemptAudit.errorCount === 0, output.attemptAudit.errorCount ? output.attemptAudit.errors.join('; ') : `${output.attemptAudit.nodes} attempts, ${output.attemptAudit.sourceFields.length} source fields, ${output.attemptAudit.tokenStreams.length} token streams, ${output.attemptAudit.omissions.length} omissions, ${output.attemptAudit.forceConnections} source-checked force connections`);
  const sourceLinks = sourceOracle.artifact_links[output.card.attempt.id];
  check('selected attempt artifact links against independent source', Boolean(sourceLinks) && JSON.stringify(output.card.attempt.writtenLinks) === JSON.stringify(sourceLinks.written) && JSON.stringify(output.card.attempt.readLinks) === JSON.stringify(sourceLinks.read) && output.card.attempt.written === sourceLinks.written.length && output.card.attempt.read === sourceLinks.read.length, `written ${output.card.attempt.writtenLinks.length}/${sourceLinks?.written.length ?? 'missing source'}, read ${output.card.attempt.readLinks.length}/${sourceLinks?.read.length ?? 'missing source'}`);
  check('prototype card eight-link limit detected', [proto1, proto5, proto3].every(proto => proto.card.attempt.writtenLinks.length === Math.min(8, proto.card.attempt.written) && proto.card.attempt.readLinks.length === Math.min(8, proto.card.attempt.read)), `prototype links ${proto1.card.attempt.writtenLinks.length}/${proto1.card.attempt.written} and ${proto1.card.attempt.readLinks.length}/${proto1.card.attempt.read}`);
  const artifactFields = ['kind', 'producer', 'writers', 'consumers', 'first write', 'writes / reads', 'language', 'size'];
  const selectedSourceArtifact = sourceOracle.artifacts[output.card.artifact.index];
  const expectedArtifactFields = [...artifactFields, ...(selectedSourceArtifact?.hint ? ['hint'] : [])];
  check('complete artifact card inventory', JSON.stringify(output.card.artifact.fields) === JSON.stringify(expectedArtifactFields), output.card.artifact.fields.join(', '));
  for (const [name, prototype] of [['prototype 1', proto1], ['prototype 5', proto5], ['prototype 3', proto3]]) {
    const required = card => card.artifact.entries.filter(([field]) => artifactFields.includes(field));
    const same = JSON.stringify(required(output.card)) === JSON.stringify(required(prototype.card));
    check(`artifact card against ${name}`, same, same ? artifactFields.join(', ') : `output=${JSON.stringify(required(output.card))} prototype=${JSON.stringify(required(prototype.card))}`);
  }
  const optionalFieldMatchesSource = item => {
    const source = sourceOracle.artifacts[item.card.artifact.index];
    const entries = new Map(item.card.artifact.entries);
    return item.card.artifact.fields.includes('hint') === Boolean(source?.hint)
      && (!source?.hint || entries.get('hint') === String(source.hint));
  };
  check('artifact optional field follows independent source', optionalFieldMatchesSource(output), JSON.stringify({ index: output.card.artifact.index, sourceHasHint: Boolean(selectedSourceArtifact?.hint), fields: output.card.artifact.fields }));
  check('table and three view structure', output.requirements.tableFirst && JSON.stringify(output.requirements.viewOrder) === JSON.stringify(['swim', 'force', 'cost']) && output.requirements.viewsOpen.every(Boolean), JSON.stringify({ tableFirst: output.requirements.tableFirst, views: output.requirements.viewOrder, open: output.requirements.viewsOpen }));
  check('node and edge tooltips in every view', output.requirements.nodeTips && Object.values(output.requirements.edgeTips).every(Boolean) && output.requirements.artifactTip, JSON.stringify({ node: output.requirements.nodeTips, edges: output.requirements.edgeTips, artifact: output.requirements.artifactTip }));
  check('row and node cards in every view', output.requirements.tableCard && output.requirements.artifactCard && Object.values(output.requirements.nodeCards).every(Boolean), JSON.stringify({ table: output.requirements.tableCard, nodes: output.requirements.nodeCards, artifact: output.requirements.artifactCard }));
  check('force connection data and control', output.requirements.forceConnectionData && output.requirements.forceToggleCount === 0, JSON.stringify({ data: output.requirements.forceConnectionData, toggleCount: output.requirements.forceToggleCount }));
  check('viewer accounting is surfaced', output.requirements.accounting.includes('attempt durations unavailable: 1') && output.requirements.accounting.includes('token stream values unavailable: 226'), output.requirements.accounting);
  for (const [view, prototype] of [['swim', proto1], ['force', proto5], ['cost', proto3]]) {
    for (const feature of ['role', 'model', 'color', 'empty', 'untimed', 'styles']) {
      check(`${view} ${feature} behaviour against prototype`, JSON.stringify(output.interactions[view][feature]) === JSON.stringify(prototype.interactions[feature]));
    }
    for (const mode of ['all', 'verified', 'none']) check(`${view} ${mode} handoffs against prototype`, JSON.stringify(output.interactions[view].handoffs[mode]) === JSON.stringify(prototype.interactions.handoffs[mode]));
    if (view === 'cost') for (const mode of ['start', 'role', 'cost']) check(`cost ${mode} stack against prototype`, JSON.stringify(output.interactions.cost.stacks[mode]) === JSON.stringify(prototype.interactions.stacks[mode]));
    if (view !== 'force') {
      for (const part of Object.keys(output.interactions[view].artifacts)) {
        const actual = output.interactions[view].artifacts[part], expected = prototype.interactions.artifacts[part], same = JSON.stringify(actual) === JSON.stringify(expected);
        const differing = Array.isArray(actual) && Array.isArray(expected) ? actual.findIndex((value, index) => JSON.stringify(value) !== JSON.stringify(expected[index])) : -1;
        check(`${view} artifacts enabled ${part} against prototype`, same, same || differing < 0 ? '' : `first difference ${differing}: output=${JSON.stringify(actual[differing])}, prototype=${JSON.stringify(expected[differing])}`);
      }
    }
  }
  process.stdout.write(`SUMMARY failed=${failures}\n`);
  await call('Browser.close');
  await Promise.race([new Promise(done => chrome.once('exit', done)), delay(3000)]);
} catch (error) {
  failures++;
  process.stdout.write(`FAIL parity probe: ${error.stack || error}\nSUMMARY failed=${failures}\n`);
} finally {
  if (chrome.exitCode == null) chrome.kill('SIGTERM');
  if (socket && socket.readyState === WebSocket.OPEN) socket.close();
  await rm(profile, { recursive: true, force: true });
}

process.exit(failures ? 1 : 0);
