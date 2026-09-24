export function attemptAuditSource(sourceOracle) {
  const encoded = Buffer.from(JSON.stringify(sourceOracle), 'utf8').toString('base64');
  return `(() => {
  document.querySelector('[data-reset]').click();
  const bytes = Uint8Array.from(atob('${encoded}'), value => value.charCodeAt(0));
  const SOURCE = JSON.parse(new TextDecoder().decode(bytes));
  const sourceFields = SOURCE.source_fields;
  const tokenStreams = SOURCE.token_streams;
  const roots = {
    id: ['id'], harness: ['harness'], source: ['src'], session_path: ['session'],
    model: ['model'], model_tier: ['mt'], effort: ['effort'], workspace: ['ws'],
    ts: ['ts'], wall_s: ['dur'], tokens: ['tok', 'tok_record'], usd: ['usd'],
    parent: ['parent_source'], spawn: ['spawn'], launched_by: ['launch'],
    role: ['role_source'], role_tier: ['rt'], role_evidence: ['re'],
    phase: ['phase'], phase_tier: ['pt'],
  };
  const payload = n => ({
    id: n.id, harness: n.harness, source: n.src, session_path: n.session,
    model: n.model, model_tier: n.mt, effort: n.effort, workspace: n.ws,
    ts: n.ts, wall_s: n.dur, tokens: n.tok_record ? n.tok : null, usd: n.usd,
    parent: n.parent_source, spawn: n.spawn, launched_by: n.launch,
    role: n.role_source, role_tier: n.rt, role_evidence: n.re,
    phase: n.phase, phase_tier: n.pt,
  });
  const normalized = value => {
    if (Array.isArray(value)) return value.map(normalized);
    if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, normalized(value[key])]));
    return value;
  };
  const same = (left, right) => JSON.stringify(normalized(left)) === JSON.stringify(normalized(right));
  const sameKeys = (left, right) => same([...left].sort(), [...right].sort());
  const markerValue = marker => {
    if (!marker) return { missing: true };
    try { return JSON.parse(marker.dataset.attemptValue); }
    catch { return { invalid: marker.dataset.attemptValue }; }
  };
  const errors = [];
  const sourceById = new Map();
  for (const source of SOURCE.nodes) {
    if (sourceById.has(source.id)) errors.push('source oracle has duplicate id ' + source.id);
    sourceById.set(source.id, source);
  }
  const payloadById = new Map();
  for (const node of DATA.nodes) {
    if (payloadById.has(node.id)) errors.push('payload has duplicate id ' + node.id);
    payloadById.set(node.id, node);
  }
  if (SOURCE.schema !== 'GraphNode') errors.push('source oracle schema differs');
  if (!sameKeys(sourceFields, Object.keys(roots))) errors.push('source schema does not equal payload field map');
  if (!same(DATA.run.attempt_source_fields, sourceFields)) errors.push('payload source field inventory differs from oracle');
  if (!same(DATA.run.attempt_field_omissions, SOURCE.omissions)) errors.push('payload omission list differs from oracle');
  if (!same(DATA.run.token_streams, SOURCE.token_streams)) errors.push('payload token stream inventory differs from oracle');
  if (!same(DATA.run.token_total_streams, SOURCE.token_total_streams)) errors.push('payload token total inventory differs from oracle');
  if (!sameKeys(sourceById.keys(), payloadById.keys())) errors.push('payload node inventory differs from oracle');
  for (const source of SOURCE.nodes) {
    const n = payloadById.get(source.id);
    if (!n) { errors.push('source node ' + source.id + ' is absent from payload'); continue; }
    const row = document.querySelector('#table tr[data-i="' + n.i + '"]');
    if (!row) { errors.push('node ' + n.i + ' has no table row'); continue; }
    row.click();
    const card = document.querySelector('#card');
    if (card.hidden) errors.push('node ' + n.i + ' card did not open');
    const terms = [...card.querySelectorAll('dt')].map(el => el.textContent.trim());
    const expectedTerms = ['attempt id', 'role', 'model', 'harness', 'phase', 'started', 'duration', 'cost', 'tokens', 'status', 'origin', 'children', 'written (' + n.written.length + ')', 'read (' + n.read.length + ')', 'source fields (' + sourceFields.length + ')', 'session'];
    if (!same(terms, expectedTerms)) errors.push('node ' + n.i + ' card labels differ');
    const values = payload(n);
    const markers = new Map([...card.querySelectorAll('[data-attempt-field]')].map(el => [el.dataset.attemptField, el]));
    for (const field of sourceFields) {
      for (const root of roots[field] || []) if (!Object.hasOwn(n, root)) errors.push('node ' + n.i + ' payload missing ' + root);
      if (!same(values[field], source[field])) errors.push('node ' + n.i + ' payload value differs for ' + field);
      const marker = markers.get(field);
      if (!marker) errors.push('node ' + n.i + ' card missing ' + field);
      else if (!same(markerValue(marker), source[field])) errors.push('node ' + n.i + ' card value differs for ' + field);
    }
    if (!sameKeys(markers.keys(), sourceFields)) errors.push('node ' + n.i + ' rendered field inventory differs');
    const sourceRows = new Map([...card.querySelectorAll('[data-source-field]')].map(el => [el.dataset.sourceField, el]));
    if (!sameKeys(sourceRows.keys(), sourceFields)) errors.push('node ' + n.i + ' visible source field inventory differs');
    for (const field of sourceFields) {
      const row = sourceRows.get(field);
      if (!row || !same(markerValue(row), source[field])) errors.push('node ' + n.i + ' visible source value differs for ' + field);
      const expectedText = JSON.stringify(source[field]);
      if (!row || row.querySelector('[data-source-visible]')?.textContent !== (expectedText == null ? 'unavailable' : expectedText)) errors.push('node ' + n.i + ' visible source text differs for ' + field);
    }
    for (const field of ['spawn', 'launched_by']) {
      const expected = source[field];
      const nested = new Map([...card.querySelectorAll('[data-nested-field="' + field + '"]')].map(el => [el.dataset.nestedKey, markerValue(el)]));
      if (expected == null) {
        if (!card.querySelector('[data-nested-empty="' + field + '"]')) errors.push('node ' + n.i + ' card does not show empty ' + field);
      } else {
        if (!sameKeys(nested.keys(), Object.keys(expected))) errors.push('node ' + n.i + ' card nested inventory differs for ' + field);
        for (const [key, value] of Object.entries(expected)) {
          if (!same(nested.get(key), value)) errors.push('node ' + n.i + ' card nested value differs for ' + field + '.' + key);
          const marker = card.querySelector('[data-nested-field="' + field + '"][data-nested-key="' + CSS.escape(key) + '"]');
          const shown = value == null ? 'n/a' : typeof value === 'object' ? JSON.stringify(value) : String(value);
          if (!marker || marker.textContent !== shown) errors.push('node ' + n.i + ' card nested text differs for ' + field + '.' + key);
        }
      }
    }
    const expectedTokens = source.tokens || {};
    const expectedTokenKeys = tokenStreams;
    const tokenMarkers = new Map([...card.querySelectorAll('[data-token-stream]')].map(el => [el.dataset.tokenStream, el]));
    if (!sameKeys(Object.keys(n.tok || {}), expectedTokenKeys)) errors.push('node ' + n.i + ' payload token inventory differs');
    if (!sameKeys(tokenMarkers.keys(), expectedTokenKeys)) errors.push('node ' + n.i + ' card token inventory differs');
    for (const stream of expectedTokenKeys) {
      const marker = tokenMarkers.get(stream);
      const expectedValue = expectedTokens[stream] ?? null;
      if (!same(n.tok?.[stream] ?? null, expectedValue)) errors.push('node ' + n.i + ' payload token differs for ' + stream);
      let renderedValue = { missing: true };
      try { if (marker) renderedValue = JSON.parse(marker.dataset.tokenValue); }
      catch { renderedValue = { invalid: marker?.dataset.tokenValue }; }
      if (!marker || !same(renderedValue, expectedValue)) errors.push('node ' + n.i + ' card token differs for ' + stream);
      if (!marker?.textContent.trim()) errors.push('node ' + n.i + ' card token is blank for ' + stream);
    }
    const complete = n.tok_record && tokenStreams.every(stream => Number.isFinite(n.tok?.[stream]));
    const total = complete ? SOURCE.token_total_streams.reduce((sum, stream) => sum + n.tok[stream], 0) : null;
    if (n.tokComplete !== complete || n.tokTotal !== total) errors.push('node ' + n.i + ' token completeness or total differs');
    const tokenText = terms.includes('tokens') ? card.querySelectorAll('dd')[terms.indexOf('tokens')].textContent : '';
    if (n.tokComplete ? !tokenText.includes('complete total') : !tokenText.includes('total unavailable')) errors.push('node ' + n.i + ' token completeness label differs');
    const expectedLinks = SOURCE.artifact_links[source.id] || { written: [], read: [] };
    const writtenLinks = [...card.querySelectorAll('[data-written-links] [data-art-link]')].map(link => Number(link.dataset.artLink));
    const readLinks = [...card.querySelectorAll('[data-read-links] [data-art-link]')].map(link => Number(link.dataset.artLink));
    if (!same(n.written, expectedLinks.written) || !same(writtenLinks, expectedLinks.written) || Number(row.cells[8].textContent) !== expectedLinks.written.length) errors.push('node ' + n.i + ' written artifact links differ from source');
    if (!same(n.read, expectedLinks.read) || !same(readLinks, expectedLinks.read) || Number(row.cells[9].textContent) !== expectedLinks.read.length) errors.push('node ' + n.i + ' read artifact links differ from source');
  }
  const forceConnections = [...document.querySelectorAll('[data-view="force"] .alink')];
  const connectionRecord = connection => {
    const artifactIndex = Number(connection.dataset.alink);
    const nodeIndex = Number(connection.dataset.connectionNode);
    const kind = connection.dataset.connectionKind;
    const artifact = SOURCE.artifacts[artifactIndex];
    const node = DATA.nodes[nodeIndex];
    if (!artifact || !node || !['write', 'read'].includes(kind)) {
      return { invalid: true, artifact: artifactIndex, node: nodeIndex, kind };
    }
    const edges = (connection.dataset.edgeList || '').split(',').filter(Boolean).map(key => DATA.edges[Number(key)]).filter(Boolean).map(edge => ({ src: DATA.nodes[edge[0]]?.id, dst: DATA.nodes[edge[1]]?.id, tier: edge[3] }));
    return { artifact: artifactIndex, node: node.id, kind, edges };
  };
  const expectedForceConnections = SOURCE.artifacts.flatMap((artifact, artifactIndex) => {
    if (!artifact.consumers?.length || artifact.first_write_ts == null) return [];
    const records = [];
    for (const [kind, participants] of [['write', artifact.writers || []], ['read', artifact.consumers || []]]) {
      for (const node of participants) {
        if (!sourceById.has(node)) continue;
        const endpoint = kind === 'write' ? 'src' : 'dst';
        const edges = SOURCE.edges.filter(edge => edge.kind === 'artifact' && edge.detail?.path === artifact.id && edge[endpoint] === node).map(edge => ({ src: edge.src, dst: edge.dst, tier: edge.tier }));
        records.push({ artifact: artifactIndex, node, kind, edges });
      }
    }
    return records;
  });
  const actualForceConnections = forceConnections.map(connectionRecord);
  const connectionOrder = (left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right));
  if (!same(actualForceConnections.sort(connectionOrder), expectedForceConnections.sort(connectionOrder))) errors.push('force connection inventory or data differs from source');
  return { nodes: SOURCE.nodes.length, completeTokens: DATA.nodes.filter(n => n.tokComplete).length, incompleteTokens: DATA.nodes.filter(n => !n.tokComplete).length, sourceFields, payloadFields: Object.keys(roots), tokenStreams, omissions: SOURCE.omissions, forceConnections: forceConnections.length, errorCount: errors.length, errors: errors.slice(0, 40) };
})()`;
}
