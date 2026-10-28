import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { IDLE, band, begin, fieldsOf, fold, lines, optionsFor, planLines, said, shown, valueAt, watchEnd }
  from './events'
import type { HarnessEvent } from './events'

// The loopmath CLI is stubbed through the test's own process hooks: nothing runs, no model is called.

const TUNE = 'tune-20261024-102000-7f3a'
const T0 = '2026-01-15T10:20:00-07:00'

const ev = (seq: number, type: string, data: Record<string, unknown> = {}, extra: Partial<HarnessEvent> = {}) =>
  ({ v: 1, seq, ts: `2026-01-15T10:${String(20 + seq).padStart(2, '0')}:00-07:00`, tune: TUNE, type, data,
    stage: null, label: null, ...extra }) as HarnessEvent

const STREAM: HarnessEvent[] = [
  ev(1, 'tune.start', { plan: { stages: ['base', 'edit', 'probe', 'fit', 'deliver'], caps: { usd: 6 },
    cost: { point: 2.38 } } }),
  ev(2, 'stage.start', { planned: 3 }, { stage: 'base' }),
  ev(3, 'session.start', { role: 'executor', model: 'claude-haiku-5-5' }, { stage: 'base', label: 'base-1' }),
  ev(4, 'session.end', { role: 'executor', exit: 0, cost_usd: 0.25 }, { stage: 'base', label: 'base-1' }),
  ev(5, 'edit.proposed', { id: 'procedure-1' }, { stage: 'edit' }),
  ev(6, 'edit.refused', { id: 'procedure-2', reason: 'names a model' }, { stage: 'edit' }),
  ev(7, 'fit', { edits: [{ id: 'procedure-1', estimate: 4.5, se: 1.5, kept: true }], df: 3 }, { stage: 'fit' }),
  ev(8, 'deliver', { choice: 'build', rule: 'final-vs-base-mean', why: 'beat the base' }, { stage: 'deliver' }),
  ev(9, 'tune.end', { status: 'delivered', exit: 0 }),
]

// --------------------------------------------------------------------------------------------- the pure fold

test('fold builds the view from a stream and skips what it already folded', () => {
  let v = begin(TUNE)
  for (const e of STREAM.slice(0, 4)) v = fold(v, e)
  expect(v.stages.map(s => `${s.name}:${s.state}`)).toEqual(['base:running', 'edit:pending', 'probe:pending',
    'fit:pending', 'deliver:pending'])
  expect(v.cap).toBe(6)
  expect(v.spent).toBe(0.25)
  expect(v.sessions).toEqual([{ label: 'base-1', stage: 'base', role: 'executor', model: 'claude-haiku-5-5',
    state: 'done', exit: 0, cost: 0.25 }])
  expect(fold(v, STREAM[3]!)).toBe(v)
  for (const e of STREAM.slice(4)) v = fold(v, e)
  expect(v.edits).toEqual({ proposed: 1, refused: 1, kept: 0 })
  expect(v.effects).toEqual([{ id: 'procedure-1', estimate: 4.5, se: 1.5, kept: true }])
  expect(v.delivery).toBe('build by final-vs-base-mean: beat the base')
  expect(v.ended).toBe(true)
  expect(v.status).toBe('delivered')
  expect(v.exit).toBe(0)
  expect(v.seq).toBe(9)
})

test('a pause and a resume move the status', () => {
  let v = fold(begin(TUNE), ev(1, 'tune.paused', { reason: 'usage limit' }))
  expect(v.status).toBe('paused')
  expect(v.line).toBe('paused: usage limit')
  v = fold(v, ev(2, 'tune.resume', {}))
  expect(v.status).toBe('running')
})

test('lines keeps a cut line for the next piece and drops what is not an event', () => {
  const text = STREAM.slice(0, 2).map(e => JSON.stringify(e)).join('\n') + '\n'
  const a = lines('', text.slice(0, 30))
  expect(a.events).toEqual([])
  const b = lines(a.held, text.slice(30) + 'not json\n{"no": "type"}\n')
  expect(b.events.map(e => e.seq)).toEqual([1, 2])
  expect(b.held).toBe('')
})

test('band reads as status --line does', () => {
  let v = begin(TUNE)
  for (const e of STREAM.slice(0, 4)) v = fold(v, e)
  expect(band(v, Date.parse('2026-01-15T10:44:30-07:00'))).toBe(`harness ${TUNE}  base 1/3  0.25 of 6.00 USD  23 min`)
  expect(band(IDLE, 0)).toBe(IDLE.line)
})

test('planLines shows sessions, cost, time, warnings and failed checks', () => {
  const got = planLines({ plan: { stages: ['base', 'probe'], sessions: { executor: 15, editor: 2, judge: 0 },
    cost: { point: 2.72, low: 1.72, high: 6.38 }, time: { worst_s: 5400 }, df: 3, caps: { usd: 10 },
    warnings: ['a warning'], preflight: [{ check: 'claude', ok: false, line: 'claude is not on PATH' }] } })
  expect(got.point).toBe(2.72)
  expect(got.cap).toBe(10)
  expect(got.lines).toEqual(['stages: base, probe', 'sessions: 15 executor, 2 editor',
    'cost: 2.72 USD (1.72 to 6.38), cap 10.00', 'worst-case time: 90 min, residual df 3', 'warning: a warning',
    'failed check: claude is not on PATH'])
})

const SCHEMA = {
  type: 'object',
  properties: {
    schema: { const: 'loopmath.harness-tune/0.1' },
    task: { type: 'object', properties: { prompt: { type: 'string', default: '', description: 'the task text' } } },
    score: { type: 'object', properties: { kind: { type: 'string', enum: ['auto', 'pass'], default: 'auto',
      description: 'how the result reads' } } },
    executor: { type: 'object', properties: { model: { type: 'string', default: 'claude-haiku-5-5',
      description: 'the model', 'x-models': ['claude-opus-5-5', 'claude-sonnet-5'] } } },
    budget: { type: 'object', properties: { usd: { type: 'number', default: 10, description: 'the cap' } } },
    reflector: { type: 'object', properties: { enabled: { type: 'boolean', default: true, description: 'on' } } },
    editors: { type: 'array', items: { type: 'object', properties: {
      focus: { type: 'string', enum: ['procedure', 'domain', 'file'], default: 'procedure', description: 'focus' },
      focus_file: { type: 'string', default: '', description: 'the file a file editor edits' },
      model: { type: 'string', default: 'claude-opus-5-5', description: 'model', 'x-models': ['claude-opus-5-5'] },
      effort: { type: 'string', enum: ['low', 'high'], default: 'high', description: 'effort' },
      max_edits: { type: 'integer', default: 5, description: 'edits' },
    } } },
  },
}

test('fieldsOf reads the schema: choices from enum, boolean and x-models', () => {
  const { fields, editorFields } = fieldsOf(SCHEMA)
  expect(fields.map(f => f.key)).toEqual(['task.prompt', 'score.kind', 'executor.model', 'budget.usd',
    'reflector.enabled'])
  expect(fields.find(f => f.key === 'reflector.enabled')?.choices).toEqual(['true', 'false'])
  expect(fields.find(f => f.key === 'task.prompt')?.choices).toBe(null)
  expect(editorFields.map(f => f.name)).toEqual(['focus', 'focus_file', 'model', 'effort', 'max_edits'])
  const model = fields.find(f => f.key === 'executor.model')!
  expect(optionsFor(model, 'claude-fable-5')).toEqual(['claude-opus-5-5', 'claude-sonnet-5', 'claude-fable-5',
    'claude-haiku-5-5'])
  expect(valueAt({ editors: [{ model: 'a' }, { model: 'b' }] }, 'editors.2.model')).toBe('b')
  expect(shown([0, 100])).toBe('[0, 100]')
  expect(shown(['a', 'b'])).toBe('["a", "b"]')
})

// ------------------------------------------------------------------------------------- the module, CLI stubbed

type Run = { argv: string[] }

type StubOpts = { plan?: Record<string, unknown>; startedId?: string; errors?: unknown[]; warnings?: unknown[] }

function stubCli(on: On, calls: Run[], opts: StubOpts = {}) {
  const config: Record<string, unknown> = { task: { prompt: '' }, score: { kind: 'auto' },
    executor: { model: 'claude-haiku-5-5' }, budget: { usd: 6 }, reflector: { enabled: true },
    editors: [{ focus: 'procedure', model: 'claude-opus-5-5', effort: 'high', max_edits: 5 },
      { focus: 'domain', model: 'claude-opus-5-5', effort: 'high', max_edits: 5 }] }
  let exists = false
  const answer = (stdout: unknown, exitCode = 0) => ({ exitCode, stdout: JSON.stringify(stdout), stderr: '',
    isStdoutTruncated: false, isStderrTruncated: false })
  on('process.run', async (_$, e) => ({ value: run(e) }))
  const run = (e: { argv: readonly string[] }) => {
    const argv = [...e.argv]
    calls.push({ argv })
    if (argv[1] === '--version') return { ...answer({}), stdout: 'loopmath 0.4.0rc1\n' }
    const [verb, sub] = argv.slice(2)
    if (verb === 'schema') return answer(SCHEMA)
    if (verb === 'config' && sub === 'get') return answer({ schema: 'loopmath.harness-config/0.1',
      path: '/work/repo/loopmath-harness.toml', exists, config, errors: opts.errors ?? [] })
    if (verb === 'config' && sub === 'set') {
      const [key = '', value = ''] = argv.slice(4)
      const parts = key.split('.')
      if (parts[0] === 'editors') {
        const eds = config.editors as Record<string, unknown>[]
        const i = Number(parts[1]) - 1
        while (eds.length <= i) eds.push({ focus: 'procedure', model: 'claude-opus-5-5', effort: 'high', max_edits: 5 })
        eds[i]![parts[2] ?? ''] = value
      } else {
        const table = (config[parts[0] ?? ''] ?? {}) as Record<string, unknown>
        table[parts[1] ?? ''] = /^\d+(\.\d+)?$/.test(value) ? Number(value) : value
        config[parts[0] ?? ''] = table
      }
      exists = true
      return answer({ schema: 'loopmath.harness-config/0.1', path: 'loopmath-harness.toml', key, value, changed: true,
        warnings: opts.warnings ?? [] })
    }
    if (verb === 'config' && sub === 'unset') {
      const [key = ''] = argv.slice(4)
      const m = /^editors\.(\d+)$/.exec(key)
      if (m) (config.editors as unknown[]).splice(Number(m[1]) - 1, 1)
      return answer({ key, changed: true, warnings: [] })
    }
    if (verb === 'init') {
      exists = true
      return answer({ path: '/work/repo/loopmath-harness.toml', warnings: [] })
    }
    if (verb === 'plan') return answer(opts.plan ?? { ok: true, plan: { stages: ['base', 'probe'],
      sessions: { executor: 15 }, cost: { point: 2.38, low: 1.55, high: 5.36 }, time: { worst_s: 3600 }, df: 3,
      caps: { usd: 6 }, warnings: [], preflight: [] } })
    if (verb === 'run') return answer({ tune: opts.startedId ?? TUNE, root: '/store/tunes/x', pid: 4242, detached: true })
    if (verb === 'stop') return answer({ tune: argv[3], stop: true, now: false, line: `stop asked for ${argv[3]}` })
    if (verb === 'show') return { ...answer({}), stdout: `tune ${argv[3]}: delivered build\n` }
    if (verb === 'apply') return { ...answer({}), stdout: argv.includes('--dry-run') ? 'would write CLAUDE.md\n' : 'wrote CLAUDE.md\n' }
    return answer({ ok: false, error: `unknown ${verb}` }, 2)
  }
}

// The watch child: the stream, cut across a line, then its end.
function stubWatch(on: On, spawned: string[][], events = STREAM) {
  on('process.spawn', async function* (_$, e) {
    spawned.push([...e.argv])
    const from = Number(e.argv[e.argv.indexOf('--from') + 1] ?? 1)
    const text = events.filter(x => (x.seq ?? 0) >= from).map(x => JSON.stringify(x)).join('\n') + '\n'
    yield { stream: 'stdout' as const, text: text.slice(0, 50) }
    yield { stream: 'stdout' as const, text: text.slice(50) }
    return { value: { code: 0, signal: null } }
  })
}

const PANE = (title: string) => ({ title, isFocused: true, bodyColumns: 100, placement: 'dock' as const,
  scroll: { offset: 0, bodyRows: 40 }, view: {} })

// What the engine answers beneath the plugin in a session: the command registry, panes, toasts, the status line.
async function startSession($: Parameters<Parameters<typeof test>[1]>[0], on: On) {
  on('session.start', (_$, e) => ({ cwd: e.cwd }))
  on('command.register', (_$, e) => ({ value: { command: e.name } }))
  on('ui.open', () => ({ value: { isPlaced: true as const } }))
  on('ui.close', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.render', { component: 'AbovePrompt' }, () => null)
  await $.session.start({ cwd: '/work/repo', surface: 'terminal', isInteractive: true })
}

for (const surface of ['terminal', 'desktop'] as const) {
  test(`${surface}: /harness opens the configure pane, a field sets through config set, Plan then Start twice`,
    async ($, on) => {
      const calls: Run[] = []
      const spawned: string[][] = []
      mock.store(on)
      mock.clock(on, { now: Date.parse(T0) })
      stubCli(on, calls)
      stubWatch(on, spawned)
      await startSession($, on)
      expect(calls.some(c => c.argv.join(' ') === 'loopmath --version')).toBe(true)

      const opened = await $.command.run({ command: 'harness', args: '' })
      expect(opened.text).toContain('configure pane')
      const ui = await $.ui.mount({ plugin: 'loopmath-harness', surface, component: 'Pane',
        props: PANE('loopmath harness'), requestId: 'lm-harness-config' })
      expect((await ui.find({ key: 'task.prompt' }))?.type).toBe('Input')
      expect((await ui.find({ key: 'init' }))?.type).toBe('Button')

      await ui.input({ key: 'task.prompt', text: 'add a --json flag to the export command' })
      expect(calls.map(c => c.argv.slice(2, 6).join(' '))).toContain(
        'config set task.prompt add a --json flag to the export command')
      expect((await ui.find({ key: 'task.prompt' }))?.props.value).toBe('add a --json flag to the export command')

      await ui.press({ key: 'section-Score' })
      await ui.select({ key: 'score.kind', value: 'pass' })
      expect(calls.some(c => c.argv.slice(2).join(' ') === 'config set score.kind pass --json')).toBe(true)
      expect((await ui.find({ key: 'score.kind' }))?.props.value).toBe('pass')

      await ui.press({ key: 'section-Models' })
      expect((await ui.findAll({ type: 'Button', text: 'Remove' })).length).toBe(2)
      await ui.press({ key: 'add-editor' })
      expect(calls.some(c => c.argv.slice(2, 6).join(' ') === 'config set editors.3.focus procedure')).toBe(true)
      await ui.press({ key: 'remove-editor-1' })
      expect(calls.some(c => c.argv.slice(2, 5).join(' ') === 'config unset editors.1')).toBe(true)

      await ui.press({ key: 'plan' })
      expect(await ui.find({ text: 'cost: 2.38 USD (1.55 to 5.36), cap 6.00' })).toBeDefined()
      await ui.press({ key: 'start' })
      expect((await ui.find({ key: 'start' }))?.props.label).toBe('Spend up to 6.00 USD')
      expect(calls.some(c => c.argv[2] === 'run')).toBe(false)
      await ui.press({ key: 'start' })
      const run = calls.find(c => c.argv[2] === 'run')
      expect(run?.argv.slice(2)).toEqual(['run', '--detach', '--yes', '--json'])
      expect(spawned[0]).toEqual(['loopmath', 'harness', 'watch', TUNE, '--json', '--from', '1'])

      const progress = await $.ui.mount({ plugin: 'loopmath-harness', surface, component: 'Pane',
        props: PANE('harness progress'), requestId: 'lm-harness-progress' })
      expect(await progress.find({ text: 'delivery: build by final-vs-base-mean: beat the base' })).toBeDefined()
      expect(await progress.find({ text: /edits: 1 proposed, 1 refused, 0 kept/ })).toBeDefined()
      expect(await progress.find({ key: 'stop' })).toBeUndefined()
      await progress.press({ key: 'apply' })
      expect(calls.at(-1)?.argv.slice(2)).toEqual(['apply', TUNE, '--dry-run'])
      expect((await progress.find({ key: 'apply' }))?.props.label).toBe('Write these files')
      await progress.press({ key: 'apply' })
      expect(calls.at(-1)?.argv.slice(2)).toEqual(['apply', TUNE])
    })
}

test('a failed plan check keeps Start from arming', async ($, on) => {
  const calls: Run[] = []
  mock.store(on)
  mock.clock(on, { now: Date.parse(T0) })
  stubCli(on, calls, { plan: { ok: false, plan: { stages: [], sessions: {}, cost: { point: 1 }, time: {}, caps: { usd: 6 },
    preflight: [{ check: 'auth', ok: false, line: 'auth.token_file is not set' }] } } })
  await startSession($, on)
  await $.command.run({ command: 'harness', args: '' })
  const ui = await $.ui.mount({ plugin: 'loopmath-harness', surface: 'terminal', component: 'Pane',
    props: PANE('loopmath harness'), requestId: 'lm-harness-config' })
  await ui.press({ key: 'start' })
  expect(await ui.find({ text: 'failed check: auth.token_file is not set' })).toBeDefined()
  expect((await ui.find({ key: 'start' }))?.props.label).toBe('Start')
  await ui.press({ key: 'start' })
  expect(calls.some(c => c.argv[2] === 'run')).toBe(false)
})

// What `config get --all --json` lists under `errors`, and `config set judge.kind none --json` under `warnings`, on
// the `init` defaults.
const NO_SCORER = { key: 'score.command', reason: 'nothing scores the runs: give score.command or a judge' }

test('said reads a CLI error or warning as key: reason', () => {
  expect(said(NO_SCORER)).toBe('score.command: nothing scores the runs: give score.command or a judge')
  expect(said(['score.command', 'empty'])).toBe('score.command: empty')
  expect(said({ reason: 'no key' })).toBe('no key')
  expect(said('a line')).toBe('a line')
  expect(said({ other: 1 })).toBe('{"other":1}')
})

test('config errors and set warnings show as key: reason, never [object Object]', async ($, on) => {
  const calls: Run[] = []
  mock.store(on)
  mock.clock(on, { now: Date.parse(T0) })
  stubCli(on, calls, { errors: [NO_SCORER], warnings: [NO_SCORER] })
  await startSession($, on)
  await $.command.run({ command: 'harness', args: '' })
  const ui = await $.ui.mount({ plugin: 'loopmath-harness', surface: 'terminal', component: 'Pane',
    props: PANE('loopmath harness'), requestId: 'lm-harness-config' })
  expect(await ui.find({ text: 'score.command: nothing scores the runs: give score.command or a judge' })).toBeDefined()
  await ui.press({ key: 'section-Score' })
  await ui.select({ key: 'score.kind', value: 'pass' })
  expect(await ui.find({ text: 'score.kind = pass; score.command: nothing scores the runs: give score.command or a judge' }))
    .toBeDefined()
  expect(await ui.find({ text: /object Object/ })).toBeUndefined()
})

test('an editor row shows focus_file only while its focus is file, and sets it', async ($, on) => {
  const calls: Run[] = []
  mock.store(on)
  mock.clock(on, { now: Date.parse(T0) })
  stubCli(on, calls)
  await startSession($, on)
  await $.command.run({ command: 'harness', args: '' })
  const ui = await $.ui.mount({ plugin: 'loopmath-harness', surface: 'terminal', component: 'Pane',
    props: PANE('loopmath harness'), requestId: 'lm-harness-config' })
  await ui.press({ key: 'section-Models' })
  expect(await ui.find({ key: 'editors.1.focus_file' })).toBeUndefined()
  await ui.select({ key: 'editors.1.focus', value: 'file' })
  expect(calls.some(c => c.argv.slice(2).join(' ') === 'config set editors.1.focus file --json')).toBe(true)
  expect((await ui.find({ key: 'editors.1.focus_file' }))?.type).toBe('Input')
  expect(await ui.find({ key: 'editors.2.focus_file' })).toBeUndefined()
  await ui.input({ key: 'editors.1.focus_file', text: 'docs/style.md' })
  expect(calls.some(c => c.argv.slice(2).join(' ') === 'config set editors.1.focus_file docs/style.md --json')).toBe(true)
  expect((await ui.find({ key: 'editors.1.focus_file' }))?.props.value).toBe('docs/style.md')
})

test('/harness watch follows a tune, the band shows it, and stop, show and apply run the CLI', async ($, on) => {
  const calls: Run[] = []
  const spawned: string[][] = []
  mock.store(on)
  mock.clock(on, { now: Date.parse('2026-01-15T10:44:30-07:00') })
  stubCli(on, calls)
  stubWatch(on, spawned, STREAM.slice(0, 4))
  await startSession($, on)
  const r = await $.command.run({ command: 'harness', args: `watch ${TUNE}` })
  expect(r.text).toBe(`Following ${TUNE}.`)
  expect(spawned[0]).toEqual(['loopmath', 'harness', 'watch', TUNE, '--json', '--from', '1'])
  const bandUi = await $.ui.mount({ plugin: 'loopmath-harness', surface: 'terminal', component: 'AbovePrompt',
    props: { hasSurvey: false, isWorking: false, maxRows: 4, bodyColumns: 100, scroll: { offset: 0, bodyRows: 4 },
      view: {} } })
  expect((await bandUi.drawn())).toMatchObject({ type: 'Text' })
  expect((await bandUi.find({ type: 'Text' }))?.text).toBe(`harness ${TUNE}  base 1/3  0.25 of 6.00 USD  23 min`)

  expect((await $.command.run({ command: 'harness', args: 'stop' })).text).toBe(`stop asked for ${TUNE}`)
  expect(calls.at(-1)?.argv.slice(2)).toEqual(['stop', TUNE, '--json'])
  expect((await $.command.run({ command: 'harness', args: 'show' })).text).toBe(`tune ${TUNE}: delivered build`)
  expect((await $.command.run({ command: 'harness', args: 'apply' })).text).toBe('would write CLAUDE.md\nNothing written yet.')
  expect((await $.command.run({ command: 'harness', args: 'apply --yes' })).text).toBe('wrote CLAUDE.md')
  expect((await $.command.run({ command: 'harness', args: 'help' })).text).toContain('/harness watch [ID]')
})

// A tune whose engine is gone: three events, then no engine (no pid). `watch --json` exits 1, with the reason on
// stderr; from past its last event it prints only a `loopmath.error/1` object, over several lines.
const DEAD = 'tune-20260115-103000-abcd'
const DEAD_STREAM: HarnessEvent[] = [
  { v: 1, seq: 1, ts: '2026-01-15T10:30:00-07:00', tune: DEAD, type: 'tune.start',
    data: { plan: { stages: ['base', 'edit', 'probe'], caps: { usd: 10 } } } },
  { v: 1, seq: 2, ts: '2026-01-15T10:31:00-07:00', tune: DEAD, type: 'stage.start', stage: 'probe', data: { planned: 12 } },
  { v: 1, seq: 3, ts: '2026-01-15T10:32:00-07:00', tune: DEAD, type: 'session.start', stage: 'probe', label: 'probe-01',
    data: { role: 'executor', model: 'claude-haiku-5-5' } },
]
const NO_ENGINE = `no engine is running ${DEAD} and it has not ended; \`loopmath harness resume ${DEAD}\` continues it`
const ERROR_OBJECT = JSON.stringify({ schema: 'loopmath.error/1', ok: false, command: 'harness watch',
  error: 'loopmath harness watch stopped with exit 1', exit: 1 }, null, 1) + '\n'

test('watchEnd: an error exit marks the view dead with the error object, else stderr, else the exit', () => {
  let v = begin(DEAD)
  for (const e of DEAD_STREAM) v = fold(v, e)
  const fail = { code: 1, signal: null }
  const got = lines('', ERROR_OBJECT)
  expect(got.events).toEqual([])
  expect(watchEnd(v, fail, got.other, '').dead).toBe('loopmath harness watch stopped with exit 1')
  const dead = watchEnd(v, fail, [], `${NO_ENGINE}\n`)
  expect(dead.dead).toBe(NO_ENGINE)
  expect(watchEnd(v, fail, [], '').dead).toBe('watch ended with exit 1')
  expect(watchEnd(v, { code: 0, signal: null }, [], '')).toBe(v)
  expect(watchEnd(v, { code: null, signal: 'SIGTERM' }, [], '')).toBe(v)
  expect(watchEnd({ ...v, status: 'paused' }, fail, [], NO_ENGINE).dead).toBe(null)
  const ended = fold(v, { v: 1, seq: 4, ts: '2026-01-15T10:40:00-07:00', type: 'tune.end', data: { status: 'stopped', exit: 5 } })
  expect(watchEnd(ended, { code: 5, signal: null }, [], '')).toBe(ended)
  const at = (iso: string) => band(dead, Date.parse(iso))
  expect(at('2026-01-15T10:34:00-07:00')).toBe(`harness ${DEAD}  probe 0/12  0.00 of 10.00 USD  2 min  error: ${NO_ENGINE}`)
  expect(at('2026-01-15T11:30:00-07:00')).toBe(at('2026-01-15T10:34:00-07:00'))
  expect(fold(dead, { v: 1, seq: 4, type: 'session.end', stage: 'probe', label: 'probe-01', data: { exit: 0 } }).dead)
    .toBe(null)
})

test('a dead engine: the band and the pane show its error and stop counting, and the check starts no watch', async ($, on) => {
  const calls: Run[] = []
  const spawned: string[][] = []
  mock.store(on)
  const clock = mock.clock(on, { now: Date.parse('2026-01-15T10:34:00-07:00') })
  stubCli(on, calls)
  on('process.spawn', async function* (_$, e) {
    spawned.push([...e.argv])
    const from = Number(e.argv[e.argv.indexOf('--from') + 1] ?? 1)
    if (from <= DEAD_STREAM.length) {
      yield { stream: 'stdout' as const, text: DEAD_STREAM.filter(x => (x.seq ?? 0) >= from)
        .map(x => JSON.stringify(x) + '\n').join('') }
      yield { stream: 'stderr' as const, text: `${NO_ENGINE}\n` }
    } else {
      yield { stream: 'stdout' as const, text: ERROR_OBJECT.slice(0, 40) }
      yield { stream: 'stdout' as const, text: ERROR_OBJECT.slice(40) }
    }
    return { value: { code: 1, signal: null } }
  })
  await startSession($, on)
  await $.command.run({ command: 'harness', args: `watch ${DEAD}` })
  await clock.settle()
  const bandUi = await $.ui.mount({ plugin: 'loopmath-harness', surface: 'terminal', component: 'AbovePrompt',
    props: { hasSurvey: false, isWorking: false, maxRows: 4, bodyColumns: 100, scroll: { offset: 0, bodyRows: 4 },
      view: {} } })
  const want = `harness ${DEAD}  probe 0/12  0.00 of 10.00 USD  2 min  error: ${NO_ENGINE}`
  expect((await bandUi.find({ type: 'Text' }))?.text).toBe(want)
  const progress = await $.ui.mount({ plugin: 'loopmath-harness', surface: 'terminal', component: 'Pane',
    props: PANE('harness progress'), requestId: 'lm-harness-progress' })
  expect(await progress.find({ text: `error: ${NO_ENGINE}` })).toBeDefined()

  await clock.advance(10 * 60_000)
  await clock.settle()
  expect(spawned.length).toBe(1)
  expect((await bandUi.find({ type: 'Text' }))?.text).toBe(want)

  // /harness watch asks again: from past the last event the child prints only the error object.
  await $.command.run({ command: 'harness', args: `watch ${DEAD}` })
  await clock.settle()
  expect(spawned[1]).toEqual(['loopmath', 'harness', 'watch', DEAD, '--json', '--from', '4'])
  expect(await progress.find({ text: 'error: loopmath harness watch stopped with exit 1' })).toBeDefined()
})

test('when the watch child ends before tune.end, the 5 s check starts it again from the next seq', async ($, on) => {
  const calls: Run[] = []
  const spawned: string[][] = []
  mock.store(on)
  const clock = mock.clock(on, { now: Date.parse(T0) })
  stubCli(on, calls)
  let first = true
  on('process.spawn', async function* (_$, e) {
    spawned.push([...e.argv])
    const from = Number(e.argv[e.argv.indexOf('--from') + 1] ?? 1)
    const upto = first ? 4 : STREAM.length
    first = false
    yield { stream: 'stdout' as const, text: STREAM.filter(x => (x.seq ?? 0) >= from && (x.seq ?? 0) <= upto)
      .map(x => JSON.stringify(x) + '\n').join('') }
    return { value: { code: 0, signal: null } }
  })
  await startSession($, on)
  await $.command.run({ command: 'harness', args: `watch ${TUNE}` })
  await clock.settle()
  expect(spawned.length).toBe(1)
  await clock.advance(5000)
  await clock.settle()
  expect(spawned[1]).toEqual(['loopmath', 'harness', 'watch', TUNE, '--json', '--from', '5'])
  const progress = await $.ui.mount({ plugin: 'loopmath-harness', surface: 'terminal', component: 'Pane',
    props: PANE('harness progress'), requestId: 'lm-harness-progress' })
  expect(await progress.find({ text: 'ended delivered (exit 0)' })).toBeDefined()
  await clock.advance(5000)
  await clock.settle()
  expect(spawned.length).toBe(2)
})
