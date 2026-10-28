import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { ConfigView, FieldSpec, TuneView } from '../types'
import { IDLE, SECTIONS, UNLOADED, band, begin, fieldsOf, fold, lines, optionsFor, planLines, said, shown, valueAt,
  watchEnd } from './events'

// The engine is the loopmath CLI. This module runs it by argv and folds its event stream
// (`loopmath harness watch ID --json`); it never starts a Claude turn or calls a model.
// The tune runs detached (`run --detach`), so closing the session or reloading the module leaves it running,
// and a check every 5 s restarts the feed where the view stopped.
const CONFIG_PANE = 'lm-harness-config'
const PROGRESS_PANE = 'lm-harness-progress'
// An editor's row; `focus_file` only while its focus is `file`.
const EDITOR_ROW = ['focus', 'focus_file', 'model', 'effort', 'max_edits']
const view = atom({ plugin: 'loopmath-harness', key: 'view' } as const, IDLE)
const conf = atom({ plugin: 'loopmath-harness', key: 'config' } as const, UNLOADED)

let BIN = 'loopmath'
let CONFIG_PATH = ''

type Out = { ok: boolean; json: Record<string, unknown> | null; text: string; exit: number | null }

const configArgs = (): string[] => (CONFIG_PATH ? ['--config', CONFIG_PATH] : [])

// One CLI call with --json: its JSON (the error object too, on a failure), and a line saying what happened.
async function cli($: EngineInterface, args: string[], timeoutMs = 30_000): Promise<Out> {
  try {
    const r = await $.process.run([BIN, 'harness', ...args, '--json'], { timeoutMs })
    let json: Record<string, unknown> | null = null
    try {
      json = JSON.parse(r.stdout) as Record<string, unknown>
    } catch {
      json = null
    }
    const said = json && typeof json.error === 'string' ? json.error : (r.stderr || r.stdout).trim().split('\n')[0]
    return { ok: r.exitCode === 0, json, text: r.exitCode === 0 ? '' : said || `exit ${r.exitCode}`, exit: r.exitCode }
  } catch {
    return { ok: false, json: null, text: `${BIN} did not run (not on PATH, or no answer in time)`, exit: null }
  }
}

// One CLI call for people: its text output, cut to `max` lines.
async function cliText($: EngineInterface, args: string[], max = 40): Promise<{ ok: boolean; text: string }> {
  try {
    const r = await $.process.run([BIN, 'harness', ...args], { timeoutMs: 60_000 })
    const all = (r.exitCode === 0 ? r.stdout : r.stderr || r.stdout).trimEnd().split('\n')
    const cut = all.length > max ? [...all.slice(0, max), `(${all.length - max} more lines)`] : all
    return { ok: r.exitCode === 0, text: cut.join('\n') }
  } catch {
    return { ok: false, text: `${BIN} did not run` }
  }
}

// ---------------------------------------------------------------------------------------------------- the feed

let feeding: string | null = null
let ticking = false

// Follows one tune's stream from the view's last seq; ends with the watch child (the tune's end, or an unload).
// A child that exits with an error (no engine runs the tune) marks the view dead with its message.
function feed($: EngineInterface, id: string) {
  if (feeding === id) return
  feeding = id
  void (async () => {
    try {
      const from = (await read($, view)).seq + 1
      const child = $.process.spawn({ argv: [BIN, 'harness', 'watch', id, '--json', '--from', String(from)] })
      let held = ''
      let stderr = ''
      const other: string[] = []
      for await (const piece of child) {
        if (piece.stream !== 'stdout') {
          stderr = (stderr + piece.text).slice(-4000)
          continue
        }
        const got = lines(held, piece.text)
        held = got.held
        other.push(...got.other)
        if (other.length > 200) other.splice(0, other.length - 200)
        for (const ev of got.events) await update($, view, v => fold(v, ev))
        if (got.events.length > 0) await mirror($)
      }
      if (held.trim()) other.push(held)
      const end = await child.result
      await update($, view, v => (v.id === id ? watchEnd(v, end, other, stderr) : v))
    } catch {
      // the check restarts it
    } finally {
      if (feeding === id) feeding = null
    }
    await mirror($).catch(() => undefined)
  })()
}

// The last seq in the store and the band on the status line; the end once, as a toast.
async function mirror($: EngineInterface) {
  const v = await read($, view)
  if (v.id === null) return
  await $.store.set('seq', v.seq)
  if (v.ended) {
    if ((await $.store.get('told')) !== v.id) {
      await $.store.set('told', v.id)
      $.ui.toast(`harness ${v.id} ${v.line}`)
    }
    $.ui.status(undefined)
    return
  }
  $.ui.status(band(v, await $.clock.now()))
}

// The 5 s check: a live tune with no feed gets one again; an ended or dead one does not (`/harness watch` asks
// again). A check that fails (the module going away) waits for the next.
async function check($: EngineInterface) {
  try {
    const id = (await $.store.get('tune')) as string | undefined
    if (!id) return
    const v = await read($, view)
    if (v.id !== id) await update($, view, () => begin(id))
    else if (v.ended || v.dead !== null) return
    feed($, id)
  } catch {
    // the next check tries again
  }
}

async function follow($: EngineInterface, id: string) {
  const v = await read($, view)
  if (v.id !== id) {
    await update($, view, () => begin(id))
    await $.store.set('seq', 0)
  } else if (v.dead !== null) {
    await update($, view, w => ({ ...w, dead: null, line: `following ${id} again` }))
  }
  await $.store.set('tune', id)
  feed($, id)
}

// ---------------------------------------------------------------------------------------------------- the config

async function load($: EngineInterface, note: string | null = null) {
  const [schema, got] = await Promise.all([cli($, ['schema']), cli($, ['config', 'get', '--all', ...configArgs()])])
  if (!schema.json || !got.json || got.json.ok === false) {
    await update($, conf, c => ({ ...c, loaded: false, note: `could not read the config: ${got.text || schema.text}` }))
    return
  }
  const { fields, editorFields } = fieldsOf(schema.json)
  const errors = Array.isArray(got.json.errors) ? got.json.errors.map(said) : []
  await update($, conf, c => ({
    ...c, loaded: true, exists: got.json?.exists === true, path: String(got.json?.path ?? ''), fields, editorFields,
    config: (got.json?.config ?? {}) as Record<string, unknown>, errors, note,
  }))
}

// `config set` (or `unset` for an empty value), then the file read back; a change drops the plan.
async function setKey($: EngineInterface, key: string, value: string) {
  const unset = value.trim() === ''
  const r = await cli($, unset ? ['config', 'unset', key, ...configArgs()] : ['config', 'set', key, value, ...configArgs()])
  const warn = r.json && Array.isArray(r.json.warnings) && r.json.warnings.length
    ? `; ${r.json.warnings.map(said).join('; ')}` : ''
  const note = r.ok ? (unset ? `${key} back to its default` : `${key} = ${shown(r.json?.value)}`) + warn
    : `${key} not set: ${r.text}`
  await update($, conf, c => ({ ...c, plan: null, planOk: false, planUsd: null, startArmed: false }))
  await load($, note)
}

async function init($: EngineInterface) {
  const r = await cli($, ['init', ...(CONFIG_PATH ? ['--out', CONFIG_PATH] : [])])
  await load($, r.ok ? `wrote ${String(r.json?.path ?? 'the config')}` : `init refused: ${r.text}`)
}

async function plan($: EngineInterface): Promise<boolean> {
  await update($, conf, c => ({ ...c, note: 'planning (no model call)...' }))
  const r = await cli($, ['plan', ...configArgs()], 120_000)
  const body = r.json && typeof r.json.plan === 'object' && r.json.plan !== null ? planLines(r.json) : null
  if (!body) {
    await update($, conf, c => ({ ...c, plan: null, planOk: false, planUsd: null, startArmed: false,
      note: `plan refused: ${r.text}` }))
    return false
  }
  const ok = r.ok && r.json?.ok !== false
  await update($, conf, c => ({ ...c, plan: body.lines, planOk: ok, planUsd: body.point, capUsd: body.cap,
    startArmed: false,
    note: ok ? 'planned: Start asks once more before it spends' : 'the plan has a failed check: fix it, then plan again' }))
  return ok
}

// Start: the first press shows the cap on the button, the second starts the tune.
async function start($: EngineInterface) {
  const c = await read($, conf)
  if (!c.startArmed) {
    if (!(c.plan === null ? await plan($) : c.planOk)) return
    await update($, conf, w => ({ ...w, startArmed: true }))
    return
  }
  await update($, conf, w => ({ ...w, startArmed: false, note: 'starting...' }))
  const r = await cli($, ['run', '--detach', '--yes', ...configArgs()], 120_000)
  const id = r.ok && r.json ? String(r.json.tune ?? '') : ''
  if (!id) {
    await update($, conf, w => ({ ...w, note: `start refused: ${r.text}` }))
    return
  }
  await update($, conf, w => ({ ...w, note: `started ${id}` }))
  await follow($, id)
  await $.ui.open({ id: PROGRESS_PANE, title: 'harness progress' })
}

// ---------------------------------------------------------------------------------------------------- the verbs

async function stop($: EngineInterface, id: string | null, now = false): Promise<string> {
  if (!id) return 'No tune to stop.'
  const r = await cli($, ['stop', id, ...(now ? ['--now'] : [])])
  return r.ok ? String(r.json?.line ?? `stop asked for ${id}`) : `stop refused: ${r.text}`
}

async function show($: EngineInterface, id: string | null): Promise<string> {
  if (!id) return 'No tune to show.'
  return (await cliText($, ['show', id])).text
}

// Apply: a dry run first; `write` writes the files (committing nothing).
async function apply($: EngineInterface, id: string | null, write: boolean): Promise<string> {
  if (!id) return 'No tune to apply.'
  const r = await cliText($, write ? ['apply', id] : ['apply', id, '--dry-run'])
  if (!r.ok) return `apply refused: ${r.text}`
  return write ? r.text : `${r.text}\nNothing written yet.`
}

async function tuneId($: EngineInterface, given?: string): Promise<string | null> {
  if (given) return given
  const v = await read($, view)
  return v.id ?? (((await $.store.get('tune')) as string | undefined) ?? null)
}

const HELP = [
  '/harness             configure, plan and start a tune',
  '/harness watch [ID]  follow a tune in the progress pane',
  '/harness stop [ID]   ask the tune to stop (--now kills its sessions)',
  '/harness show [ID]   what the tune delivered and why',
  '/harness apply [ID]  the files apply would write; --yes writes them',
].join('\n')

export const register: Register = (on, options) => {
  BIN = typeof options.loopmath_bin === 'string' && options.loopmath_bin ? options.loopmath_bin : 'loopmath'
  CONFIG_PATH = typeof options.config_path === 'string' ? options.config_path : ''

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'harness',
      description: 'loopmath harness: configure, plan, start, watch and stop a tune',
      argumentHint: '[watch|stop|show|apply] [ID]',
    })
    void (async () => {
      try {
        const r = await $.process.run([BIN, '--version'], { timeoutMs: 30_000 })
        if (r.exitCode !== 0) throw new Error('no version')
      } catch {
        $.ui.toast(`loopmath was not found: pip install loopmath, or set loopmath_bin`, { timeoutMs: 8000 })
      }
    })()
    if (!ticking) {
      ticking = true
      $.clock.every(5000, () => void check($))
    }
    void check($)
    return next(e)
  })

  on('command.run', { command: 'harness' }, async ($, e) => {
    const words = (e.args ?? '').trim().split(/\s+/).filter(Boolean)
    const flags = words.filter(w => w.startsWith('--'))
    const [verb = '', given] = words.filter(w => !w.startsWith('--'))
    if (verb === '') {
      await $.ui.open({ id: CONFIG_PANE, title: 'loopmath harness' })
      await load($)
      return { text: 'loopmath harness: the configure pane is open.' }
    }
    if (verb === 'watch') {
      const id = await tuneId($, given)
      if (!id) return { text: 'No tune to watch: give its id (loopmath harness list shows them).' }
      await follow($, id)
      await $.ui.open({ id: PROGRESS_PANE, title: 'harness progress' })
      return { text: `Following ${id}.` }
    }
    if (verb === 'stop') return { text: await stop($, await tuneId($, given), flags.includes('--now')) }
    if (verb === 'show') return { text: await show($, await tuneId($, given)) }
    if (verb === 'apply') return { text: await apply($, await tuneId($, given), flags.includes('--yes')) }
    return { text: HELP }
  })

  on('ui.render', { component: 'Pane', requestId: CONFIG_PANE }, async ($, e) => {
    const { Box, Button, Input, Select, Text } = $.ui.resolve(e)
    const c: ConfigView = await read($, conf)
    const mobile = e.surface === 'mobile'
    if (!c.loaded) {
      return (
        <Box flexDirection="column">
          <Text>{c.note ?? 'Reading the config...'}</Text>
          <Button key="reload" label="Read again" onPress={() => void load($)} />
        </Box>
      )
    }
    const control = (f: FieldSpec, key: string) => {
      const value = valueAt(c.config, key)
      if (mobile || f.choices === null) {
        if (mobile) return <Text key={key}>{f.name}: {shown(value)}</Text>
        return <Input key={key} label={`${f.name} `} value={shown(value)} placeholder={shown(f.default)}
          onSubmit={(text: string) => void setKey($, key, text)} />
      }
      return <Select key={key} label={`${f.name} `} value={shown(value)}
        options={optionsFor(f, value).map(o => ({ value: o }))} onSelect={(pick: string) => void setKey($, key, pick)} />
    }
    const tables = SECTIONS.find(([name]) => name === c.section)?.[1] ?? []
    const fields = c.fields.filter(f => tables.includes(f.key.split('.')[0] ?? ''))
    const editors = Array.isArray(c.config.editors) ? c.config.editors : []
    const rows = editors.map((_, i) => (
      <Box key={`editor-${i + 1}`} flexDirection="column">
        <Text bold>editor {i + 1}: {shown(valueAt(c.config, `editors.${i + 1}.name`)) || shown(valueAt(c.config, `editors.${i + 1}.focus`))}</Text>
        {c.editorFields.filter(f => EDITOR_ROW.includes(f.name) &&
          (f.name !== 'focus_file' || valueAt(c.config, `editors.${i + 1}.focus`) === 'file'))
          .map(f => control(f, `editors.${i + 1}.${f.name}`))}
        {!mobile && <Button key={`remove-editor-${i + 1}`} label="Remove" dimColor
          onPress={() => void setKey($, `editors.${i + 1}`, '')} />}
      </Box>
    ))
    const cap = c.capUsd ?? (valueAt(c.config, 'budget.usd') as number | undefined) ?? null
    return (
      <Box flexDirection="column">
        <Text dimColor>{c.exists ? c.path : `${c.path ?? 'loopmath-harness.toml'} (not written yet: Init writes it)`}</Text>
        <Box>
          {SECTIONS.map(([name]) => (
            <Button key={`section-${name}`} label={name} variant={name === c.section ? 'primary' : undefined}
              onPress={() => void update($, conf, w => ({ ...w, section: name }))} />
          ))}
        </Box>
        {fields.map(f => (
          <Box key={`row-${f.key}`} flexDirection="column">
            {control(f, f.key)}
            <Text dimColor>  {f.description}</Text>
          </Box>
        ))}
        {tables.includes('editors') && rows}
        {tables.includes('editors') && !mobile && <Button key="add-editor" label="Add an editor"
          onPress={() => void setKey($, `editors.${editors.length + 1}.focus`, 'procedure')} />}
        {c.errors.map((err, i) => <Text key={`error-${i}`} color="red">{err}</Text>)}
        {c.plan !== null && c.plan.map((line, i) => <Text key={`plan-${i}`} dimColor>{line}</Text>)}
        {c.note !== null && <Text italic>{c.note}</Text>}
        <Box>
          {!c.exists && <Button key="init" label="Init" onPress={() => void init($)} />}
          <Button key="plan" label="Plan" onPress={() => void plan($)} />
          <Button key="start" variant="primary"
            label={c.startArmed ? `Spend up to ${cap === null ? '?' : Number(cap).toFixed(2)} USD` : 'Start'}
            onPress={() => void start($)} />
          <Button key="close" label="Close" role="dismiss" onPress={() => void $.ui.close({ id: CONFIG_PANE })} />
        </Box>
      </Box>
    )
  })

  on('ui.render', { component: 'Pane', requestId: PROGRESS_PANE }, async ($, e) => {
    const { Box, Button, Text } = $.ui.resolve(e)
    const v: TuneView = await read($, view)
    const now = await $.clock.now()
    if (v.id === null) return <Text>{IDLE.line}</Text>
    const mark = (state: string) => (state === 'done' ? '[x]' : state === 'running' ? '[>]' : '[ ]')
    const recent = v.sessions.slice(-12)
    const pad = (s: string, n: number) => (s.length >= n ? s.slice(0, n) : s.padEnd(n))
    const id = v.id
    const live = !v.ended
    const delivered = v.ended && (v.status === 'delivered' || v.status === 'base' || v.status === 'no-signal')
    return (
      <Box flexDirection="column">
        <Text bold>{band(v, now)}</Text>
        <Box>
          {v.stages.map(s => <Text key={`stage-${s.name}`}>{mark(s.state)} {s.name}  </Text>)}
        </Box>
        {recent.length > 0 && <Text dimColor>{pad('session', 22)}{pad('role', 11)}{pad('model', 20)}{pad('state', 7)}cost</Text>}
        {recent.map(s => (
          <Text key={`session-${s.label}`} mono>
            {pad(s.label, 22)}{pad(s.role, 11)}{pad(s.model, 20)}{pad(s.state, 7)}{s.cost === null ? '' : s.cost.toFixed(2)}
          </Text>
        ))}
        <Text>edits: {v.edits.proposed} proposed, {v.edits.refused} refused, {v.edits.kept} kept</Text>
        {v.effects !== null && v.effects.map(x => (
          <Text key={`effect-${x.id}`} mono>
            {pad(x.id, 28)}{x.estimate === null ? '?' : x.estimate.toFixed(2)} (se {x.se === null ? '?' : x.se.toFixed(2)}){x.kept ? '  kept' : ''}
          </Text>
        ))}
        {v.delivery !== null && <Text>delivery: {v.delivery}</Text>}
        <Text italic>{v.line}</Text>
        {v.note !== null && v.note.split('\n').map((line, i) => <Text key={`note-${i}`} dimColor>{line}</Text>)}
        <Box>
          {live && <Button key="stop" label="Stop" onPress={async () => {
            const said = await stop($, id)
            await update($, view, w => ({ ...w, note: said }))
          }} />}
          <Button key="show" label="Show" onPress={async () => {
            const said = await show($, id)
            await update($, view, w => ({ ...w, note: said }))
          }} />
          {delivered && <Button key="apply" label={v.applyArmed ? 'Write these files' : 'Apply'} onPress={async () => {
            const write = v.applyArmed
            const said = await apply($, id, write)
            await update($, view, w => ({ ...w, note: said, applyArmed: !write && !said.startsWith('apply refused') }))
          }} />}
        </Box>
      </Box>
    )
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const v = await read($, view)
    if (e.props.hasSurvey || v.id === null || v.ended) return next(e)
    const { Text } = $.ui.resolve(e)
    return <Text dimColor wrap="truncate">{band(v, await $.clock.now())}</Text>
  })
}
