// The pure half of the mod: the event fold, the band line, the plan summary and the configure pane's fields.
// Nothing here touches `$`, so `claude plugin test` checks it without a tune.

import type { ConfigView, EffectRow, FieldSpec, SessionRow, StageRow, TuneView } from '../types'

// One line of `loopmath harness watch ID --json`.
export type HarnessEvent = {
  v?: number
  seq?: number
  ts?: string
  tune?: string
  type?: string
  stage?: string | null
  label?: string | null
  data?: Record<string, unknown>
}

export const IDLE: TuneView = {
  id: null,
  seq: 0,
  status: 'idle',
  stage: '',
  stages: [],
  sessions: [],
  edits: { proposed: 0, refused: 0, kept: 0 },
  effects: null,
  delivery: null,
  spent: 0,
  cap: null,
  started: null,
  latest: null,
  ended: false,
  exit: null,
  dead: null,
  line: 'No tune yet. /harness to configure, plan and start one.',
  applyArmed: false,
  note: null,
}

export const UNLOADED: ConfigView = {
  loaded: false,
  exists: false,
  path: null,
  fields: [],
  editorFields: [],
  config: {},
  errors: [],
  section: 'Task',
  plan: null,
  planOk: false,
  planUsd: null,
  capUsd: null,
  startArmed: false,
  note: null,
}

// The configure pane's sections, each a list of config tables.
export const SECTIONS: ReadonlyArray<readonly [string, readonly string[]]> = [
  ['Task', ['task']],
  ['Score', ['score', 'judge']],
  ['Loop', ['base', 'signal', 'edit', 'final', 'check', 'deliver', 'reflector', 'library', 'surface', 'loop']],
  ['Models', ['executor', 'editors']],
  ['Screen', ['screen', 'fit']],
  ['Budget', ['budget']],
  ['Run', ['run', 'auth']],
]

// The view of a tune just started (or picked up by id).
export function begin(id: string): TuneView {
  return { ...IDLE, id, status: 'running', line: `${id} started` }
}

const num = (x: unknown): number | null => (typeof x === 'number' && Number.isFinite(x) ? x : null)
const str = (x: unknown, or = ''): string => (typeof x === 'string' ? x : or)
const usd = (x: number | null): string => (x === null ? '?' : x.toFixed(2))

function withStage(stages: StageRow[], name: string, state: StageRow['state'], planned?: number | null): StageRow[] {
  const hit = stages.some(s => s.name === name)
  const rows = hit ? stages : [...stages, { name, state: 'pending' as const, planned: null }]
  return rows.map(s => (s.name === name ? { ...s, state, planned: planned === undefined ? s.planned : planned } : s))
}

function withSession(rows: SessionRow[], label: string, change: Partial<SessionRow>): SessionRow[] {
  const hit = rows.some(s => s.label === label)
  const base: SessionRow = { label, stage: '', role: '', model: '', state: 'running', exit: null, cost: null }
  const all = hit ? rows : [...rows, base]
  return all.map(s => (s.label === label ? { ...s, ...change } : s))
}

// One event onto the view, as `loopmath harness status` folds the same stream. An event already folded (its seq
// at or under the view's) changes nothing, so a restarted feed may overlap.
export function fold(v: TuneView, ev: HarnessEvent): TuneView {
  const seq = num(ev.seq)
  if (seq !== null && seq <= v.seq) return v
  const d = ev.data ?? {}
  const stage = str(ev.stage, v.stage)
  const label = str(ev.label)
  let w: TuneView = { ...v, seq: seq ?? v.seq, id: v.id ?? str(ev.tune) ?? null, latest: ev.ts ?? v.latest, dead: null }
  switch (ev.type) {
    case 'tune.start': {
      const plan = (d.plan ?? {}) as Record<string, unknown>
      const names = Array.isArray(plan.stages) ? plan.stages.map(s => String(s)) : []
      const caps = (plan.caps ?? {}) as Record<string, unknown>
      w = { ...w, status: 'running', started: ev.ts ?? w.started, cap: num(caps.usd) ?? w.cap, ended: false }
      w.stages = names.reduce((rows, n) => (rows.some(s => s.name === n) ? rows : [...rows,
        { name: n, state: 'pending' as const, planned: null }]), w.stages)
      w.line = 'started'
      break
    }
    case 'stage.start':
      w.stage = stage
      w.stages = withStage(w.stages, stage, 'running', num(d.planned))
      if (w.status === 'paused') w.status = 'running'
      w.line = `${stage} started`
      break
    case 'stage.end':
      w.stages = withStage(w.stages, stage, 'done')
      w.line = `${stage} done`
      break
    case 'session.start':
      w.sessions = withSession(w.sessions, label, { stage, role: str(d.role), model: str(d.model), state: 'running' })
      if (w.status === 'paused') w.status = 'running'
      w.line = `${label} running`
      break
    case 'session.end': {
      const cost = num(d.cost_usd)
      w.sessions = withSession(w.sessions, label, { stage, state: 'done', exit: num(d.exit), cost,
        ...(d.role ? { role: str(d.role) } : {}), ...(d.model ? { model: str(d.model) } : {}) })
      if (cost !== null) w.spent = Math.round((w.spent + cost) * 1e6) / 1e6
      w.line = `${label} ended, exit ${String(d.exit ?? '?')}`
      break
    }
    case 'edit.proposed':
    case 'edit.refused':
    case 'edit.kept': {
      const kind = ev.type.split('.')[1] as 'proposed' | 'refused' | 'kept'
      w.edits = { ...w.edits, [kind]: w.edits[kind] + 1 }
      w.line = `edit ${kind}: ${str(d.id, label)}` + (kind === 'refused' && d.reason ? ` (${str(d.reason)})` : '')
      break
    }
    case 'fit': {
      const rows = Array.isArray(d.edits) ? d.edits : []
      w.effects = rows.map((r): EffectRow => {
        const o = (r ?? {}) as Record<string, unknown>
        return { id: str(o.id), estimate: num(o.estimate), se: num(o.se), kept: o.kept === true }
      })
      w.line = d.refused ? `fit refused: ${str(d.refused)}` : `fit: ${w.effects.filter(e => e.kept).length} kept`
      break
    }
    case 'deliver':
      w.delivery = `${str(d.choice)} by ${str(d.rule)}` + (d.why ? `: ${str(d.why)}` : '')
      w.line = `delivered ${str(d.choice)}`
      break
    case 'signal':
      w.line = str(d.line, 'a signal line')
      break
    case 'budget':
      if (d.what === 'usd' && num(d.cap) !== null) w.cap = num(d.cap)
      w.line = `budget ${str(d.what)}: ${usd(num(d.spent))} of ${usd(num(d.cap))}`
      break
    case 'error':
      w.line = `error: ${str(d.message, str(d.kind))}`
      break
    case 'stop.requested':
      w.line = d.now ? 'stop now asked' : 'stop asked'
      break
    case 'tune.paused':
      w.status = 'paused'
      w.line = `paused: ${str(d.reason)}`
      break
    case 'tune.resume':
      w.status = 'running'
      w.line = 'resumed'
      break
    case 'tune.end':
      w = { ...w, status: str(d.status, 'ended'), ended: true, exit: num(d.exit) }
      w.sessions = w.sessions.map(s => ({ ...s, state: 'done' as const }))
      w.line = `ended ${w.status}` + (w.exit !== null ? ` (exit ${w.exit})` : '')
      break
    default:
      break
  }
  return w
}

// The text a watch child wrote so far, cut into whole lines: the events in them, the other lines (an error object
// spans several), and what is left over.
export function lines(held: string, text: string): { events: HarnessEvent[]; other: string[]; held: string } {
  const parts = (held + text).split('\n')
  const rest = parts.pop() ?? ''
  const events: HarnessEvent[] = []
  const other: string[] = []
  for (const line of parts) {
    if (!line.trim()) continue
    try {
      const ev = JSON.parse(line) as unknown
      if (ev && typeof ev === 'object' && typeof (ev as HarnessEvent).type === 'string') {
        events.push(ev as HarnessEvent)
        continue
      }
    } catch {
      // not one JSON line
    }
    other.push(line)
  }
  return { events, other, held: rest }
}

// The view once its watch child ended. A tune that ended or paused, and a child that exited 0 or by a signal, keep
// the view (the 5 s check follows a live one again). Any other exit means the feed cannot go on: `dead` is the
// child's `loopmath.error/1` message, else its last stderr line, else its exit.
export function watchEnd(v: TuneView, end: { code: number | null; signal: string | null } | null, other: string[],
  stderr: string): TuneView {
  if (v.ended || v.status === 'paused' || end === null || end.code === null || end.code === 0) return v
  let text = ''
  try {
    const o = JSON.parse(other.join('\n')) as unknown
    if (o && typeof o === 'object' && typeof (o as Record<string, unknown>).error === 'string') {
      text = (o as Record<string, unknown>).error as string
    }
  } catch {
    // not one error object
  }
  if (!text) text = stderr.split('\n').map(s => s.trim()).filter(Boolean).at(-1) ?? ''
  if (!text) text = `watch ended with exit ${end.code}`
  return { ...v, dead: text, line: `error: ${text}` }
}

const parseTs = (ts: string | null): number | null => {
  if (!ts) return null
  const t = Date.parse(ts)
  return Number.isFinite(t) ? t : null
}

// Whole minutes from the tune's start to `nowMs` (to its last event once it ended or its engine is gone).
export function minutes(v: TuneView, nowMs: number): number | null {
  const a = parseTs(v.started)
  const b = v.ended || v.dead !== null ? parseTs(v.latest) : nowMs
  return a === null || b === null ? null : Math.max(0, Math.floor((b - a) / 60000))
}

// The band: `harness <id>  <stage> done/planned  <spent> of <cap> USD  <n> min`, as `status --line` prints it.
export function band(v: TuneView, nowMs: number): string {
  if (v.id === null) return v.line
  const s = v.stages.find(x => x.name === v.stage)
  const done = v.sessions.filter(x => x.state === 'done' && x.stage === v.stage).length
  const count = s && s.planned !== null ? ` ${done}/${s.planned}` : ''
  const m = minutes(v, nowMs)
  const tail = v.ended || v.dead !== null ? `  ${v.line}` : v.status === 'paused' ? '  paused' : ''
  return `harness ${v.id}  ${v.stage || 'start'}${count}  ${usd(v.spent)} of ${usd(v.cap)} USD` +
    (m === null ? '' : `  ${m} min`) + tail
}

// `plan --json` as the lines the configure pane shows, and its cost point and cap.
export function planLines(out: Record<string, unknown>): { lines: string[]; point: number | null; cap: number | null } {
  const p = (out.plan ?? {}) as Record<string, unknown>
  const cost = (p.cost ?? {}) as Record<string, unknown>
  const time = (p.time ?? {}) as Record<string, unknown>
  const caps = (p.caps ?? {}) as Record<string, unknown>
  const sessions = (p.sessions ?? {}) as Record<string, unknown>
  const counts = Object.entries(sessions).filter(([, n]) => typeof n === 'number' && n > 0)
    .map(([role, n]) => `${String(n)} ${role}`).join(', ')
  const worst = num(time.worst_s)
  const out1 = [
    `stages: ${Array.isArray(p.stages) ? p.stages.join(', ') : '?'}`,
    `sessions: ${counts || 'none'}`,
    `cost: ${usd(num(cost.point))} USD (${usd(num(cost.low))} to ${usd(num(cost.high))}), cap ${usd(num(caps.usd))}`,
    `worst-case time: ${worst === null ? '?' : `${Math.round(worst / 60)} min`}, residual df ${String(p.df ?? '?')}`,
  ]
  const warn = Array.isArray(p.warnings) ? p.warnings.map(w => `warning: ${String(w)}`) : []
  const checks = Array.isArray(p.preflight) ? p.preflight : []
  const failed = checks.filter(c => (c as Record<string, unknown>).ok === false)
    .map(c => `failed check: ${String((c as Record<string, unknown>).line ?? (c as Record<string, unknown>).check)}`)
  return { lines: [...out1, ...warn, ...failed], point: num(cost.point), cap: num(caps.usd) }
}

// The fields of `schema --json`: one per table key, and the per-editor ones apart.
export function fieldsOf(schema: Record<string, unknown>): { fields: FieldSpec[]; editorFields: FieldSpec[] } {
  const spec = (table: string, name: string, p: Record<string, unknown>): FieldSpec => {
    const type = Array.isArray(p.type) ? p.type.map(String).join('|') : String(p.type ?? 'string')
    const choices = Array.isArray(p.enum) ? p.enum.map(String)
      : type === 'boolean' ? ['true', 'false']
        : Array.isArray(p['x-models']) ? p['x-models'].map(String) : null
    return { key: `${table}.${name}`, name, type, description: str(p.description), default: p.default, choices }
  }
  const props = (schema.properties ?? {}) as Record<string, Record<string, unknown>>
  const fields: FieldSpec[] = []
  let editorFields: FieldSpec[] = []
  for (const [table, node] of Object.entries(props)) {
    if (table === 'schema' || !node || typeof node !== 'object') continue
    if (table === 'editors') {
      const item = ((node.items ?? {}) as Record<string, unknown>).properties as Record<string, Record<string, unknown>>
      editorFields = Object.entries(item ?? {}).map(([n, p]) => spec('editors', n, p))
      continue
    }
    const inner = (node.properties ?? {}) as Record<string, Record<string, unknown>>
    for (const [n, p] of Object.entries(inner)) fields.push(spec(table, n, p))
  }
  return { fields, editorFields }
}

// A config error or warning as one line. The CLI gives `{key, reason}` objects (`config get` errors, `config set`
// warnings): `key: reason`. A `[key, reason]` pair reads the same, and a string is itself.
export function said(x: unknown): string {
  if (Array.isArray(x)) return x.map(String).join(': ')
  if (x !== null && typeof x === 'object') {
    const o = x as Record<string, unknown>
    const reason = str(o.reason, str(o.message))
    if (reason) return typeof o.key === 'string' && o.key ? `${o.key}: ${reason}` : reason
    return JSON.stringify(x)
  }
  return String(x)
}

// A config value as the configure pane shows it and as `config set` reads it back (TOML).
export function shown(value: unknown): string {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return '[' + value.map(x => (typeof x === 'string' ? JSON.stringify(x) : String(x))).join(', ') + ']'
  return String(value)
}

// What a configure pane field reads from the effective config: `task.prompt`, or `editors.2.model`.
export function valueAt(config: Record<string, unknown>, key: string): unknown {
  let cur: unknown = config
  for (const part of key.split('.')) {
    if (cur === null || typeof cur !== 'object') return undefined
    cur = /^\d+$/.test(part) && Array.isArray(cur) ? cur[Number(part) - 1] : (cur as Record<string, unknown>)[part]
  }
  return cur
}

// The picker's options: the field's choices, then the value it holds and its default when they are not among them.
export function optionsFor(f: FieldSpec, value: unknown): string[] {
  const out: string[] = []
  for (const x of [...(f.choices ?? []), shown(value), shown(f.default)]) if (x !== '' && !out.includes(x)) out.push(x)
  return out
}
