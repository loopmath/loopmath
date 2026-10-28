// What the mod keeps in `$.state`: the live tune's view (folded from its event stream) and the configure pane.

export type StageRow = {
  name: string
  state: 'pending' | 'running' | 'done'
  planned: number | null
}

export type SessionRow = {
  label: string
  stage: string
  role: string
  model: string
  state: 'running' | 'done'
  exit: number | null
  cost: number | null
}

export type EffectRow = {
  id: string
  estimate: number | null
  se: number | null
  kept: boolean
}

export type TuneView = {
  id: string | null
  seq: number
  status: string
  stage: string
  stages: StageRow[]
  sessions: SessionRow[]
  edits: { proposed: number; refused: number; kept: number }
  effects: EffectRow[] | null
  delivery: string | null
  spent: number
  cap: number | null
  started: string | null
  latest: string | null
  ended: boolean
  exit: number | null
  // Why the feed cannot go on (the engine is gone): the watch child's error. The band stops counting.
  dead: string | null
  line: string
  applyArmed: boolean
  note: string | null
}

export type FieldSpec = {
  key: string
  name: string
  type: string
  description: string
  default: unknown
  choices: string[] | null
}

export type ConfigView = {
  loaded: boolean
  exists: boolean
  path: string | null
  fields: FieldSpec[]
  editorFields: FieldSpec[]
  config: Record<string, unknown>
  errors: string[]
  section: string
  plan: string[] | null
  planOk: boolean
  planUsd: number | null
  capUsd: number | null
  startArmed: boolean
  note: string | null
}

declare module 'claude-code' {
  interface PluginState {
    'loopmath-harness': { view: TuneView; config: ConfigView }
  }
}
