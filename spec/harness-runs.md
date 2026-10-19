# The run log `loopmath.harness-runs/0.1`

A plain format for "this harness ran this task, and here is what came out and what it cost". An auto-research loop, a plugin eval suite or a hand A/B writes one JSON line per rollout; `loopmath adapt harness-runs FILE` turns each line into an OCP v0.3 run, and `loopmath run import` stores them. loopmath's own harness2 importer writes the same rows, so a new tool joins without new code.

The JSON Schema is [`harness-runs-0.1.schema.json`](harness-runs-0.1.schema.json) (draft 2020-12). Worked examples are in [`examples/harness-runs/`](examples/harness-runs/).

## 1. The file

- JSON lines in UTF-8 without a byte order mark, one object per rollout. Lines end at a newline (`\n`) only: U+2028, U+2029 and U+0085 inside a string are text. Blank lines are skipped.
- A reader refuses the whole file on its first bad line and names the line number (`runs.jsonl:4: task_id: a non-empty string is required`). It never imports part of a file.
- Rows are independent: order does not matter, and appending is safe. `harness_runs.append` writes each line with one `os.write` on an `O_APPEND` file, so concurrent writers do not interleave. A write that stops short is cut back and raised, and a file whose last line has no newline is refused, so a torn line is never left or joined to. Every append holds the file's exclusive `flock` across the write, the fsync and any cut, so a cut never takes another writer's row; `harness_runs.locked(path)` holds that lock for a caller that reads, checks and appends as one step.

## 2. One row

```json
{"schema": "loopmath.harness-runs/0.1",
 "task_id": "ledger-1",
 "experiment": "ledger-loop-h2-5d0c7e21a9b34f68",
 "arm": "adopted",
 "status": "done",
 "harness": "auto-research/c1",
 "agent": "opencode",
 "model": "gpt-5.4",
 "outcome": {"name": "judge", "score": 1.0, "judge": "judge-model-x", "scale": "fraction"},
 "cost": {"usd": 0.0231, "tokens": {"input": 2000, "cache_read": 600, "cache_write": 10, "output": 160},
          "recovered": true, "usd_source": "opencode"},
 "started_at": "2026-01-15T09:10:00-08:00",
 "ended_at": "2026-01-15T09:12:44-08:00",
 "tool": {"name": "my-loop", "version": "0.3.0", "run_id": "loop-1/ledger-1/c1", "task_id": "ledger-1", "stage": "c1"},
 "ext": {"org.example.loop": {"editor_usd": 0.035, "selector_usd": 0.012}}}
```

| field | required | type | meaning |
|---|---|---|---|
| `schema` | yes | string | `loopmath.harness-runs/0.1`. Any other value, including `loopmath.harness-runs/0.2`, is refused by name. |
| `task_id` | yes | non-empty string | The task the rollout ran. |
| `id` | no | string | A stable row id. Default: `<tool.name>:<experiment>:<tool.run_id>` when `tool.name` and `tool.run_id` exist (`<tool.name>:<tool.run_id>` without an experiment), else SHA-256 of the row's canonical JSON. The imported run id derives from it, so importing the same row again changes nothing (section 6). |
| `task` | no | object | `type`, `repo`, `title`, `base_commit` and `source`, as OCP `run.task` has them. `type` is a loopmath task type (`loopmath task-types`: `bug_fix`, `feature`, `refactor`, `tests`, `docs`, `research`, `infra` or `data`); any other value is refused, naming the closest. A missing `type` is `research` unless `--type` says otherwise. |
| `arm` | no | string | Which side of a comparison: `base`, `candidate`, `adopted`, or any label. It is imported as `run.labels.arm`. |
| `pair` | no | string | The replicate pair key: `<task_id>#<j>`, or any stable id. Two rows with the same `pair` and different arms are one pair. |
| `experiment` | no | string | The experiment the row came from. It must be stable and **never a path**, so moving a folder keeps it: a value with `/` or `\`, one that starts with `~`, and `.` or `..` are refused. Native run ids are often unique only within one experiment, so the experiment is part of the row id. A pair is `(experiment, pair)`; a missing experiment is the null namespace. |
| `status` | no | `done` or `failed` | Whether the rollout ran to its end. Default `done`. A `failed` rollout was started and did not finish; it imports as a failed attempt, and its score and cost stay unknown unless measured. A rollout whose files are gone is not a row. |
| `harness` | no | object, string or null | The harness tree. An object is a `loopmath.harness/0.1` record, kept as given. A string is a tree path relative to the rows file, recorded at import with `harness.record_from_tree`; a path that leaves the rows file's folder is refused. Null or absent means no record. |
| `agent` | no | string or null | The agent CLI that ran the rollout: `opencode`, `codex`, `claude-code`, or any name. It becomes the OCP setting's harness. Absent means `unknown`. |
| `model` | no | string or null | The model id as the tool reported it. Absent means `unknown`, and tokens then stay unpriced. |
| `effort` | no | string or null | Reasoning effort as the tool reported it. Null means not recorded. |
| `outcome` | no | object or null | Section 3. Null or absent means unscored. |
| `cost` | no | object or null | Section 4. Null or absent means not known. |
| `started_at`, `ended_at` | no | RFC 3339 time with an offset, or null | Times as the source recorded them. Never invented. |
| `tool` | no | object | The producing tool's ids, all strings, each optional: `name`, `version`, `run_id`, `task_id`, `stage`. |
| `ext` | no | object | Producer data, keyed by reverse-domain namespace. Kept whole. |

Unknown top-level fields are kept, not refused: they land in the run's ext with the rest of the row.

## 3. `outcome`

Exactly one of these shapes, or null:

- **A score:** `{"name": "judge", "score": <number or null>, "judge": "<who scored>", "scale": "linear" | "log" | "fraction", "better": "higher" | "lower", "observed_at": "<RFC 3339>", ...}`.
  - `score` is required as a key. **null means the judge ran but gave no score.** It is never 0, and it imports as a declared signal with value null.
  - `name` defaults to `judge`, `scale` to `linear` and `better` to `higher`. Name the `judge`: the model or program that scored.
  - A `fraction` score lies in [0, 1]; a reader refuses one outside it.
  - Other members (`n_passed`, `n_criteria`, `all_pass`) are kept in the signal's ext.
- **A verdict:** `{"accepted": true | false, "by": "<who decided>", "observed_at": "<RFC 3339>"}`.
- **Unscored:** null or absent. A rollout nobody judged has no signal.

`observed_at` defaults to `ended_at`, then `started_at`, then the rows file's modification time, so the same file always gives the same documents. These fallbacks set only when the score was observed: a run's `started_at` and `ended_at` come only from the row, and a row without them imports with none.

In the store a score is an OCP score signal named `name`, with `source {kind: "judge", ref: <judge>}` and tier `reported`; compare reads it with the rule `score:<name>` (`score:<name>:lower` when lower is better). A verdict is the signal `accepted` with value `accept` or `reject`, under an acceptance rule that requires it. `--accept-at X` turns a score into an acceptance rule: accepted when the score reaches X.

## 4. `cost`

`{"usd": <number or null>, "tokens": {"input": n, "cache_read": n, "cache_write": n, "output": n} | null, "recovered": true | false, "usd_source": "...", "unpriced": "<reason>", "complete": true | false, "incomplete": "<reason>", "partial_usd": <number>, "files": n, "skipped_lines": n}`

- `usd`: dollars the rollout itself spent, or null when not known. Spend on editing or judging between rollouts is not the rollout's; put it in `ext`.
- `tokens`: the four streams that `loopmath.price` prices: `input`, `cache_read`, `cache_write` and `output`. `input` excludes cached reads and `output` includes reasoning. A missing stream counts as 0; an absent `tokens` means not known.
- A row with tokens and a null `usd` is priced at import from loopmath's price table (`usd_source: loopmath-prices`). A model with no price row keeps its tokens, gets no dollars and keeps the reason in `unpriced`. **It is never priced at 0.**
- `recovered` is true when the numbers were read back from a trajectory or stream after the fact. It is a flag, not a quality mark.
- `usd_source`: `opencode` (OpenCode's own step cost), `loopmath-prices`, `reported`, or any name.
- **Coverage:**
  - `usd` is a complete total or null, never a partial sum.
  - `complete` (default true) is false when the stream shows spend its counts miss: a step or turn that started with no usage after it. Then `usd` must be null. The importer keeps the tokens and `partial_usd` in the attempt's ext only, never as its cost.
  - `partial_usd` is the dollars a tool reported for only some billable steps. When the tokens can be priced in full, `usd` is that priced total and `partial_usd` stays as a note.
  - `incomplete` names what is missing; `unpriced` names why `usd` is null.
- **A reported zero is not an absent cost.** A step whose `cost` is the number 0 reported 0; a step with no `cost` reported nothing. OpenCode writes 0 for a model it has no price for, so a rollout whose billable steps all report 0 is priced from tokens.

### Reading costs from streams

`harness_runs.rollout_cost(paths, kind, *, model=None)` returns the `cost` object for one rollout's stream files (a list sums retries). `harness_runs.rollout_usage(paths, kind)` is the same without pricing.

- `kind="opencode"` (`opencode run --format json` lines): each `step_finish` part's `tokens {input, output, reasoning, cache {read, write}}` and `cost`. A step is billable when its tokens or cost are above 0. `usd` is OpenCode's own total only when every billable step reports a number and the total is above 0; otherwise the whole token total is priced and a reported subtotal is kept as `partial_usd`. A `step_start` with no `step_finish` after it, or a `step_finish` whose `input`, `output` or `cache {read, write}` is missing or not a whole number, makes the counts incomplete (an absent `reasoning` adds 0).
- `kind="codex"` (`codex exec --json` lines): the sum of `turn.completed.usage`, with `input = input_tokens - cached_input_tokens`, `cache_read = cached_input_tokens` and `output = output_tokens`, priced. A `turn.started` with no `turn.completed` after it, a `turn.failed`, or a `turn.completed` whose `input_tokens`, `cached_input_tokens` or `output_tokens` is missing or not a whole number, makes the counts incomplete. A missing count is never read as 0.
- Lines that do not parse are counted in `skipped_lines`, never fatal. A named stream file that cannot be read makes the counts incomplete.

## 5. Python API

```python
from loopmath import harness_runs
harness_runs.SCHEMA                          # "loopmath.harness-runs/0.1"
harness_runs.row(task_id, **fields)          # a checked row with schema set; ValueError names each bad field
harness_runs.check_row(row)                  # list of problems, empty when valid
harness_runs.append(path, row)               # one line, one checked os.write on an O_APPEND file, then fsync, under the lock
with harness_runs.locked(path) as log:       # the file's exclusive flock; other appends wait until the block ends
    log.read(); log.append(row)              # appends in the block (and append(path, row) in this thread) reuse it
harness_runs.read(path)                      # every row, checked; ValueError("<path>:<line>: ...")
harness_runs.rollout_cost(paths, kind, *, model=None)
harness_runs.rollout_usage(paths, kind)
harness_runs.to_ocp(row, *, base_dir=None, accept_at=None, observed_default=None, task_type=None, repo=None)
```

## 6. Import

```
loopmath adapt harness-runs runs.jsonl --out ocp/     # one <run id>.ocp.json per row
loopmath run import ocp/ --finish                      # store them; the same documents again change nothing
```

- Without `--out`, the documents go to stdout as JSON lines. `--json` prints the import summary as `loopmath.adapt.runs/1`: on stdout with `--out`, and on stderr without it, since stdout then carries the documents.
  - The summary has `ok`, the counts, `experiments` (each named experiment with its number of runs) and `unpriced_models` (each model with runs left unpriced, the number and the reason).
  - A refusal under `--json` prints one object on stdout, `{"schema": "loopmath.adapt.runs/1", "ok": false, "importer": ..., "error": ..., "exit": n}`, and the error line on stderr.
- `--experiment ID` fills the `experiment` of rows that have none. `--type` and `--repo` fill missing task fields. A `--type` that is not a loopmath task type, or an `--experiment` that is a path, exits 2.
- Nothing is written until every row has made a valid run. A bad file exits 1 and writes nothing; a bad row is named by `<file>:<line>`, a harness tree error too. An empty or blank file exits 1. A missing input exits 2.
- Importing a run again: `run import` keeps a record of the content it stored (`run.ext["dev.loopmath.import"].content`).
  - The same content again changes nothing: the run, its finish state and the outcomes added since all stay, and the summary counts it `unchanged`. `--finish` still finishes such a run if it is open, as `run finish` would.
  - Different content under a stored run id is refused by name, with the fields that differ, and the command exits 1. The other runs import as usual.
  - An open run that `loopmath run start` made and `run import` never stored takes its first import as it is (an orchestrator writing the document for a run loopmath started). That import records its content even when the document is the one the run holds. From then on it holds imported content like any other.
  - A second harness2 runs folder with the same config makes the same run ids: adapt it with `--experiment` to keep both.
  - `--replace` takes the new copy and keeps the outcomes added after the import. It never overwrites one: a new copy that holds an outcome's id with another value is refused, naming the id. The record keeps the ids of the outcomes added after the import (`later`), so each stays kept through every later replace, even after a copy that carried it equal. A run stored before loopmath kept that record cannot tell them from imported ones; the outcomes it drops are counted.
- Each row becomes one OCP v0.3 run with one attempt, configured as the catalog `solo` workflow with the row's `agent`, `model` and `effort`, so the run lands in the same configuration a `loopmath run start` with those settings would.
  - The harness record goes to `run.ext["dev.loopmath.harness"]`.
  - `arm` goes to `run.labels.arm`.
  - `pair` and `experiment` go to `run.ext["dev.loopmath.pair"]` as `{"pair": <pair or null>, "arm": <arm or null>, "experiment": <experiment or null>}`, written when the row has either. A null `pair` there means no pair key: runs pair on the task within the experiment.
  - `status: failed` gives a failed attempt.
  - The rest of the row is kept at `run.ext["dev.loopmath.adapter.harness-runs"]`.
  - The document's `privacy.profile` is `metadata_only` unless the run keeps text the row's writer chose: a string in an unknown row field, in an `ext` namespace not under `dev.loopmath.`, or in an `outcome` field this spec does not define. Then it is `full`, and `retained_text` in the kept row lists where. loopmath cannot tell a label from a prompt, so it never promises that such text is not one.

`loopmath adapt harness2 RUNS --out DIR` reads a harness2 runs folder into these rows first (`--rows FILE` keeps them), then imports them the same way; see `docs/adapters.md`.

## 7. Worked examples

Each file in [`examples/harness-runs/`](examples/harness-runs/) imports with `loopmath adapt harness-runs`.

1. **`auto-research-loop.jsonl`: an auto-research loop.** One task, one experiment. Two base rollouts on the base tree, a candidate edit that was not adopted (its OpenCode cost covered only one step, so the tokens are priced and the reported subtotal kept as `partial_usd`), a candidate that died mid-step (`status: failed`, `complete: false`), and the adopted compose with its judge score. Trees are paths under `auto-research/`. The loop's own editing and selection spend is in `ext`, not in `cost`.
2. **`plugin-evals.jsonl`: plugin evals.** Three tasks, each run once without a plugin and once with `lint@acme`, as replicate pairs `<task>#1` in one experiment. The harness records are inline, naming the factor `plugin:lint@acme`. Outcomes are `accepted` verdicts from the test suite, costs as the tool reported them.
3. **`hand-ab.jsonl`: a hand A/B.** Two CLAUDE.md variants under `hand-ab/` on three tasks, with no experiment and no pair, so compare pairs the runs by task. Outcomes are the person's own accept or reject; costs are dollars only.

## 8. Versions

- A reader refuses any `schema` other than `loopmath.harness-runs/0.1`, by name.
- Changes inside 0.1 are additive only: a new optional field or a new value. Readers keep unknown fields.
