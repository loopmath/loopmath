# loopmath reference for agents

Shared by every loopmath skill; `loopmath skill install` puts a copy next to each `SKILL.md`, and `loopmath skill show reference` prints it. Every command, flag and JSON field the skills use is here. Never run loopmath or any of its commands with `--help`: if something is missing here, run `loopmath doctor --json` and tell the user.

## Conventions

- `--json` prints exactly one JSON object on stdout, `schema` first. Progress and notes go to stderr. Read stdout only, as printed: do not pipe it into another program, which needs its own permission.
- `--html` writes a page and prints its path as the only line on stdout. With `--json` too, the path goes to stderr, and the JSON object carries it as `page`.
- Store: `$LOOPMATH_HOME`, default `~/.loopmath`. Never edit files there by hand.
- Exit codes: 0 ok; 1 bad arguments or validation failure (the message names what to fix); 2 not found; 4 store locked for more than 30 s (wait a minute, retry once); 5 no usable fit: none yet, or one an older loopmath made (run `loopmath fit --json`, then retry once).
- Money is in dollars, with tokens beside it. JSON ranges are `{mean, lo, hi}`, 80 percent intervals.
- Show the user plain numbers: dollars with 2 decimals (4 under $0.10), chances as whole percents, ranges as `lo to hi`.

## Task types and features

Types (`--type`): `bug_fix`, `feature`, `refactor`, `tests`, `docs`, `research`, `infra`, `data`.

Features (`--feature K=V`, repeatable; leave out what you cannot tell):

| Key | Values |
|---|---|
| `size` | `xs` (under 20 changed lines), `s` (under 100), `m` (under 400), `l` (under 1500), `xl` |
| `lang` | the main language, lowercase: `python`, `typescript`, `rust`, ... |
| `has_tests` | `yes`, `no`: the touched code has tests that can judge the change |
| `spec_clarity` | `clear`, `partial`, `vague` |
| `needs_design` | `yes`, `no` |
| `touches` | `one` (1 file), `few` (2 to 5), `many` |

Repo (`--repo`): `owner/name` from the git `origin` remote, lowercased (`git@github.com:Acme/API.git` is `acme/api`); with no origin, the name of the git top-level folder. Onboarding named the user's past runs this way, so use exactly this.

Workflow shapes (`--workflow`): `solo`, `best_of_n`, `plan_implement`, `implement_review`, `plan_implement_review`, `swarm`. A piece setting is `--set PIECE=HARNESS:MODEL:EFFORT`, for example `--set implement=claude-code:claude-opus-5-5:high`; harnesses are `claude-code`, `codex` and `command`.

## Commands

### Status
`loopmath status --json`. Read `exists` (the store exists), `fit.latest` (fit id or null), `fit.usable` (false with no fit, or with a fit an older loopmath made: refit), `fit.problem` (why not, or null), `fit.running`, `counts.runs`, `open_runs[]` (`run`, `type`, `repo`, `started_at`, `slate`), `budget.cap_usd`, `budget.spent.usd`.

### Onboard
- `loopmath onboard --dry-run --json`: reads the history, writes nothing. Read `groups.total`, `groups.to_label`, `window.from`, `window.to`, `sessions`, `cost_in_logs.usd`, `labeler.chosen`, `labeler.spec`, `labeler.expected.usd`, `labeler.options[]` (`spec`, `title`, `expected.usd`): one entry per labeller this machine can run, plus `none`.
- `loopmath onboard --labeler SPEC --yes --json`: labels, records one run per session group, saves the usual workflow per task type and repo, runs the first fit. Read `runs.written`, `classified.by_type`, `unclassified.total`, `unclassified.by_reason`, `usual[]` (`type`, `repo`, `label`, `runs`, `total`), `fit.id`, `fit.error`, `labeler.actual.usd`, `labeler.failed_chunks`.
- Labeller forms (`SPEC`), always the user's choice: `claude:MODEL` (their `claude` CLI, for example `claude:claude-haiku-4-5`), `codex:MODEL` (their `codex` CLI, for example `codex:gpt-6-luna`), `command:CMD` (their own command, for example a local model; nothing leaves the machine), `none` (keyword guess, no model).
- `--since 90d` (default) sets how far back; `30d`, `12w` or a date also work.

### Validate and import OCP files
- `loopmath ocp validate FILE... --json`. Read `ok`, `passed`, `failed`, `files[]` (`path`, `ok`, `errors`, `warnings`, `findings[]` (`code`, `message`)).
- `loopmath run import DIR --finish --no-fit --json`: a directory (its `*.ocp.json`, not recursive) or several files. Read `imported`, `failed`, `files[]` (`file`, `ok`, `run`, `error`). With exactly one FILE the JSON is `{run, path, state, finished}` instead.

### Fit
- `loopmath fit --json`: waits for the fit (seconds on a laptop). Read `fit.id`, `fit.n_runs.user`, `fit.n_runs.prior`, `runs_by_source`, `options`, `seconds`, `dropped`.
- `--without SOURCE` (repeatable) leaves out a shipped source (`e0`, `sweep`, `rq1`, `benchmark`, `shared`); `--no-prior` fits the user's runs alone. Use them only when the user asks.

### Pages
- Results page (after any fit): `loopmath posterior --html`. Narrow it with `--workflow CFG`, or `--type TYPE --repo REPO`.
- Planning page for one task: `--html` on the `recommend` command.
- One run: `loopmath runs --run RUN --html`.
- Each prints the page path. Open it for the user with `open PATH` (macOS) or `xdg-open PATH` (Linux). If that fails or there is no display, give the path.

### Recommend
```sh
loopmath recommend --type TYPE --repo REPO --title "TEXT" --feature size=m --base-commit SHA --json --brief --html
```
- `--target 'heldout_perf>=2400'` or `--target 'runtime_s<=200'` (quoted) when the user states a score goal. Else the configured rule applies (by default: the task's tests pass).
- `--models M,M` limits candidate models when the user names them.
- `--brief` keeps what an agent needs. Read `rec`, `page`, `message` (the summary for the user), `reference` (`kind`, `label`, `text`), `rescue`, `choices[]`. The baseline is always `reference`: `kind` is `usual` (the user's habit), `best_recorded` (their best recorded workflow, when there is no habit) or `default`.
- Each choice: `key` (`goal`, `pair`, `reference`, `cheapest_run`), `title` (one line saying what the option does), `label`, `recommended` (true for `goal` only), `chance` (`mean`, `lo`, `hi`, `of`: what the chance is of), `cost_per_accepted_usd` (`mean`, `lo`, `hi`), `run_cost_usd` (`mean`, `median`), `expected_rescue_usd`, `config`, `members` (config ids; two for a pair).
- There is a `pair` choice only when a second workflow is worth trying beside the goal; without one, `message` says so, and `run start --choice` takes only the keys in `choices[]`. The `pair` choice adds `explore_config` (the second member), `price_now_usd` (the extra run's cost now), `gain_per_future_run_usd` (the expected saving on each future similar run) and `p_beats_goal`.
- The `goal` choice adds `strategy`: null, or the try-then-rescue plan when the pick is less likely than the reference. Show `strategy.text` word for word. Its numbers are in `chance`, `misses`, `rescue`, `cost_per_accepted_usd` and `reference_cost_per_accepted_usd`. The same object is `goal.strategy`.

### Run start
```sh
loopmath run start --rec REC --choice KEY --base-commit SHA --json
```
Starts the chosen option: one run, or for `pair` a new slate with both runs. The task, configuration and source come from the recommendation and the choice, so never pass `--source` with `--choice`. The run is judged by the rule the recommendation was made for. Read `slate` (null unless a pair), `runs[]` (`run`, `config`, `label`, `source`, `pieces`, `piece_settings[]` (`piece`, `role`, `width`, `harness`, `model`, `effort`)). `pieces` is the piece ids and `piece_settings` their settings, both in workflow order.

For a workflow the user describes instead of a choice:
```sh
loopmath run start --rec REC --type TYPE --repo REPO --title "TEXT" --workflow SHAPE \
  --set PIECE=HARNESS:MODEL:EFFORT --source user_edit --base-commit SHA --json
```
One `--set` per piece. The JSON is one run, flat: `run`, `label`, `pieces`, `piece_settings[]`.

### Run record
```sh
loopmath run record --run RUN --session [PIECE=]ID --verified NAME=VALUE --reported NAME=VALUE --json
```
Records a finished run in one call: finds each session in the logs and adds one attempt per session (a Claude Code session counts its sub-agents), the commits made since the base commit, the verdicts and scores, then finishes (cost from the logs, receipt, background refit). Every check happens before the first write: an error writes nothing.
- `--session ID` (repeatable): a Claude Code session id (the UUID given to `claude --session-id`), a Codex thread id (`thread_id` from `codex exec --json`), or `self` (the Claude Code session running the command). `PIECE=ID` names the piece; else loopmath maps each session to a piece by model, then harness, then order, and says how in `attempts[].how`.
- `--cwd PIECE=PATH` (repeatable): a piece run by an agent with no session id (interactive Codex, for example), matched by folder and time when the run finishes.
- `--since TS`: when the work began (an ISO time, or `2h`), for a run opened here. With `self` and no `--since`, the whole session counts. `--cwd` without `--run` needs it.
- `--verified NAME=VALUE`: a verdict you observed yourself (a test command's exit code, CI read from its API, a merged PR). `--reported NAME=VALUE`: what an agent or the user told you. Verdict values: `pass`, `fail`, `accept`, `reject`, `error`. Usual names: `tests`, `review`, `build`.
- `--score NAME=VALUE`: a measured score (`runtime_s=182`); its direction and target come from the run's rule. `NAME=` (empty) declares a score you could not measure.
- `--commit SHA` (repeatable) adds commits by hand; `--no-commits` skips the automatic ones.
- `--no-fit` skips the background refit.
- With `--run RUN`, the task, configuration, rule and base commit are the run's; giving any of them again exits 1.
- Without `--run` it opens the run: `--rec REC --choice KEY` (not `pair`: start a pair with `run start`), or the task and workflow flags as in `run start` with `--source` (`habit` for a run loopmath did not plan).
- Read `run`, `attempts[]` (`piece`, `session`, `model`, `how`), `unmatched[]` (`attempt`, `reason`), `commits[]` (`sha`), `signals[]`, `cost.usd`, `outcome`, `receipt.line` (predicted against actual cost, one line), `fit.started`, `notes[]`. The `outcome` is `accepted`, `rejected` or `unknown`.
- Exit 2: a session, run or recommendation is not found; check the id you noted. Never guess an id.

### Pairs
- `loopmath config get referee.model --json`. Read `value`: the model family the referee should come from (null: any family other than both runs' implementers).
- `loopmath outcome --slate SLT --prefer RUN --judge referee --blinded --json`: the referee's preference; `--prefer tie` for a tie.

### Late events
`loopmath outcome --commit SHA --signal incident=REF --kind event --source user --json`: an incident, revert or hotfix tied to a commit made under loopmath.

### Doctor
`loopmath doctor --json`. Read `ok`, `checks[]` (`name`, `status`, `summary`). Use it when a command fails in a way this page does not explain.
