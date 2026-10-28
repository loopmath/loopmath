# loopmath harness

Tune a Claude Code harness on a repeatable task. Config schema `loopmath.harness-tune/0.1`, events `loopmath.harness-events/0.1`.

## When to use it

You have a task you will run again and again (a kind of change in one repo), and a way to score a run: a command that checks the work (tests, a script that prints a number) or the tester judge, which writes acceptance tests once per task. The tune proposes edits to the repo's Claude Code harness (CLAUDE.md, skills, agents, settings hooks), measures each edit in a screening design of task runs, and delivers the kept edits only when they beat the base.

Not for a one-off change, and not without a scorer: with nothing to score the runs, the plan refuses. Every run spends model calls, so the person agrees to the spend first.

## The flow

1. `loopmath harness init`: Write loopmath-harness.toml in the repo. Set task.prompt (or task.prompt_file), score.command when there is a check, and auth.token_file (a `claude setup-token` token, mode 600). `loopmath harness config set KEY VALUE` edits one key and keeps the comments.
2. `loopmath harness plan --json`: The dry run, with no model call. Show the person the sessions, the cost point and range, the worst-case time, the warnings and any failed preflight check.
3. Ask the person to agree to the spend: up to budget.usd, with the plan's point and range. Do not go on without a yes.
4. `loopmath harness run --detach --yes --json`: Only after the yes. It prints the tune id and returns; the engine runs on in its own session.
5. `loopmath harness watch ID --json`: Follow the events (one JSON object per line) to tune.end, or `loopmath harness status ID --json` for a snapshot. The exit code is the tune's.
6. `loopmath harness show ID --effects`: What was delivered and why, and the effects table. Report it plainly, including a base delivery.
7. `loopmath harness apply ID --dry-run`: Show the person which files would change. Only after they say yes, run `loopmath harness apply ID`. It writes the files and commits nothing.

## Rules

- Never run a tune before the person agrees to its spend, and never apply a harness before they agree to it.
- Never raise budget.usd, budget.runs, budget.wall or budget.set_usd unasked.
- When a tune ends no-signal (exit 4), say plainly that the measurements could not separate an edit from the base, so the base is delivered. That is the rule working, not a failure to retry.
- A base delivery (exit 0, status base) means the tuned build did not beat the base. Do not apply edits by hand around it.
- A paused tune (exit 6) hit a usage limit or an auth error. Tell the person; `loopmath harness resume ID` continues it once they say so.
- To stop a tune, run `loopmath harness stop ID` (running sessions finish) or `--now` (they are killed).
- Read the event stream and the CLI's JSON; never read session transcripts or the token file.

## Commands

Every command takes `--json`.

| Command | Does |
|---|---|
| `loopmath harness init [--out PATH] [--force] [flags]` | write the config file with one comment per field |
| `loopmath harness plan [flags]` | the dry run: stages, sessions, cost point and range, time, df, warnings, preflight |
| `loopmath harness run [flags] [--detach] [--yes]` | start a tune and print its id |
| `loopmath harness status [ID] [--line]` | a snapshot folded from the event stream |
| `loopmath harness watch [ID] [--json] [--from SEQ]` | follow the events until tune.end |
| `loopmath harness stop [ID] [--now]` | write the stop file |
| `loopmath harness resume ID [--detach]` | continue a paused or interrupted tune |
| `loopmath harness show [ID] [--diff] [--effects] [--ledger]` | the delivered harness and why |
| `loopmath harness apply [ID] [--to REPO] [--work] [--dry-run]` | write the delivered harness into the repo; commits nothing |
| `loopmath harness list` | tunes, newest first |
| `loopmath harness config get\|set\|unset KEY [VALUE]` | read or edit the config file in place, keeping comments |
| `loopmath harness schema --json` | the config's JSON Schema |
| `loopmath harness skill [--format md\|json]` | this doc |
| `loopmath harness mod install\|uninstall\|path\|validate [--print]` | the Claude Code mod (/harness); install prints each claude command and asks before running it |
| `loopmath harness herdr link\|unlink\|path [--print]` | the herdr plugin; link prints the herdr command and asks before running it |

## Config fields

Every field has a flag. A flag wins over the file, and the file over the default. `--set KEY=VALUE` takes any dotted key and is repeatable. `--editors N` sets the number of editors; `--editor-model M` and the other `--editor-*` flags set every editor; `--set editors.2.model=M` sets one.

### [task]

| Key | Default | Flag | What |
|---|---|---|---|
| `task.repo` | `"."` | `--task-repo` | git working tree the agent works in |
| `task.base_ref` | `"HEAD"` | `--task-base-ref` | commit every run starts from |
| `task.prompt` | `""` | `--task-prompt` | the task text |
| `task.prompt_file` | `""` | `--task-prompt-file` | a file holding the task text |
| `task.set` | `""` | `--task-set` | a JSONL task set: one {id, prompt, base_ref?, score_command?} per line |
| `task.id` | `""` | `--task-id` | stable task id; default: sha256 of the prompt and the base commit, 12 hex |

### [score]

| Key | Default | Flag | What |
|---|---|---|---|
| `score.command` | `""` | `--score-command` | your own check, run in the run's worktree after the agent stops |
| `score.kind` | `"auto"` | `--score-kind` | how the command's result reads: auto (a number if the last line parses as one, else pass), pass (exit 0 is 100), number (last stdout line, rescaled from score.range), junit (auto / pass / number / junit) |
| `score.range` | `[0, 100]` | `--score-range` | what a number from the command means: [low, high] maps to 0..100 |
| `score.junit_path` | `""` | `--score-junit-path` | the JUnit XML file a junit command writes, in the worktree |
| `score.timeout_s` | `900` | `--score-timeout-s` | seconds the score command may run |
| `score.combine` | `"auto"` | `--score-combine` | how the command and the judge combine: auto (gate with both, else the one present), gate (a failed command scores 0, else the judge), command, judge, mean (auto / gate / command / judge / mean) |

### [judge]

| Key | Default | Flag | What |
|---|---|---|---|
| `judge.kind` | `"tester"` | `--judge-kind` | the judge: tester (one test-writer session per task writes an acceptance bundle that scores every run), rubric (a rubric file), none (tester / rubric / none) |
| `judge.model` | `"claude-sonnet-5-5"` | `--judge-model` | the test writer's model |
| `judge.effort` | `"high"` | `--judge-effort` | the test writer's effort (low / medium / high / xhigh / max) |
| `judge.attempts` | `3` | `--judge-attempts` | test-writer attempts: a bundle that fails on the starting repo for another reason than the missing feature is written again |
| `judge.tries` | `2` | `--judge-tries` | tries to score one run; a run with no score after them is left out of the fit |
| `judge.rubric_file` | `""` | `--judge-rubric-file` | the rubric file, for judge.kind = rubric |

### [executor]

| Key | Default | Flag | What |
|---|---|---|---|
| `executor.agent` | `"claude-code"` | `--executor-agent` | the task agent (claude-code) |
| `executor.model` | `"claude-haiku-5-5"` | `--executor-model` | the task agent's model |
| `executor.effort` | `"high"` | `--executor-effort` | the task agent's effort (low / medium / high / xhigh / max) |
| `executor.max_turns` | `200` | `--executor-max-turns` | turn limit per run |
| `executor.timeout_s` | `3600` | `--executor-timeout-s` | seconds per run |
| `executor.tries` | `2` | `--executor-tries` | tries per run when the executor fails outside the harness; then the run is left out of the fit |
| `executor.fallback_model` | `""` | `--executor-fallback-model` | Claude Code's --fallback-model, empty for none |
| `executor.permission_mode` | `"bypassPermissions"` | `--executor-permission-mode` | Claude Code's --permission-mode (acceptEdits / auto / bypassPermissions / manual / dontAsk / plan) |
| `executor.sandbox` | `true` | `--executor-sandbox` | sandbox the agent's shell: writes only in its folder (its clone's .git among it), TMPDIR and Claude Code's temp folder, and an executor's reads only in its own run's folders (the file tools are limited either way) |

### [base]

| Key | Default | Flag | What |
|---|---|---|---|
| `base.runs` | `2` | `--base-runs` | base runs, which also give the first noise read |
| `base.reuse` | `""` | `--base-reuse` | a folder of earlier scored base runs to import instead of running them |

### [signal]

| Key | Default | Flag | What |
|---|---|---|---|
| `signal.enabled` | `true` | `--signal-enabled` | check the base runs for a ceiling and a floor |
| `signal.on_ceiling` | `"stop"` | `--signal-on-ceiling` | every base run scores the maximum: warn and go on, or stop and deliver the base before any editor runs (warn / stop) |
| `signal.on_floor` | `"warn"` | `--signal-on-floor` | every base run scores 0: warn and go on, or stop (warn / stop) |
| `signal.min_t` | `0.0` | `--signal-min-t` | an edit is kept when its effect is above min_t standard errors; 0 keeps any positive effect |

### [edit]

| Key | Default | Flag | What |
|---|---|---|---|
| `edit.enabled` | `true` | `--edit-enabled` | run the editors; off needs edit.from |
| `edit.from` | `""` | `--edit-from` | your own edit folders (one subfolder per edit), used when the editors are off |

### [[editors]]

| Key | Default | Flag | What |
|---|---|---|---|
| `editors.name` | `""` | `--editor-name` | the editor's name, used in labels and folders; default: its focus, numbered when repeated |
| `editors.focus` | `"procedure"` | `--editor-focus` | procedure (how to go about this kind of task), domain (facts and conventions a newcomer gets wrong) or file (focus_file) (procedure / domain / file) |
| `editors.focus_file` | `""` | `--editor-focus-file` | a file stating the editor's focus, for focus = file |
| `editors.model` | `"claude-opus-5-5"` | `--editor-model` | the editor's model |
| `editors.effort` | `"high"` | `--editor-effort` | the editor's effort (low / medium / high / xhigh / max) |
| `editors.max_edits` | `5` | `--editor-max-edits` | edits one editor may propose |
| `editors.attempts` | `2` | `--editor-attempts` | editor attempts; the retry runs only when an attempt yields no usable edit |
| `editors.timeout_s` | `3600` | `--editor-timeout-s` | seconds per editor session |

The defaults are 2 editors: procedure, domain.

### [screen]

| Key | Default | Flag | What |
|---|---|---|---|
| `screen.design` | `"pb12"` | `--screen-design` | the probe design: pb12 (12 single runs, Plackett-Burman), fold-over (2 x pairs runs), one-at-a-time (one run per edit), full (every subset, up to 4 edits) (pb12 / fold-over / one-at-a-time / full) |
| `screen.max_edits` | `10` | `--screen-max-edits` | edits the design takes; the round robin order sets which are kept |
| `screen.repeats` | `1` | `--screen-repeats` | runs per probe row |
| `screen.pairs` | `8` | `--screen-pairs` | pairs, for the fold-over design |
| `screen.seed` | `0` | `--screen-seed` | seed of the column assignment and the run order |
| `screen.base_rows` | `true` | `--screen-base-rows` | base runs enter the fit as all-off rows |

### [fit]

| Key | Default | Flag | What |
|---|---|---|---|
| `fit.select` | `"least-squares"` | `--fit-select` | what selects the kept edits: least-squares, or head (the shrinkage model) (least-squares / head) |
| `fit.report` | `["head"]` | `--fit-report` | fits reported beside the selector (head) |

### [final]

| Key | Default | Flag | What |
|---|---|---|---|
| `final.runs` | `1` | `--final-runs` | runs of the kept set's build; 0 needs deliver.rule fit or always-build |

### [check]

| Key | Default | Flag | What |
|---|---|---|---|
| `check.max_pairs` | `3` | `--check-max-pairs` | replicate base and final pairs the gate rule may add |

### [deliver]

| Key | Default | Flag | What |
|---|---|---|---|
| `deliver.rule` | `"final-vs-base-mean"` | `--deliver-rule` | the delivery rule: final-vs-base-mean (deliver the build unless its final run scores below the base mean), fit (the build, no final run), gate (loopmath.gate.decide), always-build (the build whenever an edit was kept) (final-vs-base-mean / fit / gate / always-build) |

### [reflector]

| Key | Default | Flag | What |
|---|---|---|---|
| `reflector.enabled` | `true` | `--reflector-enabled` | one session writes a lesson into the role library |
| `reflector.model` | `"claude-opus-5-5"` | `--reflector-model` | the reflector's model |
| `reflector.effort` | `"high"` | `--reflector-effort` | the reflector's effort (low / medium / high / xhigh / max) |
| `reflector.when` | `"end"` | `--reflector-when` | when the reflector runs: end, or after-fit (end / after-fit) |
| `reflector.timeout_s` | `1800` | `--reflector-timeout-s` | seconds for the reflector session |

### [library]

| Key | Default | Flag | What |
|---|---|---|---|
| `library.enabled` | `true` | `--library-enabled` | editors read, and the reflector writes, the role library |
| `library.path` | `""` | `--library-path` | the library folder; default $LOOPMATH_HOME/harness/library |
| `library.domain` | `""` | `--library-domain` | the role key lessons are filed under; default: the repo name |

### [surface]

| Key | Default | Flag | What |
|---|---|---|---|
| `surface.scope` | `"project"` | `--surface-scope` | where the harness lives: project (the repo's CLAUDE.md, .claude/skills, .claude/agents, settings hooks) or user (a clean config folder, as the study ran) (project / user) |
| `surface.always_on` | `["CLAUDE.md"]` | `--surface-always-on` | always-on files: added text only |
| `surface.always_on_cap` | `8000` | `--surface-always-on-cap` | characters the always-on files may hold |
| `surface.skills` | `true` | `--surface-skills` | skills are editable |
| `surface.agents` | `true` | `--surface-agents` | agents are editable |
| `surface.hooks` | `true` | `--surface-hooks` | hooks are not editable and do not run in tune sessions; true shows the base settings.json to editors |
| `surface.allow_model_names` | `false` | `--surface-allow-model-names` | harness files may name a model |

### [budget]

| Key | Default | Flag | What |
|---|---|---|---|
| `budget.usd` | `10.0` | `--budget-usd` | dollar cap per task, checked before every session at its high estimate |
| `budget.runs` | `40` | `--budget-runs` | executor run cap per task |
| `budget.wall` | `"6h"` | `--budget-wall` | wall-clock cap per task |
| `budget.recheck_after_base` | `true` | `--budget-recheck-after-base` | price the plan again from the base runs' measured cost |
| `budget.set_usd` | `0.0` | `--budget-set-usd` | dollar cap for a whole task set; 0: no set cap |

### [loop]

| Key | Default | Flag | What |
|---|---|---|---|
| `loop.rounds` | `1` | `--loop-rounds` | rounds of edit to deliver, each on the last round's delivered harness |
| `loop.overlap` | `false` | `--loop-overlap` | not in this release; true is refused |

### [run]

| Key | Default | Flag | What |
|---|---|---|---|
| `run.width` | `4` | `--run-width` | executor sessions at once (judge sessions count against it) |
| `run.editor_width` | `0` | `--run-editor-width` | editor sessions at once; 0: all |
| `run.runner` | `"local"` | `--run-runner` | where sessions run: local, or herdr (one pane each) (local / herdr) |
| `run.work_root` | `""` | `--run-work-root` | the tunes folder; default $LOOPMATH_HOME/harness/tunes |
| `run.keep_worktrees` | `"failed"` | `--run-keep-worktrees` | worktrees kept after scoring (failed / all / none) |
| `run.record` | `true` | `--run-record` | write each executor run into the loopmath store as a harness-runs row |

### [auth]

| Key | Default | Flag | What |
|---|---|---|---|
| `auth.mode` | `"token-file"` | `--auth-mode` | how sessions log in: token-file (a `claude setup-token` token, passed by file descriptor) or api-key-env (token-file / api-key-env) |
| `auth.token_file` | `""` | `--auth-token-file` | the token file, mode 600; read at start, never put in argv, env or files |
| `auth.api_key_env` | `"ANTHROPIC_API_KEY"` | `--auth-api-key-env` | the variable holding the API key, for api-key-env |

## Events

`loopmath harness watch ID --json` prints one event per line: `{"v", "seq", "ts", "tune", "type", "stage", "label", "data"}`.

| Type | Data keys | What |
|---|---|---|
| `tune.start` | config_digest, plan, task, config | the config digest and the plan |
| `stage.start` | planned | a stage begins, with its planned session count |
| `stage.end` | counts | a stage ends, with its counts |
| `session.start` | role, model, effort, pid, worktree, attempt | a model session starts: role, model, pid |
| `session.end` | role, model, exit, is_error, wall_s, cost_usd, cli_cost_usd, tokens, error_kind | it ends: exit, wall time, cost, tokens |
| `run.score` | score, kind, command_exit, judge_score, deliverable, on | an executor run's score |
| `edit.proposed` | editor, slot, id, component, files | an editor proposed an edit |
| `edit.refused` | editor, slot, id, reason, when | an edit broke a rule |
| `edit.kept` | editor, slot, id, component, files | an edit enters the design |
| `design` | kind, edits, matrix, order, alias, df | the probe design: matrix, run order, residual df |
| `fit` | select, edits, df, sigma, refused, kept | the effect of each edit, and which are kept |
| `build` | kept, digest | the kept set built into one harness |
| `deliver` | choice, rule, why, run | the delivery choice, the rule and why |
| `signal` | kind, line | a line about the measurements (ceiling, floor, no separation) |
| `budget` | spent, cap, projected, what | spend against a cap |
| `stop.requested` | now | the stop file was seen |
| `error` | kind, message | an error |
| `tune.paused` | reason | a usage limit or an auth error paused the tune |
| `tune.resume` | rerun, killed, was | resume continued it |
| `tune.end` | status, harness, totals, exit | the end status, the delivered harness and the totals |

## Exit codes

| Code | Status | What |
|---|---|---|
| 0 | delivered | the tuned harness is delivered |
| 0 | base | the base is delivered (by the delivery rule) |
| 1 | error | an error |
| 1 | failed | the tune failed |
| 2 | usage | a usage or config error |
| 3 | refused | refused by the budget or the preflight, before any session |
| 3 | budget | a cap ended the tune |
| 4 | no-signal | not enough signal: the base is delivered |
| 5 | stopped | stopped |
| 6 | paused | paused; resume continues it |
