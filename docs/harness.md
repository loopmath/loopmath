# Tuning a Claude Code harness (experimental)

`loopmath harness` is experimental in 0.4.0rc1: it has tests on stub binaries and a recorded event stream, and its
output may change. It follows the design measured in stream 1 of our harness2 study; this release has not been
measured on its own yet.

A harness is what a repo gives Claude Code beyond the task: CLAUDE.md, skills, agents and settings; a tune edits the
first three. Use `loopmath harness` when you run one kind of task in a repo again and again and can score a run,
with your own command (tests, a script that prints a number) or with the tester judge, which writes acceptance tests
once per task. The tune:

1. runs the task on the base harness a few times and checks that the scores can tell runs apart;
2. starts editor sessions that propose small, separate edits to the harness;
3. measures every edit at once in a screening design of task runs (a Plackett-Burman design by default), fits
   each edit's effect and keeps the ones that help;
4. builds the kept edits into one harness, runs it against the base and delivers it only when it wins, by the
   delivery rule; otherwise the base is delivered;
5. writes a lesson for the next tune of the same kind of task (the reflector), when it is on.

Every session spends model calls. `plan` says how many and roughly what they cost before any runs, and `run`
asks before it spends.

## The flow

```
$ loopmath harness init
wrote loopmath-harness.toml
$ loopmath harness config set task.prompt "add a --json flag to the export command"
$ loopmath harness config set score.command "pytest -q tests/test_export.py"
$ loopmath harness config set auth.token_file ~/.config/loopmath/claude-token
$ loopmath harness plan
stages: judge, base, signal, edit, design, probe, fit, build, final, deliver, reflect
sessions: 15 executor runs, 2 editor sessions, 1 reflector, 1 test writer
cost: 2.72 USD (1.72 to 6.38)
time: typical unknown; worst case 10.5 h
design: pb12, 12 probe runs over up to 10 edits, residual df 3
caps: 10.00 USD, 40 runs, 6.0 h
...
$ loopmath harness run --detach --yes
$ loopmath harness watch
$ loopmath harness show --effects
$ loopmath harness apply --dry-run
$ loopmath harness apply
```

- `init` writes the config file with the defaults and one comment per field. `config set KEY VALUE` and `config
  unset KEY` edit it line by line, keeping comments, unknown keys and the order, and check each value against its
  field. Every field also has a flag (`--budget-usd 6`, `--executor-model claude-sonnet-5`), which wins over the
  file; `--set KEY=VALUE` takes any dotted key, and `--editor-model` sets every editor (`--set
  editors.2.model=...` sets one).
- `plan` is the dry run, with no model call: the stages in order, the sessions by role, the cost point and its
  range, the worst-case time, the design's residual degrees of freedom, the warnings and the preflight checks
  (Claude Code's version, the repo, instruction files above the work root, the token file, the price rows).
- `run` prints the plan and asks on a terminal; without a terminal it needs `--yes`. With `--detach` it prints the
  tune id and returns, and the engine runs on as its own session, so closing the terminal does not stop it. A
  cap in `[budget]` stops the tune when it is reached.
- `watch` follows the events to the end, `status` gives a snapshot (`--line` gives one line), `stop` asks the tune
  to stop (`--now` kills its sessions by recorded pid), and `resume` continues a paused or interrupted tune
  without repeating a finished session.
- `show` says what was delivered and why; `--effects` adds the effects table, `--diff` the harness against the
  base, and `--ledger` the spend by role and model beside Claude Code's own figure.
- `apply` writes the delivered CLAUDE.md, skills and agents into the repo's working tree, refuses when those
  paths have uncommitted changes, and commits nothing.
- `loopmath harness skill` prints the guide an agent follows to drive these commands for a person: ask before
  any spend, report a base delivery plainly, never read transcripts or the token file.

A tune whose measurements cannot separate an edit from the base ends `no-signal` (exit 4) and delivers the base.
That is the rule working: run more probes (`screen.repeats`) or sharpen the score, rather than applying edits by
hand.

## Isolation and auth

- Each run gets its own local clone of the task repo (`git clone --shared`), checked out at the base commit with
  no remote, no branch and no git hooks, and a fresh, empty `CLAUDE_CONFIG_DIR` holding only the settings the tune
  writes, so a run never loads your own CLAUDE.md, plugins, MCP servers, memory or login. The tune never reads or
  writes your Claude Code config folder, and a run adds no branch, worktree or object to the task repo. A clone
  kept after its run (`run.keep_worktrees`) is repacked and its link to the repo's objects removed, so a later
  `git gc` in your repo cannot break it.
- **User hooks do not run inside tune sessions.** Every session's settings, passed with `--settings` (which
  outranks the repo's own settings files), set `disableAllHooks` and `disableSkillShellExecution`: no hook from
  any settings file, skill or plugin runs, and a skill's or command's `!` lines are not run. The repo's CLAUDE.md,
  skills and agents still load. MCP servers do not start: sessions run with `--strict-mcp-config` and no MCP
  config, so a project `.mcp.json` or an agent's `mcpServers` is ignored. A tune therefore measures no hook, MCP
  server or `!` line, and an edit that writes a settings file or a `.mcp.json`, gives a skill or agent a `hooks` or
  `mcpServers` key, or adds a `!` line is refused.
- Sessions get a fresh environment, not a copy of yours: `PATH`, `HOME`, the locale and terminal variables, a
  per-run `TMPDIR`, and the settings that turn off auto memory, telemetry and the auto-updater.
- **Auth.** By default, run `claude setup-token` once and save the token to a file, mode 600, named by
  `auth.token_file`. The tune reads it at start and hands it to each session on an inherited pipe; it is never on
  a command line, in an environment variable, in the config, the ledger or an event. `auth.mode = "api-key-env"`
  passes the named variable to the sessions' environment instead. Under a subscription token the dollars are what
  the calls would cost through the API, and they count against your plan's usage limits; a usage limit pauses the
  tune, and `resume` continues it.
- **Limits.** Every session's file tools (Write, Edit and NotebookEdit, in subagents too) write only in its own
  clone or work folder, its config folder and its `TMPDIR`; an executor's sessions also read only their own
  run's folders of the work root. The agent's sandbox confines its shell's writes to its folder (the clone's own
  `.git` among it), its `TMPDIR` and Claude Code's temp folder. A subagent's own worktree is made inside the run's
  clone and left out of the run's diff; EnterWorktree, `git push` and `git remote` are denied. With
  `executor.sandbox = false` the file tools keep their limits and the shell is not confined. Stop and kill act on
  recorded pids and process groups only.
- **Not covered.** These limits do not cover shell writes to Claude Code's own temp folder (`/tmp/claude-<uid>`,
  shared by every session of your user), which the sandbox allows by itself, or a hard link into a run's folder
  that exists before the session starts. They have been checked on macOS only, not on Linux.
- `score.command` runs in the run's clone after the session ends, in its own process group, with
  `score.timeout_s`. The tester judge writes its acceptance tests once, into the tune's folder; they are copied
  into each run's clone after the agent stops, run there, and removed before the run's diff is taken. An edit
  that names the judge's folder or copies a line of its tests is refused.
- Every session is a row of `calls.jsonl` in the tune's folder, priced against the packaged price table, with
  Claude Code's own cost beside it.

## In Claude Code: /harness

`loopmath harness mod install` installs the mod. It copies the packaged mod to
`$LOOPMATH_HOME/mods/loopmath-harness-<version>/`, then prints and, after a yes, runs:

```
claude plugin marketplace add <that folder> --scope user
claude plugin install loopmath-harness --marketplace <that folder> --scope user
```

`--print` only prints them, `--scope project` installs for one repo, and `mod uninstall` reverses both.
`mod validate --test` runs `claude plugin validate --strict` and `claude plugin test` on it under a throwaway
config folder.

- `/harness` opens the configure pane: sections for Task, Score, Loop, Models, Screen, Budget and Run, drawn from
  `loopmath harness schema --json`, with a picker for each field that has choices and a text field for the rest.
  Each change runs `loopmath harness config set`, so the file stays the record. **Plan** shows the plan;
  **Start** asks once more, with the cap on the button, and the second press starts the tune detached.
- `/harness watch [ID]` opens the progress pane (stages, sessions, edits, the effects table, the delivery) with
  Stop, Show and Apply (a dry run first, then a second press writes the files). While a tune runs, a band above
  the prompt and the status line show it in one line.
- `/harness stop`, `/harness show` and `/harness apply` (`--yes` to write) run the same CLI verbs.

The mod runs only the `loopmath` CLI and makes no model call of its own. Reloading it or closing the session
leaves the tune running; the progress feed picks the stream up again where it stopped. Its options are
`loopmath_bin` (the executable) and `config_path` (the config file).

## In herdr

With `run.runner = "herdr"` (herdr 0.9.3 or later, its server running), the tune opens one herdr tab, follows its
own events in the tab's first pane, and runs each session in a pane of its own beside it, reporting the
session's state to herdr so `herdr agent list` and `herdr agent wait` see it. Stopping the tune closes those
panes; the events pane stays.

- To end a tune, stop it (`loopmath harness stop`, `--now` to kill its sessions) rather than close its panes. A
  session pane closed by hand ends without its record: its session runs on until the engine kills it, a minute
  past its time limit, and the run counts as failed. Ctrl-C in a session pane stops that session, and the run
  counts as failed.
- With `auth.mode = "api-key-env"`, each session reads the key from its pane's environment, which herdr gives it,
  not from the shell that ran the tune: the herdr server needs the variable. The token file (the default) does
  not depend on it.
- The plugin's actions run `loopmath` by name, so `loopmath` must be on herdr's PATH, and they act on the newest
  tune under herdr's `$LOOPMATH_HOME`. The runner's own panes run the engine's Python by its path.

`loopmath harness herdr link` prints and, after a yes, runs `herdr plugin link` on the packaged plugin, which adds
three actions: `watch` (the newest tune's events in a split pane), `status` (a popup) and `stop`.
`loopmath harness herdr unlink` reverses it. To watch the newest tune on a key, add to herdr's `config.toml` (a
`pane` command opens a temporary pane, which closes when the watch ends with the tune):

```toml
[[keys.command]]
key = "prefix+alt+h"
type = "pane"
command = "loopmath harness watch --pretty"
```

`loopmath harness status --line` prints the one-line status, for a status bar.

## The tune's folder

A tune lives in `run.work_root` (default: `$LOOPMATH_HOME/harness/tunes/`), one folder per tune id:
`config.toml` (the frozen config), `plan.json`, `events.jsonl`, `state.json`, `calls.jsonl`, `effects.md`,
`report.md`, the base and delivered harness trees, and one folder per session. `loopmath harness list` lists the
tunes, newest first.

## Reference

This part is generated by `loopmath harness skill`, which prints the same tables.

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
