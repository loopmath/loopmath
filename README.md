# loopmath

loopmath helps an orchestrator agent choose how to run a coding task, and learns from how it went.

Before a task, it predicts how the user's usual agent workflow will do: the chance of an accepted result, and the cost in tokens and dollars. It shows the other workflows on a success-cost curve and offers two exploration picks. Each pick comes with its gain on every future similar run, its one-time price, and the number of runs it takes to pay back. After the task, the orchestrator records what ran and what came of it. loopmath fills in the token costs from the Claude Code and Codex logs, writes a receipt of predicted against actual, and refits.

loopmath never starts or coordinates agents. Claude Code, Codex, herdr or your own tool does that; loopmath is a CLI they call. Everything runs locally. loopmath connects to no other machine: `loopmath builder` serves its page on 127.0.0.1 only, and no other command opens a network connection. The logs are opened read only, and loopmath writes only to its store (`~/.loopmath`, which holds its caches too), to the skill folders `skill install` names, and to paths you name.

The package installs two commands: `loopmath` and its short alias `loop`. If another `loop` is already on your PATH, use `loopmath`.

## Install

Python 3.11 or newer.

```sh
pipx install loopmath          # or: uv tool install loopmath, or pip install loopmath in a venv
loopmath --version
```

From a clone of this repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Installing needs the network: pip, pipx and uv download the build backend (setuptools 77 or newer) and the dependencies (jsonschema, numpy, pandas, scipy), from a clone too, so `pip install --no-index -e .` fails offline.

## First run

```sh
loopmath skill install --dry-run   # every file it would write, and where; writes nothing
loopmath skill install             # teach your coding agents how to use loopmath
```

Then ask your agent to onboard, for example "onboard loopmath". The agent runs a dry run, asks you one question (which model labels your sessions, with the cost of each option), onboards, fits, and opens the results page.

`skill install` writes six skills, one per job. Each skill's folder also gets its own copy of the shared `reference.md`, with every command and JSON field the skills use, so each skill works on its own. An agent loads only the skill for the job at hand, and each phrase belongs to one skill:

| Skill | Ask your agent |
|---|---|
| `loopmath` | loopmath named without a job: "use loopmath", "what can loopmath do", "what next with loopmath" |
| `loopmath-onboard` | "onboard", "set up", "start" or "try" loopmath; or loopmath has no runs yet |
| `loopmath-import-runs` | OCP files (`*.ocp.json`), a folder of finished runs, or the output of an orchestrator or experiment harness that writes OCP |
| `loopmath-update-fit` | "update", "refresh" or "rerun" the fit; what loopmath learned; runs imported or recorded without a refit |
| `loopmath-plan-task` | "plan a task with loopmath"; which workflow, model or effort to use; a coding task of more than a few minutes in a repo that uses loopmath |
| `loopmath-record-run` | a task planned with loopmath is done; "record", "log" or "save" a run; both runs of a pair need a blind judge |

By default the skills go to every agent whose home exists: Claude Code (`~/.claude/skills/<name>/`, or under `$CLAUDE_CONFIG_DIR`) and Codex (`~/.codex/skills/<name>/`, or under `$CODEX_HOME`). `--target claude-code`, `codex` or `both` picks them yourself. If your Codex has no skills folder, the files go to `~/.codex/loopmath/<name>/` instead, with a short marked block in `~/.codex/AGENTS.md` that lists them. Add `--scope project` to install into the current repository (or `--dir PATH` for another one).

It is safe to run again. It records the files it wrote, with their hashes, in `.loopmath-skills.json` beside the skill folders, and it replaces or removes only those. A skill file you changed is kept, with a note. Any other file where a skill goes stops the install before it writes anything, and names the path to move aside. The single skill that 0.1 installed is replaced when it is as 0.1 wrote it. `loopmath skill show NAME` prints one skill, and `loopmath skill uninstall` removes the files it wrote. `loopmath doctor` shows what loopmath can see: logs, agents, store, fit, prices and skills. To read the logs it builds the log parse cache in the store and keeps it, so the next read is fast, and it says how big the cache is; it changes nothing else in the store.

To onboard by hand instead:

```sh
loopmath onboard --dry-run                                  # what it found, and what each labeller costs
loopmath onboard --labeler claude:claude-haiku-4-5 --yes    # record it and run a first fit
```

The dry run makes no labelling call and writes no runs or fit; like `doctor`, it keeps the log parse cache in the store and says how big it is. The first read of a long history takes several minutes; with the cache, the next one is quick.

`onboard` reads your recent Claude Code and Codex sessions (the last 90 days unless you pass `--since`). It groups them into runs, labels each group's task type with a model you choose, names your usual workflow per task type and repository (sessions outside a git repository go under `(no repo)`), and fits. loopmath never picks the labelling model for you. `--labeler` takes one of:

- `claude:<model>`, through your `claude` CLI;
- `codex:<model>`, through your `codex` CLI;
- `command:<cmd>`, any local command such as `command:ollama run <model>`, so nothing leaves the machine;
- `none`, to skip labelling.

Reading the history uses up to 8 worker processes (one per CPU, at most 8). Set `LOOPMATH_WORKERS=1` to keep everything in one process, or another number to change the count.

loopmath ships with a prior built from our own sweeps and experiments, so `recommend` gives an answer before you have any history. Your own runs then move it.

The prior ships inside the package: our runs, in the shared form (no titles, paths, commands or free text), and the published benchmark results. They are the same files we fit with, and each release updates them. `onboard` and the results page say in one line which prior they start from, and `loopmath prior show` lists it source by source.

## The loop

The plan and record skills take an agent through the loop in three commands:

```sh
loopmath recommend --type bug_fix --repo acme/api --title "Fix the parser crash" --json --brief --html
loopmath run start --rec REC --choice goal --base-commit SHA --json      # the harness, model and effort per piece
loopmath run record --run RUN --session UUID --verified tests=pass --json  # costs from the logs, the outcome, the receipt
```

With `--run`, a session that started before the run counts from the run's start; a Codex session counts the usage it logged inside the window: each reading's increase at the reading's time, a reading on either edge included. A sub-agent or a Codex child thread counts in full when it started inside the window, and not at all when it started before it. A Codex reading lower than the one before it (a compaction reset) counts from zero. A Codex session that cannot be cut to the window counts whole, with a note that says why: a reading with no time, or increases inside the window that add up to less than zero. Give `--since TS` and `--until TS` when the task's work began or ended at another time; `--since` before `run start` moves the run's start back. When no commit is found after the base commit, the record says where it looked; pass `--commit SHA`. `loopmath run sessions` lists the recent Claude Code and Codex sessions in this folder with their ids, for `--session ID`. Outside a Claude Code tool shell, `--session self` uses the one Claude Code session in this folder active in the last 10 minutes.

The same loop step by step, for a tool of your own:

```sh
# 1. Plan: classify the task, then ask.
loopmath task-types
loopmath recommend --type bug_fix --repo acme/api --feature size=s --feature lang=python --json
#    the usual workflow (without one, your best recorded workflow is the reference), the success-cost curve,
#    alternatives, two exploration picks, a suggested pair

# 2. Record: one run per workflow you run, one attempt per agent you launch.
loopmath run start --type bug_fix --repo acme/api --base-commit SHA --config CFG --source usual --rec REC
loopmath run attempt --run RUN --piece implement --harness claude-code --model claude-opus-5-5 \
  --effort high --cwd . --session UUID          # launch the agent with: claude --session-id UUID
loopmath run attempt --run RUN --end ATT --status done
loopmath run artifact --run RUN --kind commit --path SHA --by ATT

# 3. Outcome: verdicts, scores, and later events.
loopmath outcome --run RUN --signal tests=pass --kind verdict --tier verified
loopmath outcome --run RUN --signal runtime_s=182 --kind score --unit s --better lower

# 4. Finish: costs from the logs, a receipt, a refit.
loopmath run finish --run RUN

# Later: an incident or revert traced back to the run that made the commit.
loopmath outcome --commit SHA --signal incident=INC-42 --kind event --source tracker
```

The usual workflow is the one most of your habit runs used: the runs onboard brought in, and runs started with `--source habit`. Runs of a recommended alternative or an exploration pick are recorded but never make a workflow your usual; `recommend --usual CFG` names one. Under the reference, `recommend` says where it comes from; each alternative shows its own chance and costs and what it changes from the pick, and an exploration pick says after how many similar runs it pays back. When the text would run past 25 lines it lists fewer alternatives and says how many more `recommend --json` has.

A pair runs the goal workflow and an exploration pick from the same base commit, in separate worktrees. `loopmath run start --rec REC --choice pair` starts both, on a new slate. To start them one at a time, save the recommendation (`recommend ... --json > rec.json`) and start each run from it, so they share one task: `loopmath run start --task-file rec.json --config CFG --source alternative --new-slate` for the goal, then the same with the pick's configuration, `--source exploration` and `--slate SLT`. A blinded referee then picks the better change, which you record with `loopmath outcome --slate SLT --prefer RUN --judge referee --blinded`. `loopmath runs --slate SLT` says which run the referee preferred.

A run counts as a success when it is accepted under your rule. By default that means the task's tests pass. A run that `onboard` brought in from your logs, or that `run import` stored, with no tests verdict but a review or accepted verdict, is judged by the review rule instead: a success when its latest review or accepted verdict is accept, approve or pass. `loopmath runs` lists other finished runs such a verdict could settle, with the command that moves them to the review rule, `loopmath outcome --rule review RUN...`. `loopmath outcome --undo-rule RUN...` undoes that, restoring each run's rule as it was; naming that rule (`tests` when it had none) does too, unless the run's own rule was also named review. A review verdict on a reviewer's last attempt then counts twice: as the review gate's pass, and as the run's success. A rule can also be a score target: `loopmath recommend ... --target 'heldout_perf>=2400'` or `--target 'runtime_s<=200'` gives the expected score and the chance of reaching the target. Set standing rules and other options with `loopmath config set KEY VALUE` (`loopmath config get` lists every key, set or not, `config set` refuses a key loopmath does not read, and `loopmath config unset KEY` returns a key to its default), and the spending cap with `loopmath budget --usd X --period month`. Spend counts your runs and onboard's labelling calls; runs onboard brought in from your logs are shown apart and not counted. A run you record with `--source habit` counts.

Runs recorded by another tool that writes OCP v0.3, such as an experiment runner, come in with `loopmath run import DIR --finish`: several files or a directory, with one refit at the end (`--no-fit` skips it). It says how many runs were new, unchanged or replaced; importing finished runs into a store with no fit starts a fit. Importing the same documents again changes nothing and keeps the outcomes you added since, and with `--finish` it finishes the runs left open (without it, the runs are stored open and it says to import again with `--finish`). A document whose run id is already in the store with different content is refused by name, with the fields that differ; `--replace` takes it and keeps the outcomes added after the import, and refuses a copy that would overwrite one. Runs in your store always count as yours, whatever source their documents name. `loopmath fit --without SOURCE` leaves out one shipped prior source, and `loopmath status` names the options of the current fit.

## The three views

Each view is a self-contained HTML file written by a command; opening one needs no server. The same numbers are available with `--json`.

- **Previous runs:** `loopmath runs --html`. Every recorded run with its workflow graph, costs, signals and receipt.
- **Plans for a task:** `loopmath recommend ... --html`. The candidates on a success-cost chart. Click one to see its workflow graph with per-piece predictions.
- **Current estimates:** `loopmath posterior --html`. What loopmath believes at each level (model, effort, role, topology, task type, repository, feature), each with a range. Add `--workflow CFG` (a configuration id from `loopmath recommend` or your runs) to see the estimate for each piece of one workflow, on the task its runs were for, or, if it never ran, the task it was recommended for; `--type` and `--repo` pick another.

Without a path, `--html` writes the page under `~/.loopmath/views/` and prints where it went.

The workflow builder is the one page that is served: `loopmath builder --type bug_fix --repo acme/api` serves it on 127.0.0.1 and opens it in your browser (`--no-open` prints the address instead); Ctrl+C stops it. Pick a shape, change models, efforts, widths and pieces, and see the estimated numbers for the task, the same ones `recommend` gives.

## Commands

| Group | Commands |
|---|---|
| Plan | `task-types`, `workflows` (list, show, validate, diff), `recommend`, `builder` |
| Record | `run` (start, record, import, attempt, artifact, finish, sessions), `outcome`, `budget`, `config` (get, set, unset) |
| Learn | `fit`, `onboard`, `share`, `prior` |
| View | `runs`, `posterior`, `status`, `report`, `doctor`, `verify-receipts` |
| Harness testing | `check-change`, `gate` (experimental), `screen` (experimental) |
| Task subsets | `tasks pick` (experimental) |
| Evolution loops | `adapt openevolve` (experimental), `adapt shinkaevolve` (experimental), `recommend --role` (experimental) |
| Skill | `skill` (install, uninstall, show) |
| OCP | `ocp` (validate, migrate) |
| Log analysis | `analyze`, `graph`, `adapt`, `prices`, `validate-prices` |

These are experimental: `gate`, `screen`, `tasks pick`, the harness-part estimates (`recommend --harness-variants` and `--harness`, `posterior --level harness` and its `factors`), `recommend --role`, `adapt harness2`, `adapt plugin-eval`, `adapt openevolve`, `adapt shinkaevolve` and the ADE adapters (`adapt NAME`). Each works and has tests, but has had little use on real work so far, and its output may change. Their help says so too. The gate at DSPy's GEPA and MIPROv2 accept step (`loopmath.integrations.dspy`, Python) is experimental too: it is tested against fakes written from their source, not yet with DSPy installed, and its guide says so. So is the opt-in cost model for arms that mix models, `loopmath.compare.model.COST_MODEL = "sw"`, a Python setting with no help of its own; the [changelog](CHANGELOG.md) describes it.

Each command prints a short summary by default. With `--json`, it prints exactly one JSON object instead. One exception: `loopmath adapt harness2`, `adapt harness-runs`, `adapt plugin-eval`, `adapt openevolve` and `adapt shinkaevolve` without `--out` print the run documents on stdout, one per line, so their `--json` summary goes to stderr. Exit codes: 0 ok, 1 user error, 2 not found, 3 not implemented, 4 store locked, 5 no fit yet. For flags, `loopmath <command> --help` is the source of truth. Commands, pages and the `runs` table show times in your local time with the zone; the store and `--json` keep UTC. A cost under a cent shows one significant digit (`$0.004`), and an exact zero is `$0`.

A workflow is a TOML file. `loopmath workflows list` shows the catalog and yours, which live in `~/.loopmath/workflows/`. It also lists the shapes loopmath composes from catalog pieces, such as `team`; `loopmath workflows show team` says what one is and that `recommend` does not offer it. This one implements, then sends a rejected change back once or twice:

```toml
id = "implement_review_mine"
version = 1
title = "Implement, then review"
edges = ["issue -> implement", "repo -> implement", "implement -> diff", "diff -> review", "review -> verdict"]

[[pieces]]
id = "implement"
role = "implementer"

[[pieces]]
id = "review"
role = "reviewer"

[[artifacts]]
id = "issue"
kind = "issue"

[[artifacts]]
id = "repo"
kind = "repo"

[[artifacts]]
id = "diff"
kind = "diff"

[[artifacts]]
id = "verdict"
kind = "verdict"

[control]
budget_rounds = 3             # rounds, counting the first; 1 means no repair

[[control.gates]]
after = "review"              # the review's verdict decides
on_fail = "implement"         # a rejected change goes back to implement

[settings.implement]          # optional; with a setting for every piece, the file is a configuration
harness = "claude-code"
model = "claude-opus-5-5"
effort = "high"

[settings.review]
harness = "codex"
model = "gpt-5.6-sol"
effort = "high"
```

Check it with `loopmath workflows validate FILE`, compare it with `loopmath workflows diff implement_review FILE`, and pass it to `recommend --workflow FILE` to have it considered, or to `run start --workflow FILE --source user_edit` to record a run with its settings (`--set PIECE=HARNESS:MODEL:EFFORT` overrides a piece).

## Harness parts

A run can record the harness it ran with: the prompt files, skills, plugins, MCP servers, hooks, sub-agents and commands in its working tree. loopmath stores a digest of those files and a list of named parts (`skill:pdf`, `mcp:github`, `hook:PostToolUse:Edit|Write`, `prompt:3f2a9c1d`), never a path, a hook command or an environment value. The fit then learns each part's effect on cost and success, as a change from its harness label.

- `loopmath run finish --harness auto` (or `run record --harness auto`) records the git working tree the run's attempts ran in; `--harness PATH` names a tree; `--harness none` records nothing. The default is `none`; `loopmath config set harness.record auto` makes `auto` the default, and the flag wins.
- `loopmath run import` keeps a record another tool wrote.
- `loopmath posterior --json` lists each part under `factors`, and `--level harness` shows the labels and their parts:

  ```json
  {"factor": "skill:pdf", "harness": "claude-code", "kind": "skill", "name": "pdf",
   "node": "hfactor:claude-code|skill:pdf", "runs": 12,
   "cost": {"mean": 0.87, "lo": 0.74, "hi": 1.02, "level": 0.8, "lo95": 0.68, "hi95": 1.11},
   "success": {"mean": 4.1, "lo": -1.0, "hi": 9.0, "level": 0.8, "lo95": -3.6, "hi95": 11.5,
               "log_odds": {"mean": 0.18, "lo": -0.04, "hi": 0.41, "level": 0.8, "lo95": -0.16, "hi95": 0.52},
               "calibrated": false}}
  ```

  `cost` is the multiplier on an attempt's dollars when the part is present; `success` is percentage points.
- `loopmath recommend --harness-variants` prices the trees your runs used most, and `--harness PATH` prices one more, against the bare harness label on the same draws.

An estimate is the change when a named part is present, pooled over its versions. It is not the effect of one edit, and a part on every run cannot be told apart from its harness. The cost ranges met their stated coverage in simulation. **The success ranges did not:** they pull effects toward 0, and their 80% ranges held 60% of nonzero effects. So they are marked `calibrated: false`, and so are the success-side numbers of a priced variant, and the harness-part estimates are experimental. With no record anywhere, every number is what it was before. The record, the digest, the factor map and the limits are in [`spec/harness.md`](spec/harness.md).

## Testing a harness change

These commands are for changing the harness your agents run with and finding out whether the change paid off.

- **Harness parts and `--harness`.** Record the tree each run used (`run finish --harness auto` or `PATH`), and the fit learns each named part's effect, as in the section above. `recommend --harness-variants` prices the trees your runs used on the configurations those runs used, and on the recommendation's own when their harness has data on the parts; `--harness PATH` prices one more tree. Where a tree is not priced, `not_priced` in `--json` says why. The record, the factor map and the limits are in [`spec/harness.md`](spec/harness.md).
- **`adapt harness-runs` and `adapt harness2`.** These bring in runs another tool made; `adapt harness2` is experimental, since its input format is read from harness2's code and it has not yet read a real runs folder. `loopmath adapt harness-runs FILE --out DIR` reads a `loopmath.harness-runs/0.1` file, one JSON line per rollout, as an auto-research loop, a plugin eval suite or a hand A/B writes it. `loopmath adapt harness2 RUNS --out DIR` reads a harness2 runs folder. Then `loopmath run import DIR --finish` stores the runs. The row format, outcomes, costs and worked examples are in [`spec/harness-runs.md`](spec/harness-runs.md).
- **`check-change`.** It decides whether a change beats what it replaces, on paired runs:
  - `check-change start --base LABEL|PATH --cand LABEL|PATH --n N` plans the pairs and each pair's arm order; `--base-set PIECE=HARNESS:MODEL:EFFORT` and `--cand-set` give each arm a setting, as `run record --set` takes it, for a model switch on one harness: the arm's runs take its set, a run with another is refused, and `status` names the models each arm ran and any run whose log ran another;
  - `check-change status` gives the comparison, the verdict once there is one (adopt or reject) and the next pair worth running. A check that reaches its cap or spends its budget before the evidence decides keeps the base and says `not decided`. The next pair's value and price are shown at each look and stop nothing; `--value-stop` also stops a check, not decided, when the next pairs are certified not worth their price (every plan's net below 0 by more than two standard errors, or no look left can change the action), from 4 scored pairs on (`--min-pairs K`). Use it when a pair is dear against what an adoption is worth over `--horizon-runs`: in simulated checks at the default 50 future runs it saved 6 to 16 pairs a check but dropped 26 to 40 percent of the adoptions of real cuts, and at 500 it dropped none (`check-change start --help`); `--pair-price 0` compares the value with 0 for runs on a subscription, and the budget still counts the runs' recorded dollars; `--json` keeps `verdict: reject` there, with `comparison.stop` and `comparison.decided: false`;
  - `check-change from-runs --base-label SEL --cand-label SEL` compares finished runs already in the store, picked by a run label, a configuration id, a harness digest (a prefix of one digest will do) or `arm:NAME`.

  `--rule` names what counts as accepted, and `loopmath check-change --help` lists the rest. A run with no dollar cost leaves the decision to the outcomes alone, and the cost reads unknown; a budget or `--objective cpar` then refuses, naming the runs without one. When an arm's runs mix a cheap model and a dear one, the default cost model can put the dearer arm's cost per run below the cheaper arm's. In Python, the experimental setting `loopmath.compare.model.COST_MODEL = "sw"` reads such costs for `compare.paired` and `gate.decide`; the [changelog](CHANGELOG.md) gives its limits.
- **`screen` (experimental).** It tests many harness edits at once. Put each edit in its own subfolder of an edits folder, laid over a base harness tree: `PATH` copies a file, `PATH.append` appends to one, `PATH.remove` removes lines. Then:
  - `screen start --base DIR --edits DIR --tasks FILE.jsonl --rule tests` plans one fold-over pair per task: arm A runs the edits its row turns on, arm B the others, so task difficulty cancels in the pair and every session informs every edit. 12, 16, 20, 24 or 32 pairs for up to one edit fewer give orthogonal rows;
  - `screen next` prints the next pair, with both arms' trees built in the store; `run record --screen SCR --arm A|B` records a run into it;
  - `screen status` ranks the edits on success and cost with 80% ranges and P(helps), names the predicted-best subset with its predicted cut in cost per accepted result (often a subset no arm ran), and prices the next pair. `--on EDIT,EDIT` predicts any subset;
  - `screen apply --best --out DIR` writes the base with the best edits, ready for a `check-change` against the base.

  The model is a Bayesian linear model of each pair's difference, on success and on log cost, with its noise and effect scales weighed over a grid rather than plugged in. A finished orthogonal screen weighs two views, a few edits matter (sparse) or all matter a little (dense), by how well each explains the pairs, and `best.by_view` shows both. On a dozen pairs each edit's effect is shrunk toward 0 (a large one by about half with 8 edits in 12 pairs), but in simulations at that size, of 1,000 screens drawn from effect settings no tuning used, 964 picked a best subset and its 80% range held the true cut in 812 (84%), a few points high where the true cut was small and low where it was large; the other 36 picked no edit, which predicts no change. The prediction is against the base, which no arm runs, so an effect of having the edited files at all is not in it. The [screen guide](docs/screen.md) has a worked example and the limits.
- **`gate` (experimental).** For a loop that adopts changes on its own, it answers: adopt the candidate, reject it and keep the base, or replicate N more pairs first. `loopmath gate --base SEL --cand SEL` reads the finished runs in the store, or `--runs FILE` reads `loopmath.harness-runs/0.1` rows; an arm is a harness digest (`sha256:<hex>` or `digest:sha256:<hex>`, a prefix of one digest will do), a configuration id, or `arm=NAME` or `arm:NAME`; `check-change from-runs` takes the same forms. `--rule` takes the built-in rules or a rule defined under `[rules.NAME]` in the store's config, which the gate reads as `check-change` does, also with `--runs`. The comparison of the paired runs is `check-change`'s, and the gate adds the loop's policy to it: no adoption before `--r-min` pairs, looks at 1, 2, 4 and so on pairs up to `--r-max`, a `--floor` (the smallest gain worth adopting, 0 or more; under an acceptance rule it applies to the cut in cost per accepted result, a fraction, or to the gain in the chance of acceptance when a cost is unknown), and the dollars, `--budget-usd` for both arms (0 buys no further pair) and `--max-cost-ratio`. With no evidence, an arm that matches no run or a rule that scores no pair, it exits 1 and names the arms or the rules that are there. Both dollar flags need every run's cost and refuse, naming the runs without one; with neither flag, a run with no recorded cost leaves the decision to the outcomes alone, and the cost reads unknown, never $0. When nothing resolves within the cap or the budget, it keeps the base and says where it stopped. The next pairs' value is the one `check-change` states, valued on the way the gate goes on (its looks, `--r-min` and the 95% band), and named beside the number; `check-change from-runs` cannot express the gate's looks, so on the same runs the two numbers can differ. The value stop is on by default: from `--min-pairs` scored pairs (default 4) the gate rejects, keeping the base, when the next pairs are certified not worth their price, as `check-change --value-stop` does, and a cost that is not finite never stops it; `--no-value-stop` turns it off, and the value and its price are then shown and stop nothing. Under a score rule the value needs `--value-usd` (dollars per point of gain), as in `check-change`. In Python, `loopmath.gate.decide(base_runs, cand_runs, GatePolicy(...))` gives the same answer and writes nothing; `loopmath gate --help` lists the rest.
- **`gate winner` (experimental).** A search loop that scores N candidates and keeps the best overstates the winner's gain: the best of N noisy scores is high partly by luck. `loopmath gate winner --scores FILE` reads every score the search made (a CSV or `loopmath.gate-winner-scores/0.1` JSON; candidates known only as no better than the best so far count too) and gives the winner's gain over its parent, shrunk by what the scores say about the candidates' spread and the noise, with 80% and 95% ranges and P(beats parent). `--runs FILE` reads harness-runs rows instead, one arm per candidate, with the spread from their replicates. Give the spread of one evaluation across reruns with `--spread X`, or an assumed range with `--spread LO:HI`. The action is `gate`'s on confirmatory runs of the parent and the winner (`--confirm FILE`), never on the search's own scores, and a projection says how likely the gate is to adopt and how many more pairs it will want. In Python, `loopmath.gate.winner.assess`; the method, the formats and the output are in [spec/gate-winner.md](spec/gate-winner.md), with worked examples from published searches in `spec/examples/gate-winner/`.
- **DSPy's GEPA and MIPROv2 (experimental).** An optimizer's accept step keeps a new prompt when its score on a few instances goes up, and on 3 instances that keeps many children that are no better. `loopmath.integrations.dspy` puts the gate there: `GEPAAcceptance` is a GEPA acceptance criterion (run GEPA with `use_merge=False`), `MIPROv2Gate` confirms MIPROv2's replacements, and `run([], [], more)` gates any compiled program against its student. It imports neither dspy nor gepa. See [the DSPy guide](docs/dspy.md).
- **`tasks pick` (experimental).** For a loop or a benchmark run that scores each candidate on some of a suite's tasks instead of all of them. `loopmath tasks pick --candidate SEL` reads the finished runs in the store, or `--runs FILE`, learns each task's difficulty and discrimination from the other arms' runs on it, and prints the candidate's suite score so far (the mean over the suite of its chance of acceptance per task) with 80% and 95% ranges, whether to stop, and the next task to run. From a store it writes that task's file and the `run record --task-file F --label KEY=VALUE --home PATH` line that records the run in the same store, so the next call sees it. It stops when the 80% range is as narrow as one full run of the suite would give (`--halfwidth` sets another target), or, with `--versus X` or `--versus-arm SEL`, when the score is above or below a number or another arm's at `--level`. `--method` chooses the next task: the largest expected cut in the score's variance (`loss`, the default), information, one fixed order, random, or stratified by predicted rate. The default learns fastest when the candidate behaves like the arms in the history; for a candidate unlike all of them (a new model rather than a change to a known one's harness), `--method stratified` or `random` keeps the ranges closer to honest at the cost of more sessions. A candidate that scores below every arm in the history (a broken setup, say) is estimated too high by every method, and its ranges can miss even at the full suite; under `--method random` the accepted count over the sessions, which every call prints, is the check. Every estimate prints both warnings, in text and in `--json` (`estimate.warnings`). With `--versus`, a bar within about 0.01 of the candidate's score is close to a coin toss for any method; to ask whether a change helps by at least some margin, pass the bar plus that margin. The ranges are approximate: they leave out the history fit's own error shared with other arms. `loopmath tasks pick --help` lists the rest.

## Does my plugin pay?

`claude plugin eval` grades a plugin's eval cases with and without the plugin. `adapt plugin-eval` (experimental) reads its result, and loopmath turns the per-case deltas into a decision: adopt, reject, or run more, with ranges on the gain and, given each run's cost with `--cost-from`, on cost per accepted result. It reads the file only and never runs Claude Code.

```
loopmath adapt plugin-eval result.json --out ocp --rows rows.jsonl
loopmath run import ocp --finish
loopmath check-change from-runs --base-label arm:without --cand-label arm:with
loopmath gate --base arm:without --cand arm:with --runs rows.jsonl --rule accept --r-max 24
```

Runs of one case are not new tasks: to learn whether the plugin helps on tasks like your cases, add cases rather than runs. With grades (`--outcome score`), `check-change from-runs` compares case by case. The import's summary prints the last two filled in for your suite. Which cases are compared, what the suite's one cost can and cannot say, how far the answer reaches and the provisional input layout are in [`docs/plugin-eval.md`](docs/plugin-eval.md).

## The store and privacy

The store is `~/.loopmath`. Set `LOOPMATH_HOME` or pass `--home` to put it elsewhere; the log parse cache moves with it, to `cache/` inside the store, unless you set `LOOPMATH_CACHE_DIR`. It holds:

- one OCP v0.3 document per recorded run;
- signals, receipts and fits;
- your config and your workflows;
- the views you write;
- the caches: the log parse cache, and the folder where `fit --full` compiles its model (when that compile fails, it also leaves the C code it tried in a `pytensor_compilation_error_*` file in the system temp folder and prints where).

Do not edit the store by hand; use the commands. `loopmath verify-receipts` checks the receipts in your store; give a file to verify a research receipt ledger instead.

Run files use the `metadata_only` privacy profile. They keep structure, models, tokens, dollars, paths and verdicts, but not transcript text.

`loopmath share --out FILE` writes a reduced copy you can send to us. It keeps task types, workflows, models, tokens, dollars and outcomes. Repository names are hashed. Titles, paths, commands, session ids and commit shas are removed. `loopmath share --preview` prints exactly what would leave. `--out FILE.json` writes plain JSON you can read; any other name, such as `share.json.gz`, writes gzip.

`loopmath prior import-shared FILE` adds a file another organization shared with you to your priors (either format; not one this store shared itself), and `loopmath prior remove-shared ORG` takes it out again; run `loopmath fit` after either.

## Analyzing logs without recording

The log commands work on their own, with no store or fit:

```sh
loopmath analyze                 # cost per accepted run from the last 14 days of local logs
loopmath graph --workspace NAME --format html --out graph.html
```

`analyze` reads `~/.claude/projects` and `~/.codex/sessions`. It grades the outcome evidence it finds, prices known models, and prints cost per accepted run with an 80% band. It also lists what it could not use and why; when no workflow configuration is eligible to compare, it names the largest comparison exclusion, its count, and what to try. `loopmath analyze --json` prints the same as one JSON object, and `--home PATH` keeps its parse cache in that store.

`loopmath prices` lists the packaged prices with the source of each. A rate marked "rate not confirmed" is a best public-price guess, and every dollar figure priced with one says so.

`graph` rebuilds which agent launched which, and which session read what another wrote. Its formats are `ocp`, `json`, `dot`, `run` and `html`. `graph --out FILE` and `adapt NAME --out FILE` write to any path you name: missing folders are created, an existing file is replaced, and a directory is refused. `graph` prints a short summary on stderr; `--verbose` adds every count of what was skipped or could not be placed. See the [workflow graph guide](docs/graph.md) and [adapters](docs/adapters.md). The ADE adapters (`adapt NAME`) are experimental: each is tested on a synthetic fixture written from its tool's format, not yet on a real store.

## More

- [OCP, the run format](spec/OCP.md): the normative note that goes with the [schemas](spec/), and the [versioning promise](spec/VERSIONING.md)
- [DSPy's GEPA and MIPROv2](docs/dspy.md): `loopmath.integrations.dspy`, the gate at their accept step
- [Plugin eval results](docs/plugin-eval.md): `adapt plugin-eval`
- [OpenEvolve and ShinkaEvolve runs](docs/adapters.md#openevolve-and-shinkaevolve): `adapt openevolve`, `adapt shinkaevolve` and `recommend --role`, which model to give an evolution loop's mutator
- [Release policy](docs/release.md)
- [Screening harness edits](docs/screen.md): `screen`, with a worked 48-session example and its limits
- [Packaged model prices](src/loopmath/prices.toml)

## License

MIT.
