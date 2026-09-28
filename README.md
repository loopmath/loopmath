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

With `--run`, a session that started before the run counts from the run's start (a Codex session counts whole). Give `--since TS` and `--until TS` when the task's work began or ended at another time; `--since` before `run start` moves the run's start back. When no commit is found after the base commit, the record says where it looked; pass `--commit SHA`. `loopmath run sessions` lists the recent Claude Code and Codex sessions in this folder with their ids, for `--session ID`. Outside a Claude Code tool shell, `--session self` uses the one Claude Code session in this folder active in the last 10 minutes.

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

A run counts as a success when it is accepted under your rule. By default that means the task's tests pass. A rule can also be a score target: `loopmath recommend ... --target 'heldout_perf>=2400'` or `--target 'runtime_s<=200'` gives the expected score and the chance of reaching the target. Set standing rules and other options with `loopmath config set KEY VALUE` (`loopmath config get` lists every key, set or not, and `config set` refuses a key loopmath does not read), and the spending cap with `loopmath budget --usd X --period month`. Spend counts your runs and onboard's labelling calls; runs onboard brought in from your logs are shown apart and not counted. A run you record with `--source habit` counts.

Runs recorded by another tool that writes OCP v0.3, such as an experiment runner, come in with `loopmath run import DIR --finish`: several files or a directory, with one refit at the end (`--no-fit` skips it). It says how many runs were new and how many were already in the store; importing finished runs into a store with no fit starts a fit. Runs in your store always count as yours, whatever source their documents name. `loopmath fit --without SOURCE` leaves out one shipped prior source, and `loopmath status` names the options of the current fit.

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
| Record | `run` (start, record, import, attempt, artifact, finish, sessions), `outcome`, `budget`, `config` (get, set) |
| Learn | `fit`, `onboard`, `share`, `prior` |
| View | `runs`, `posterior`, `status`, `report`, `doctor`, `verify-receipts` |
| Skill | `skill` (install, uninstall, show) |
| OCP | `ocp` (validate, migrate) |
| Log analysis | `analyze`, `graph`, `adapt`, `prices`, `validate-prices`, `scan` |

Each command prints a short summary by default. With `--json`, it prints exactly one JSON object instead. Exit codes: 0 ok, 1 user error, 2 not found, 3 not implemented, 4 store locked, 5 no fit yet. For flags, `loopmath <command> --help` is the source of truth. Commands, pages and the `runs` table show times in your local time with the zone; the store and `--json` keep UTC. A cost under a cent shows one significant digit (`$0.004`), and an exact zero is `$0`.

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

## The store and privacy

The store is `~/.loopmath`. Set `LOOPMATH_HOME` or pass `--home` to put it elsewhere; the log parse cache moves with it, to `cache/` inside the store, unless you set `LOOPMATH_CACHE_DIR`. It holds:

- one OCP v0.3 document per recorded run;
- signals, receipts and fits;
- your config and your workflows;
- the views you write;
- the caches: the log parse cache, and the folder where `fit --full` compiles its model (when that compile fails, it also leaves the C code it tried in the system temp folder and prints where).

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

`analyze` reads `~/.claude/projects` and `~/.codex/sessions`. It grades the outcome evidence it finds, prices known models, and prints cost per accepted run with an 80% band. It also lists what it could not use and why; when no workflow configuration has enough runs to compare, it says how many runs it read and what kept each out, and what to try (`--min-n`, `--grading`). `loopmath analyze --json` prints the same as one JSON object, and `--home PATH` keeps its parse cache in that store.

`loopmath prices` lists the packaged prices with the source of each. A rate marked "rate not confirmed" is a best public-price guess, and every dollar figure priced with one says so.

`graph` rebuilds which agent launched which, and which session read what another wrote. Its formats are `ocp`, `json`, `dot`, `run` and `html`. `graph --out FILE` and `adapt NAME --out FILE` write to any path you name: missing folders are created, an existing file is replaced, and a directory is refused. `graph` prints a short summary on stderr; `--verbose` adds every count of what was skipped or could not be placed. See the [workflow graph guide](docs/graph.md) and [adapters](docs/adapters.md).

## Research data

`loopmath research --help` lists the verbs that read research data folders, which are not part of the package: `research fit` and `research transfer-test` read sweep run records, and `research analyze-e0` reads a session corpus. Name the folders with a flag (`--sweep-dir`, `--corpus`), an environment variable (`LOOPMATH_SWEEP_DIR`, `LOOPMATH_E0_CORPUS`), or config (`loopmath config set research.sweep_dir PATH`, `research.e0_corpus`). `research fit` and `research transfer-test` need the `bayes` extra (`pip install 'loopmath[bayes]'`), and `research analyze-e0` the `e0` extra. `loopmath prior build --out DIR` rebuilds the packaged prior bundle from research data folders like these; `loopmath prior build --help` names them.

## More

- [OCP, the run format](spec/OCP.md): the normative note that goes with the [schemas](spec/), and the [versioning promise](spec/VERSIONING.md)
- [Release policy](docs/release.md)
- [Packaged model prices](src/loopmath/prices.toml)

## License

MIT.
