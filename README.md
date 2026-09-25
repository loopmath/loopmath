# loopmath

loopmath helps an orchestrator agent choose how to run a coding task, and learns from how it went.

Before a task, it predicts how the user's usual agent workflow will do: the chance of an accepted result, and the cost in tokens and dollars. It shows the other workflows on a success-cost curve and offers two exploration picks. Each pick comes with its gain on every future similar run, its one-time price, and the number of runs it takes to pay back. After the task, the orchestrator records what ran and what came of it. loopmath fills in the token costs from the Claude Code and Codex logs, writes a receipt of predicted against actual, and refits.

loopmath never starts or coordinates agents. Claude Code, Codex, herdr or your own tool does that; loopmath is a CLI they call. Everything runs locally. No command opens a network connection, the logs are opened read only, and loopmath writes only to its store (`~/.loopmath`) and to paths you name.

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
loopmath doctor                        # what loopmath can see: logs, agents, store, fit, prices, skill
loopmath skill install --target both   # teach Claude Code and Codex how to use loopmath
```

`skill install` writes one `SKILL.md` for Claude Code (under `~/.claude/skills/loopmath/`) and one for Codex (under `~/.codex/skills/loopmath/`). If your Codex has no skills folder, it writes the file to `~/.codex/loopmath/SKILL.md` instead and adds a short marked block to `~/.codex/AGENTS.md` that points to it. Add `--scope project` to install into the current repository. It is safe to run again, and it rewrites only its own files. `loopmath skill uninstall` removes them.

Next, turn your history into a starting point:

```sh
loopmath onboard --labeler claude:claude-haiku-4-5 --dry-run    # see what it found and what labelling costs
loopmath onboard --labeler claude:claude-haiku-4-5 --yes        # record it and run a first fit
```

`onboard` reads your recent Claude Code and Codex sessions (the last 90 days unless you pass `--since`). It groups them into runs, labels each group's task type with a model you choose, names your usual workflow per task type and repository, and fits. loopmath never picks the labelling model for you. `--labeler` takes one of:

- `claude:<model>`, through your `claude` CLI;
- `codex:<model>`, through your `codex` CLI;
- `command:<cmd>`, any local command such as `command:ollama run <model>`, so nothing leaves the machine;
- `none`, to skip labelling.

Reading the history uses up to 8 worker processes (one per CPU, at most 8). Set `LOOPMATH_WORKERS=1` to keep everything in one process, or another number to change the count.

loopmath ships with a prior built from our own sweeps and experiments, so `recommend` gives an answer before you have any history. Your own runs then move it.

## The loop

The skill tells an orchestrator agent to follow these steps. You can run the same commands by hand.

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

A pair runs the goal workflow and an exploration pick from the same base commit, in separate worktrees. Save the recommendation (`recommend ... --json > rec.json`) and start both runs from it with `run start --task-file rec.json`, so they share one task: the first with `--new-slate`, the second with `--slate SLT`. A blinded referee then picks the better change, which you record with `loopmath outcome --slate SLT --prefer RUN --judge referee --blinded`.

A run counts as a success when it is accepted under your rule. By default that means the task's tests pass. A rule can also be a score target: `loopmath recommend ... --target 'heldout_perf>=2400'` or `--target 'runtime_s<=200'` gives the expected score and the chance of reaching the target. Set standing rules and other options with `loopmath config set KEY VALUE`, and the spending cap with `loopmath budget --usd X --period month`.

Runs recorded by another tool that writes OCP v0.3, such as an experiment runner, come in with `loopmath run import DIR --finish`: several files or a directory, with one refit at the end (`--no-fit` skips it). Runs in your store always count as yours, whatever source their documents name. `loopmath fit --without SOURCE` leaves out one shipped prior source, and `loopmath status` names the options of the current fit.

## The three views

Each view is a self-contained HTML file written by a command; there is no server. The same numbers are available with `--json`.

- **Previous runs:** `loopmath runs --html`. Every recorded run with its workflow graph, costs, signals and receipt.
- **Plans for a task:** `loopmath recommend ... --html`. The candidates on a success-cost chart. Click one to see its workflow graph with per-piece predictions.
- **Current estimates:** `loopmath posterior --html`. What loopmath believes at each level (model, effort, role, topology, task type, repository, feature), each with a range. Add `--workflow CFG` (a configuration id from `loopmath recommend` or your runs) to see the estimate for each piece of one workflow.

Without a path, `--html` writes the page under `~/.loopmath/views/` and prints where it went.

## Commands

| Group | Commands |
|---|---|
| Plan | `task-types`, `workflows` (list, show, validate, diff), `recommend` |
| Record | `run` (start, attempt, artifact, finish, import), `outcome`, `budget`, `config` (get, set) |
| Learn | `fit`, `onboard`, `share`, `prior` |
| View | `runs`, `posterior`, `status`, `report`, `doctor` |
| Skill | `skill` (install, uninstall, show) |
| OCP | `ocp` (validate, migrate) |
| Log analysis | `analyze`, `graph`, `adapt`, `prices`, `validate-prices`, `scan`, `verify-receipts` |

Each command prints a short summary by default. With `--json`, it prints exactly one JSON object instead. Exit codes: 0 ok, 1 user error, 2 not found, 3 not implemented, 4 store locked, 5 no fit yet. For flags, `loopmath <command> --help` is the source of truth.

A workflow is a TOML file. `loopmath workflows list` shows the catalog and yours, which live in `~/.loopmath/workflows/`. This one implements, then sends a rejected change back once or twice:

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

Check it with `loopmath workflows validate FILE`, compare it with `loopmath workflows diff implement_review FILE`, and pass it to `recommend --workflow FILE` to have it considered, or to `run start --workflow FILE` to record a run with its settings (`--set PIECE=HARNESS:MODEL:EFFORT` overrides a piece).

## The store and privacy

The store is `~/.loopmath`. Set `LOOPMATH_HOME` or pass `--home` to put it elsewhere; the log parse cache moves with it, to `cache/` inside the store, unless you set `LOOPMATH_CACHE_DIR`. It holds:

- one OCP v0.3 document per recorded run;
- signals, receipts and fits;
- your config and your workflows;
- the views you write;
- the log parse cache.

Do not edit the store by hand; use the commands.

Run files use the `metadata_only` privacy profile. They keep structure, models, tokens, dollars, paths and verdicts, but not transcript text.

`loopmath share --out FILE` writes a reduced copy you can send to us. It keeps task types, workflows, models, tokens, dollars and outcomes. Repository names are hashed. Titles, paths, commands, session ids and commit shas are removed. `loopmath share --preview` prints exactly what would leave.

## Analyzing logs without recording

The log commands work on their own, with no store or fit:

```sh
loopmath analyze                 # cost per accepted run from the last 14 days of local logs
loopmath graph --workspace NAME --format html --out graph.html
```

`analyze` reads `~/.claude/projects` and `~/.codex/sessions`. It grades the outcome evidence it finds, prices known models, and prints cost per accepted run with an 80% band. It also lists what it could not use and why.

`graph` rebuilds which agent launched which, and which session read what another wrote. Its formats are `ocp`, `json`, `dot`, `run` and `html`. See the [workflow graph guide](docs/graph.md) and [adapters](docs/adapters.md).

## Research data

`loopmath research fit|transfer-test` and `loopmath analyze-e0` read data folders that are not part of the package. Name them with a flag (`--sweep-dir`, `--corpus`), an environment variable (`LOOPMATH_SWEEP_DIR`, `LOOPMATH_E0_CORPUS`), or config (`loopmath config set research.sweep_dir PATH`, `research.e0_corpus`). The research verbs need the `bayes` extra (`pip install 'loopmath[bayes]'`).

## More

- [OCP, the run format](spec/OCP.md): the normative note that goes with the [schemas](spec/), and the [versioning promise](spec/VERSIONING.md)
- [Release policy](docs/release.md)
- [Packaged model prices](src/loopmath/prices.toml)

## License

MIT.
