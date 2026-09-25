# Changelog

## 0.2.0 (2026-09-25)

### Upgrading from 0.1

- A fit written by 0.1.0 or 0.1.1 is not read: the model's nodes changed (design version 3). `recommend` and `posterior` name the old fit and ask for `loopmath fit` (exit 5); `status` marks it not usable (`fit.usable` and `fit.problem` in `--json`) and `doctor` warns. Run `loopmath fit` once after upgrading.

### Skills

- `loopmath skill install` writes six skills, one per job (`loopmath`, `loopmath-onboard`, `loopmath-import-runs`, `loopmath-update-fit`, `loopmath-plan-task`, `loopmath-record-run`), each with a shared `reference.md` of the commands and JSON fields they use. The default target is every agent whose home exists; `--dir PATH` installs into another project. `skill show NAME` prints one skill.
- The installer touches only files it wrote: it lists them with their sha256 in `.loopmath-skills.json` beside the skill folders. A skill file you changed is kept with a note; any other file where a skill goes stops the install before its first write, naming the path. The 0.1 single skill is replaced or removed only when it is the text 0.1.0 or 0.1.1 shipped. A linked `AGENTS.md` keeps its link: the file it points to gets the block and keeps its mode.
- `doctor` warns, with the install command, when a skill is missing, older than this loopmath, or the 0.1 single skill is still there.

### recommend and run

- `recommend --json --brief` keeps what an agent reads (`rec`, `task`, `fit`, `rule`, `reference`, `goal`, `rescue`, `choices`, `message`, `notes`), under 10 KB. With `--html`, `page` is the path of the page it wrote.
- `run start --rec REC --choice KEY` starts an option of a stored recommendation (`goal`, `pair`, `reference`, `cheapest_run`) with its settings. The JSON gives each run's `label` and `piece_settings` (harness, model and effort per piece).
- Any `run start --rec` keeps the rule the recommendation was made for, a score target included. `--rule NAME` replaces it.
- A run stores the task as it was priced, never recommend's view of it (`group_chain`, `support`, `horizon`, `inherited_features`, `notes`), whether it starts from `--choice` or from a saved `recommend --json` given as `--task-file`.
- `loopmath run record` records a finished run in one call, by session id, or by working directory and time, reading the matched session logs, with test results and scores: the attempts, the costs from the logs, the outcome and the receipt (`loopmath.run.record/1`). Sessions of one piece that overlap in time are copies in one round, up to the piece's width.

### recommend: workflow search

- `recommend` searches eligible builder and recorded shapes across allowed settings, not only the catalog's workflows: every allowed model and effort per piece, and parallel copies that differ in model. It uses an exact front of cost versus the model's mean predictor and per-draw optima (the best configuration in each of 200 posterior draws), followed by rescoring with the full prediction and a one-piece polish of the pick and every reached curve row. Your usual workflow or the reference always stays in the comparison. The limits are in spec 05 section 1a: a curve row can miss a slightly cheaper configuration that is off the front, outside the rescored winners and not reached by the polish, and a shape outside the exact method is listed in `search.skipped`. With the shipped prior and a few dozen runs of your own it takes about 2 s.
- When the pick is not the reference, its chance is below the reference's, and a miss is priced (redone with the reference, or finished by a person), `goal.strategy` and the message say it in one sentence: try the pick first; if it misses (about N in 10 tasks), the reference rescues it or a person finishes it; the expected cost per accepted result against the reference alone.
- `recommend --json` gains `search` and `goal.strategy`; candidates carry `search: {on_front, wins}`, choices and curve rows carry `wins`, and search finds have the origins `front`, `thompson` and `polish`. `--brief` leaves `search` out.

### Task features and the horizon

- Declare your own task fields in config under `[features.<key>]`: a category, a yes or no, or a number with edges and labels; who fills it (`labeller` or `orchestrator`); optionally the types and repos it applies to. `loopmath config set features.<key>` checks the declaration and keeps the old one when it is wrong; `task-types` lists the fields. A run stores a declared value as given.
- A feature value enters the model once enough fitted tasks carry it (`min_tasks`). A task loopmath has seen takes the values its fitted runs agree on. The fit summary gains `features:` and `horizon:` lines.
- `--horizon TIME` on `recommend`, `posterior` and `run start`: the time a run has. Without it a task takes the horizon of its earlier runs, then of its subtype, then of its type and repo. A horizon counts relative to the fit's reference (the lower median of its timeboxed runs), so a task with the usual horizon gets the usual estimate; a time box against no time box counts only when a source has both kinds, and the fit's `horizon:` line says which applies. When every timeboxed run in the fit has the same horizon, another `--horizon` is priced from the prior's slope, and the task's note says so: its effect is from the prior, not the data. `recommend --json` shows `horizon`, `inherited_features` and `notes` in its task block.

### onboard

- A group of sessions nothing could type is stored as type `unknown`, and gets no usual workflow. loopmath's own labelling calls and short harness probes are skipped and counted (`groups.skipped`); list your own pipelines under config `onboard.skip`.
- `onboard --dry-run`, and `onboard` without `--labeler`, list each labeller this machine can run with its expected cost, and `none`.

### belief model

- Shapes: a workflow whose roles differ from the catalog workflow of the same id is its own shape, and parallel copies of one role form a group.
- Position x source and family x source: a price level measured in shipped data reaches your runs through the position and the model family, and your own runs set it for you.
- Antithetic draws: two fits of the same data give the same score means.
- Exploration gains are computed exactly: the look-ahead updates the chance of success by exact Bayes over the fit's draws, where a linearized update understated the value of trying a second workflow and left some fits with no `pair`.

### Pages

- The results page (`loopmath posterior --html`, same command) answers five questions in one scroll, each with a sentence: which workflow reaches your target (ranked by the chance to reach it, with your runs as dots), which model and effort, whether more spend is worth it, how sure the estimates are, and what was fitted. `posterior --target NAME>=X` sets the target; without it the page takes the newest recommendation's for the same type and repo. The earlier detail sections are still there, collapsed.
- The planning page (`loopmath recommend ... --html`) starts with the pick: the workflow graph, its settings, three numbers with their 80% ranges and the command to start it. The pick, the pair, the reference and the cheapest run copy as `loopmath run start --rec REC --choice KEY`, with the horizon the estimate used. Every option is below, cheapest per accepted result first.
- The posterior view JSON (`loopmath.view.posterior/1`) gains `results` (rows with `p_accepted`, and `chance_from`), `units`, `runs` and `score_name`; every 0.1 key keeps its meaning.
- With too few scored runs for a score estimate, both pages price and rank by the chance of an accepted result and say so in one sentence; the score estimate is shown apart where the fit has one.
- A second `--html` page written in the same second is named `-2`, `-3` instead of replacing the first.

### OCP

- The graph extractor writes `source_contract: loopmath_graph/1`. A document with the earlier `dagr_graph/1` reads as before.

## 0.1.1 (2026-09-24)

### recommend

- The workflows you recorded for the task's type and repo (else its type), designed runs included, are candidates with their own shape and width: up to 60, most runs first, and one-step edits of the first 30. A catalog or edited workflow that runs the same as a recorded one is listed once.
- With no usual workflow, the baseline is a reference: your best recorded workflow by expected cost per accepted result, labelled as such ("no usual workflow; reference: your best recorded workflow (...)"). Text and JSON never call a workflow you did not run "your usual".
- Codex offers `max` effort for the gpt-5.6 and gpt-6 models, and candidates use every effort you have run.
- The rescue is named once at the top. Each workflow shows its run cost with the median beside the mean, the expected rescue, and the cost per accepted result; a score target's chance to reach carries its 80% range.
- `recommend --json` is `loopmath.recommend/2`: `reference`, `numbers`, and `choices` (at most four options an agent can show before it asks one question); `usual` is null when you have none.
- `recommend --fit ID` and `posterior --fit ID` read a kept fit.

### fit, status and sources

- Runs in your store are yours, whatever their `task.source` says. A shipped source name never matches them: `fit --without rq1` leaves out only the shipped RQ1 runs, while `--without user` and `--without shared` still leave out your runs and your shared imports. When a stored run has the same id as a shipped one, your copy is used, and `run import`, `fit` and `status` say so once.
- `fit` is about four times faster (a block factorization, Newton steps, and lazy imports).
- `status` names the current fit's options and counts, for example `fit: fit_..., 0 min old, --without rq1, runs 1469 prior + 44 yours`.
- `posterior --subtype S --feature K=V` predicts for that task.

### run import, runs, ocp validate

- `run import` takes several files or a directory, with one refit at the end; `--no-fit` skips it.
- `ocp validate` ends with `N passed, M failed`.
- `runs RUN` works as `runs --run RUN`; the table shows the model and effort of each piece, and `--wide` adds the configuration id.
- Plurals and units read correctly (`1 run`, `heldout_perf 3140`).
- `plan` is hidden from `--help` and prints a note on stderr that it is experimental; its output is unchanged.

### onboard

- Reading the history uses up to 8 worker processes (one per CPU). Set `LOOPMATH_WORKERS=1` to keep everything in one process.

### OCP

- `cost.output_tokens` includes reasoning tokens; `reasoning_tokens` is a breakdown a reader never adds. The OpenCode adapter now adds reasoning into `output_tokens`.
- `model.raw` keeps a label the source wrote with the effort in it; a producer never builds one, and effort known separately goes in `effort`.
- The full-fields example records an accepted run without a rescue.

### Views

- The posterior page lists every configuration recorded for the task's type and repository, grouped by workflow graph, with no cap.
- A piece that runs several workers is drawn as one box per worker (up to 6), and configuration labels name the width (`best_of_n: 3 x gpt-5.6-sol/xhigh`).
- The plans page names the rescue once, adds an expected rescue column, shows the chance to reach a score target with its range and the median run cost beside the mean, and names the reference when you have no usual workflow.
