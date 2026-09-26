# Changelog

## 0.2.3 (2026-09-26)

### Prior

- The prior adds 40 new runs on the sweep harness, 20 each with gpt-6-sol (xhigh) and gpt-6-luna (low) as the implementer, 38 accepted (gpt-6-sol 20 of 20, gpt-6-luna 18 of 20). They are a second batch of the `sweep` source and pool with its earlier runs; `fit --without sweep` leaves out both batches. The prior now has 1,602 runs (sweep 700, e0 809, rq1 44, lanes 49).
- No real run date or clock time ships in any prior source (the price table's date and the build date stay). The `e0`, `sweep` and `rq1` runs now start at 1970-01-01 UTC and keep their real durations, as the `lanes` runs did in 0.2.2; the order of runs is as before, and on the same runs a fit's posterior is the same as with their real times. The prior's `built_at` is its build date only (`YYYY-MM-DD`, UTC).
- The `lanes` implement_review runs stop at the workflow's review budget of 3 rounds: a run that needed a fourth or fifth round to be approved counts as not accepted. 7 of the 29 turn from accepted to not accepted, all of them features. This lowers every model's chance on a feature task with no runs of your own by 0.04 to 0.09.
- With no runs of your own (solo at high effort, a feature task in a new repo), the chance of an accepted result and the run cost, 80% ranges (for the chance, over the prior's draws; for the run cost, of one simulated run), 0.2.2 after the semicolon: claude-opus-5-5 80% (46% to 99%; 90%), typical run $7.08, mean $33.12 ($0.87 to $77.07; $28.77); claude-fable-5-1 54% (8% to 95%; 68%), typical run $15.50, mean $52.98 ($1.15 to $127.72; $48.71); claude-sonnet-5 28% (2% to 71%; 33%), typical run $2.35, mean $9.40 ($0.26 to $23.91; $9.78); gpt-6-astra 61% (10% to 98%; 74%), typical run $3.88, mean $19.33 ($0.40 to $38.27; $19.09); gpt-6-sol 83% (52% to 99%; 86%), typical run $2.23, mean $7.08 ($0.30 to $17.07; $10.83); gpt-6-luna 32% (3% to 76%; 37%), typical run $0.06, mean $0.19 ($0.01 to $0.47; $0.29). gpt-6-sol and gpt-6-luna now have 21 and 20 runs of their own behind them (1 and 0 in 0.2.2). The chances are lower mostly because of the lanes review budget, and move toward the middle with the shared benchmark weight (below).
- On our own runs held out from the fit (the shipped `rq1` runs left out, 22 runs on a problem the fit has not seen), the predicted mean cost is 1.05x the actual, and 1.11x when the problem is named as a new one (0.2.2: 0.99x and 1.04x); all 22 costs and scores are inside their 80% ranges, as in 0.2.2.

### Benchmarks in the prior

- All efforts of one model on one benchmark share one weight: each of a model's k results on a benchmark gets 1/k of that benchmark's weight, and each effort keeps its own gap. A model with many published efforts no longer pulls harder than one with a single result.
- Each benchmark has its own weight in the shipped `benchmarks.toml`, chosen by leaving each model with runs out and predicting its runs from the benchmarks: all four stay at 5 runs (Terminal-Bench 4.0 and 2.1, pass rate and tokens); none moved. No other weight in 0, 1, 2, 10 or 20 lowered the held-out error by 5% for two thirds of the held-out models; the closest was 2.2%.
- `benchmark_prior_weight`: unset by default, which uses each benchmark's own weight (`config get` prints `(not set)`; it printed 5.0 in 0.2.2). A number replaces every benchmark's weight and is still shared over a model's efforts, so a value set in 0.2.2 now gives a model with five efforts about a fifth of its old pull; 0 turns benchmark priors off; a negative value is refused.

### The shipped prior is the one we fit with

- The release check proves the package carries our prior: every file of the installed prior bundle and `benchmarks.toml` has the same sha256 as the checkout it runs in, no file is missing on either side, and the run counts per source agree, for a local build and for `--online pypi` (run it from the released commit). `scripts/release-check.sh --prior-only DIR` runs that comparison alone on any installed `loopmath` folder.
- `loopmath onboard` says once what prior the answers start from, for example `Starting prior: loopmath 0.2.3, 1,602 runs (sweep 700, e0 809, rq1 44, lanes 49), 2 benchmarks (Terminal-Bench 4.0 and 2.1), built 2026-09-26.` (`prior` in `--json`). The onboard skill shows it to you; `loopmath prior show` starts with it; the results page shows it under its opening sentence, and names the fit's own starting prior instead when that fit was made by another loopmath version or left shipped runs or the benchmarks out (`--no-prior`, `--without`).
- The README says the prior ships inside the package, is the one we fit with, and is updated with each release.

### A new user sees the typical cost first

- While you have no finished runs of your own, `recommend`, the planning page, the builder page and the results page headline the typical run cost (the median) with its 80% range, show the mean on the next line ("higher because a few runs cost far more"), and say "Onboard to see your own costs". Runs in the shipped prior do not count as yours. Once you have runs, the mean comes first again, except where a workflow's 80% cost range is wider than 10x: there the typical run leads, without the onboarding line. Budgets, totals and the cost per accepted result still use the mean. The builder's chart plots the mean run cost, and while the typical run leads it says so on its axis, your build's label and a tapped point's card.
- JSON: the mean stays where it is. A prediction's `cost.usd` and `cost.tokens` gain `median`, from the same simulated runs as the 80% range; each `numbers.run_cost_usd` gains `typical_first`, the choice the pages follow; `recommend --json`, the builder's `/api/context` and the results page data gain `own_runs`.
- With no runs of your own, `recommend` for a feature task picks plan_implement_review gpt-6-sol/medium, gpt-6-luna/medium, gpt-6-sol/low: 90% in one run, a typical run $0.42 (mean $1.21), $2.59 per accepted result (0.2.2: plan_implement_review gpt-6-sol/medium, gpt-6-luna/xhigh, gpt-6-sol/low, 81%, $1.91 a run, $5.87 per accepted result). The reference, implement_review claude-opus-5-5/high, gpt-6-astra/xhigh, has a 66% chance, a typical run of $14.59 and a mean of $52.33 (0.2.2: 92%, mean $51.38).

### Fixes

- The builder's `/api/context` always lists the rescue workflow among the candidates (origin `rescue`, with its predicted numbers), as the spec says; before, it was missing when no search offered it.
- The builder page: a long workflow title shows the workflow's id when that is shorter (no special case for some titles); option rings that land on the same spot on the chart are moved apart, with a line to their true point.
- A share document with no source or org is the source `shared` (it was `shared:shared`).

## 0.2.2 (2026-09-25)

### Workflow builder

- `loopmath builder` serves the builder page: build your own workflow and see its estimates change with every edit. At the top, a chart of the chance to reach the target against cost (per run or per accepted result): your build is a large point that moves on each edit and leaves a trail of earlier builds, the options are numbered rings and the candidates dots; tap a point to start from it. In the middle, the workflow as a graph: tap a piece to change its model, effort, width or gate target on the piece itself; add and remove pieces, undo and redo. Below it, the next steps: every one-step change of your build, ranked by more chance, cheaper per accepted result or cheaper run; tap one to preview it on the chart, Apply to take it. Beside them, "Your build": its four numbers with their intervals, the change against the recommended option, and Copy build, a label to paste to your agent.
- `POST /api/predict_many` estimates a list of workflows in one call, in order, each exactly as `POST /api/predict` would.
- Every number carries `bands` (50, 80, 90 and 95% intervals), now also for the chance within the attempts. A prediction gives the expected review `rounds` and, for each piece a gate follows, `gate_pass`, the chance the gate approves.
- Errors and warnings name the piece they are about: `{piece, message}` (plain strings in 0.2.1).
- `/api/context` gains `goal_config_id`. The catalog counts the runs behind each model by role and effort (`runs_behind: {total, by_role, by_effort}`, a count in 0.2.1) and the runs of each role (`roles: [{id, runs}]`, names in 0.2.1).
- `--start CFG` opens any option's or candidate's configuration; with `--rec REC` the builder answers for a stored recommendation.

### recommend: the current models

- With neither `--models` nor `models.allowed` set, `recommend` and the builder offer the current models: claude-opus-5-5, gpt-6-astra, gpt-6-sol, gpt-6-luna, claude-sonnet-5 and claude-fable-5-1 (0.2.1 took the models of your usual workflow).
- A workflow on a model outside that list is retired: its runs stay data, and a usual workflow on it stays the reference line, labelled "(retired model)" (`reference.retired_models` in the JSON), but it is never the pick, a choice, a candidate or the rescue. With no candidate on the offered models, `recommend` exits 1 and says to name them with `--models` or `models.allowed`. The builder answers a retired model with an error on its piece.

### Planning page

- Each option has a line "Customize in the builder" with the command `loopmath builder --rec <rec> --start <config>` and a Copy button: the page is a file and cannot start the builder itself.

### Prior

- The prior adds Terminal-Bench 2.1 as run by Artificial Analysis (one harness, Terminus 2, for nine of the twelve current models, with token use). SWE-Bench Pro is not added: its public leaderboard lists none of the current models.
- A new shipped source, `lanes`: 49 runs of loopmath's own build lanes: 29 implement_review runs, claude-opus-5-5 implementing and gpt-6-astra reviewing, with their 72 review rounds and verdicts; 20 solo runs (9 on claude-opus-5-5, 9 on gpt-6-astra, one each on gpt-6-sol and claude-fable-5-1). No date or clock time ships: each run starts at 1970-01-01 UTC and keeps its real durations. `fit --without lanes` leaves it out.
- A model version's run cost moves with its list price. A version with few runs of its own is priced from its family's runs, scaled by its price against theirs; the price counts like 5 of its own runs (`meta.json` `price_offsets`). With no runs of your own (solo at high effort, a feature task in a new repo), mean run cost and 80% predictive range (10th to 90th percentile of one simulated run), 0.2.1 in brackets: gpt-6-sol $10.83 ($0.39 to $21.80; $18.85), gpt-6-luna $0.29 ($0.01 to $0.61; $0.75), claude-opus-5-5 $28.77 ($0.90 to $61.50; $32.12), claude-fable-5-1 $48.71 ($1.59 to $114.19; $72.72), gpt-6-astra $19.09 ($0.42 to $34.48; $18.98). These include the lanes runs and Terminal-Bench 2.1.
- With no runs of your own, `recommend` for a feature task now picks plan_implement_review gpt-6-sol/medium, gpt-6-luna/xhigh, gpt-6-sol/low: 81% in one run, $1.91 a run, $5.87 per accepted result (0.2.1: plan_implement_review gpt-6-astra/low three times, $10.80 a run, $15.35 per accepted result).

### Fixes

- `config get benchmark_prior_weight` shows 5.0, the weight the fit uses (it showed 1.0).
- An imported share is the source `shared:<org_hash>` in `fit`, `meta.json` and `fit --without`, the name `prior import-shared` prints (it was `shared:shared:<org_hash>`, so `fit --without shared:<org_hash>` failed). If you imported a share file, refit after upgrading: old fits keep the `shared:shared:<hash>` source name until the next fit.

## 0.2.1 (2026-09-25)

### recommend: cost per accepted result and the rescue

- A missed run is priced by retrying with a rescue workflow (rescue kind `retry`, the new default). Each further attempt on the same task has half the chance of the one before (`rescue.decay`, default 0.5), up to 3 attempts in all, counting the first run (`rescue.max_attempts`). The rescue cost is the mean spend of those retries over their mean chance of success, so draws near zero no longer hide behind the mean chance. `rescue.max_attempts = 1` means no retries. The kinds `redo_usual`, `person` and `none` stay as config options.
- The rescue workflow is the cheapest candidate whose chance is at least `rescue.min_chance` (default 0.70). When none reaches it, it is the most likely one, and `rescue.basis` says so. `rescue` gains `config`, `chance`, `run_cost_usd`, `decay`, `max_attempts`, `min_chance` and `p_accepted`.
- With a score target and no usual workflow, the reference is the recorded workflow with the highest chance to reach the target (ties: the lower run cost), no longer the one with the lowest cost per success.
- Every `numbers` object and every choice gain `p_accepted_within` (the chance of an accepted result within `attempts` attempts) and `bands` (50, 80, 90 and 95% central intervals for the chance, the run cost and the cost per accepted result).
- With a score target, `choices` gains `most_likely`, the workflow most likely to reach the target, unless it is already another choice. There are up to five choices, each with `option`, its number from 1 in array order.
- `recommend --brief` leaves `bands` out; the full `--json`, the stored recommendation and the page data keep them.
- The message and the plan skill say it plainly: the run cost is what the agents cost for one run; the cost per accepted result adds the expected cost of fixing a miss by retrying with the rescue workflow. The chance of one run and the chance within the attempts are shown apart.

### fit

- Two fits of the same data give the same output: every head's draws are seeded from the fit's data and settings, not from the fit id.
- Under a time box, effort and the workflow's shape (topology, position) move the predicted cost of a run less, since a timeboxed run's cost depends mostly on its horizon. The search prices timeboxed tasks with the same weights. Fits made before 0.2.1 keep their old weights.

### Skills and CLI

- The plan skill works outside a git repo: a repo the user names becomes `--repo`, and with no git checkout it leaves out `--base-commit` and says so in one line.
- The import and update-fit skills say once when your runs are also in a shipped prior (`shipped_overlap` in the `run import` and `fit` JSON).
- `run import --brief`: counts, failures, the overlap and the next step, in a few lines of JSON.
- Payback is one number: `payback_runs` (at least 1), used by the message and the skill.
- The plan skill lists the choices by option number, with the chance of one run, the run cost, the cost per accepted result and the chance within the attempts. You can paste "option 2" from the planning page.
- Top-level help: `run` lists `record`; `recommend` says it gives choices.
- `run start --choice` accepts `most_likely`, and run labels name each piece's width (`2 x model/effort`), as the choices and pages do.

### Workflow builder

- `loopmath builder` starts a local server on 127.0.0.1 for building your own workflow: `GET /api/context` gives the task, the choices, the candidates and the catalog of models and harnesses; `POST /api/predict` estimates a workflow you send, with the same numbers as `recommend`. The page it serves is a plain placeholder in this release.

### Pages and posterior text

- The planning page opens with a chart of chance against cost: every option as a point, numbered, with thin interval lines on both axes. Toggle between the cost per accepted result and the run cost.
- Where a page drew a distribution shape, it now shows a dot for the mean, a line for the 80% range and thinner lines for the 90% and 95% ranges.
- Tables on the planning, results and runs pages sort by a click on a column head.
- The planning page's button says "Copy option" and copies `option <n>: <label>`, to paste into the conversation with your agent. An info icon beside the run cost and the cost per accepted result explains the two, and a column shows the chance within the attempts.
- `loopmath posterior` (text) leads with the target metric and the models you ran. Rows about models never run here are hidden, with one line saying how many; `--all` shows them. `--json` is unchanged.

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
