# Changelog

## 0.3.1 (2026-10-19)

Fixes for problems found in 0.3.0. Four features are marked experimental, and `scan`, `prior build` and `research` leave the help. Under `--json`, an error is one JSON object on stdout. An unknown cost no longer reads as $0. `check-change` says `not decided` when it stops before the evidence decides, shows the next pair's value and price without stopping on them unless asked, and judges a pair's arm order by when its runs started. Fits, predictions and recommendations are 0.3.0's.

### Labels and help

- `gate`, the harness-part estimates (`posterior --level harness`, `recommend --harness-variants` and `--harness`), `adapt harness2` and the ADE adapters (`loopmath adapt NAME`) say experimental in their help and the README. They work as in 0.3.0.
- `check-change` is not marked experimental: the next pair's value is now advisory, and `--value-stop` gives 0.3.0's stop (see `check-change` below).
- `scan`, `prior build` and `research` are off `loopmath --help`, `prior --help` and the README. They stay callable by name.

### Errors and costs

- Under `--json`, a command that fails, a usage error included, prints one `loopmath.error/1` object on stdout: `ok: false`, `command`, `error`, `hint` when the error has one, and `exit`. stderr and the exit code are as before, and a command that prints its own JSON error, as `gate` and `check-change` do, is left alone. This fixes the 0.3.0 known issues for `fit`, `posterior`, `recommend` and `runs` and for usage errors (#1).
- `runs`, `status` and `budget` never show an unknown cost as $0. With no dollar cost on any run they say `cost unknown` (`unknown` in `status`), and with some, `$X known` and how many runs have none; tokens likewise. When every run has its cost, the lines are as before (#5).
- A store path (`--home` or `LOOPMATH_HOME`) that is a file, or has a file on its path, gives one error line naming it and exits 1, with no traceback (#15).
- `adapt atrium` writes its document. The adapter emits events only, and 0.3.0 refused such a document as empty.
- `run attempt --piece P` takes the harness and the model from the run's configuration for the piece, as it already took the effort. `--harness`, `--model` and `--effort` still win, and a piece with no configuration names the flags it needs.
- `verify-receipts --json` prints one `loopmath.verify-receipts/1` object, for the store's receipts and for a research ledger. The text output is unchanged.

### `check-change`

- A check that stops before P(better) decides, at its cap, with its budget spent or under one pair, or with `--value-stop` when the next pair is worth less than its price, says `not decided: ..., the base stays` and names the flag that goes on: `--n` or `--tasks` for a check, `--max-pairs` for `from-runs`. 0.3.0 said `verdict: reject`. `--json` keeps `verdict: reject` and `stop`, and adds `comparison.decided`, true only when P(better) decided (#2).
- The next pair's value no longer stops a check by default. It is priced and shown at each look, with its price, and the check goes on to a decision, its cap or its budget. In 0.3.0 the value stop could end a check after one or two pairs, and at the default 50 future runs it ended most checks of a real cut before the evidence could decide. `--value-stop` on `start` and `from-runs` turns it back on, from 4 scored pairs on (`--min-pairs K` changes the number, and `--min-pairs 1` gives 0.3.0's stop); `--min-pairs` without `--value-stop` is refused. `comparison.value_stop` and `comparison.min_pairs` report them (#4).
- `gate` decides as before: it keeps its own stop on the next pair's value. Its reason line for that stop reads `the next pair is not worth buying: worth $X, costs $Y`, where 0.3.0 said that compare stopped buying pairs (#4).
- A pair is out of order only when the run of its planned second arm started first, by the runs' start times, so arms run at the same time are in order whichever is recorded first. A pair is judged once both its runs are recorded. When a start time is missing, the record order decides, as before. `out_of_order` entries carry `by`: `start` or `record` (#7).
- Counts take the singular for 1: "1 pair", "1 candidate run". The reasons name only the arms that have runs, and a $0 cost is "counted as a tiny positive amount" (#16).

### Upgrading from 0.3.0

- Under `--json`, a failing command prints a JSON object on stdout where 0.3.0 printed nothing there. A script that took any stdout as success reads `ok` or the exit code.
- `check-change` no longer stops on the next pair's value unless `--value-stop` is given: where 0.3.0 gave `reject` with `stop: value`, it gives `more` and goes on to a decision, its cap or its budget. Pass `--value-stop --min-pairs 1` for 0.3.0's stop.

### Known issues

- `posterior` shows the first score in its levels when no `--head score:NAME` is given. After an import that can be a score your own runs do not have; pass `--head score:NAME` for the one you want (#6).
- `recommend` marks a reference on a retired model only with "(retired model)", and does not say plainly that the reference is not offered because its model is retired (#8).

## 0.3.0 (2026-10-19)

Testing a harness change. A run can carry a record of its harness tree, and the fit learns each named part's effect on cost and success. Runs another tool made come in through two run importers. `check-change` decides whether a change beats what it replaces, on paired runs, and `gate` answers adopt, reject or replicate for a loop that adopts changes on its own. A store with no harness record anywhere gives 0.2.8's numbers exactly.

### Harness parts and `--harness`

- A harness record (`loopmath.harness/0.1`, `spec/harness.md`) sits on a run at `run.ext["dev.loopmath.harness"]`: a digest of the tree's harness files and its named parts, such as `skill:pdf`, `mcp:github`, `hook:Stop` and `prompt:<sha8>`. It holds no path, hook command or environment value. `loopmath.harness` builds one from a tree (`record_from_tree`, `digest`, `factors`), reading files only.
- `run finish` and `run record` take `--harness PATH|auto|none`. `auto` is the git working tree the run's attempts ran in, and it refuses when they ran in two. The default is `none`, and the config key `harness.record` (`none` or `auto`) changes it. `run import` keeps a record another tool wrote.
- In the cost and success heads (and the score heads), each part is a node under its piece's harness label, shrunk by its own fitted scale. A part with no data adds its scale's uncertainty and nothing else.
- `posterior --json` lists each part's effect under `factors`, with 80 and 95% ranges and a note, and `--level harness` shows the labels with their parts. The success ranges carry `calibrated: false`, and the text output says so in one line. In simulation, their 80% ranges held 60% of nonzero effects. The cost ranges met their stated coverage.
- `recommend --harness-variants [N]` prices the N trees your runs used most, and `--harness PATH|auto` prices one more. Each is priced for the default pick's and the goal's configurations against the bare harness label on the same draws, with `p_better` and `delta_p_success`. The success-side numbers carry `calibrated: false`. The search itself is unchanged.
- Shared exports and prior bundles do not carry the record.

### Runs from other tools

- `loopmath adapt harness-runs FILE --out DIR` reads a `loopmath.harness-runs/0.1` file (`spec/harness-runs.md`), one JSON line per rollout, as an auto-research loop, a plugin eval suite or a hand A/B writes it. `loopmath adapt harness2 RUNS --out DIR` reads a harness2 runs folder, and `--rows FILE` also writes its rollouts as rows. Both write one OCP v0.3 document per run, and `loopmath run import DIR --finish` stores them.
- `run import` of a document already imported changes nothing when its content is the same: the run, its finish state and the outcomes added since all stay, and the summary counts it `unchanged`. A document whose run id is in the store with other content is refused by name, with the fields that differ, and the command exits 1. `--replace` takes the new copy and keeps the outcomes added after the import, and refuses a copy that would overwrite one. The summary counts new, unchanged and replaced runs.

### `check-change`

- `check-change start --base LABEL|PATH --cand LABEL|PATH --n N` plans the pairs and each pair's arm order, `check-change status` gives the comparison, the verdict once there is one and the next pair worth running, and `check-change from-runs --base-label SEL --cand-label SEL` compares finished runs already in the store, picked by a run label, a configuration id, a harness digest (`sha256:<hex>` or `digest:sha256:<hex>`, or a prefix of one digest) or `arm:NAME`. A side that matches no finished run exits 1 and lists the arms there, a prefix of more than one digest lists its matches, and a path that does not exist exits 2, as in `gate`.
- With no `--rule`, it judges the runs by the rule `runs` and `recommend` apply to them. Runs that resolve to different rules, a score whose runs disagree on which way it is better, and an unknown rule or score name are refused, naming the score names there and the rules that score at least one of the runs, one name each. A rule that scores no pair says so, with verdict more, and names the rules that score the runs.
- An unknown cost is never a reason to reject. When a run has no dollar cost, the decision is on the outcomes alone: the gain, with no stop on the next pair's value. The output says the cost is unknown and how many runs have none, and `--json` gives `comparison.outcomes_only`. A budget, or `--objective cpar` by name, needs every run's cost and refuses with exit 1, naming the runs without one; `status` still shows the check.
- `--json` gives `comparison.verdict` (adopt, reject or more) and `comparison.stop`, why the check stops: decided, cap, budget or value.

### `gate`

- `loopmath gate --base SEL --cand SEL` answers adopt the candidate, reject it and keep the base, or replicate N more pairs first, on the finished runs in the store or on the rows of `--runs FILE`. An arm is a harness digest (`sha256:<hex>`, or a prefix of one digest), a configuration id or `arm=NAME`; `digest:sha256:<hex>` and `arm:NAME` work too. `--rule` also takes a rule defined under `[rules.NAME]` in the store's config, which the gate reads as `check-change` does, also with `--runs`. In Python, `loopmath.gate.decide(base_runs, cand_runs, GatePolicy(...))` gives the same answer and writes nothing.
- The comparison is `check-change`'s, and the gate adds the loop's policy: no adoption before `--r-min` pairs, looks at 1, 2, 4 and so on pairs up to `--r-max`, a futility stop, a `--floor`, a `--budget-usd` for both arms (0 buys no further pair) and a `--max-cost-ratio`. When the cap or the budget makes the current pair count the last look, or a count is not a planned look (one past `--r-max` included), a reason line says so. When nothing resolves, it keeps the base and says where it stopped.
- Under an acceptance rule, the measure is the cut in cost per accepted result, a fraction, shown beside the gain in the chance of acceptance; when a cost is unknown, the measure and the floor are the gain in the chance of acceptance. The floor line names the measure and its unit.
- No evidence is an input error, never a replicate: an arm that matches no run, or a rule that scores no pair, exits 1 and names the arms or the rules and score names there.
- An unknown cost is never a reason to reject. `--budget-usd` and `--max-cost-ratio` need every run's cost and refuse, naming the runs without one. With neither flag, a run with no recorded cost leaves the decision to the outcomes alone, the next pair's value stops nothing, and the cost reads `unknown`, never $0, with the count of runs that have none.
- Errors name the flags, and under `--json` each is one JSON object on stdout; a missing `--runs` file exits 2.

### Upgrading from 0.2.8

`run import` no longer replaces a stored run whose document changed: it refuses it and exits 1. A script that re-imports changed documents on purpose passes `--replace`.

### Known issues

- Under `--json`, `fit`, `posterior`, `recommend` and `runs` print an error on stderr only, with no JSON object on stdout. This predates these changes.
- `posterior` shows the first score in its levels when no `--head score:NAME` is given. After an import that can be a score your own runs do not have; pass `--head score:NAME` for the one you want.
- `recommend` marks a reference on a retired model only with "(retired model)", and does not say plainly that the reference is not offered because its model is retired.
- `LOOPMATH_HOME` set to a file, not a folder, makes `status` stop with a traceback. This predates these changes.
- Under `--json`, a usage error (a missing flag, or a value of the wrong type) prints the usage and the error on stderr only and exits 2, with no JSON object on stdout. Every command does this, `gate` and `check-change` included.

## 0.2.8 (2026-10-12)

Token counts that the dollars beside them can buy, gpt-6.1-sol as the current sol at its list price, and a run line that puts its tokens next to the cost they belong to. With the same store and the same model list, every number that is not a token count is unchanged.

- A prediction's tokens now come from its dollars. The fit models the price per token, and each simulated execution's tokens are its cost over that price, with the price never below the model's cache-read rate, the cheapest rate any token bills at. Before, tokens came from a model of their own, and a piece could show more tokens than its dollars buy at its model's cheapest rate. Now a piece's mean, median and range, its cost per round, and a run's mean, median, range and expected cost with the rescue all hold at least the dollars that rate charges for their tokens. Dollars, chances, rounds and choices are unchanged.
- `posterior --head tokens` and the token rows of the levels show a node's tokens multiplier as its cost multiplier over its price-per-token multiplier. That is the effect before the cache-read floor: where the floor binds, the predicted tokens move less. The fit summary, `meta.json` and the posterior's data tab list the new `rate` head where they listed `tokens`.
- gpt-6.1-sol joins the catalog in the sol family, at its official list price: $2 input, $0.10 cached input, $2.50 cache write and $10 output per million tokens. It replaces gpt-6-sol in the default model list. It has no runs of its own: the sol family's runs inform it, and its list price sets its cost through the price offset. With no runs of your own (solo at high effort, a feature task in a new repo), its mean run cost is $5.05 (80% of runs $0.16 to $9.14), against gpt-6-sol's $7.08 ($0.31 to $14.85), with the same 83% chance. gpt-6-sol stays in the catalog and the price table; with the default list it is a retired model, so its runs stay data and a usual workflow on it is still the reference line. A store with `models.allowed` set keeps its list, and `--models` picks either.
- gpt-6-luna's row was checked against its official page again on 2026-10-12 and is unchanged; no newer luna is listed. The table's `as_of` stays 2026-08-31: a row with no check date of its own takes it as its tariff date, and those rows were not read again.
- A run line in `recommend` reads "$X a run (N tokens; median $Y)": the mean run cost, its tokens, then the median run cost. Before, the tokens came after the median, where they read as the median run's.

### Upgrading from 0.2.7

Run `loopmath fit` once. A fit made by 0.2.7 is not loaded: `recommend` and `posterior` stop and say to run `loopmath fit`, `status` and `doctor` say the fit is not usable, and the next `run finish` starts a fit in the background. On the same data and the same model list, the new fit gives 0.2.7's seeds, dollars, chances, rounds and choices; only the token counts move.

## 0.2.7 (2026-10-06)

Runs judged by their review, shared outcomes that reach the store that imports them, and `outcome.q` that moves only the chance of success. Predictions for stores with no imported share, no `outcome.q` set, and no onboarded or marked-imported run that has a review verdict and no tests verdict are unchanged.

- A run `onboard` brought in from the logs, or a run stored by `run import` from 0.2.7 on, that has a `review` or `accepted` verdict and no `tests` verdict is judged by its latest review: accept, approve or pass is a success, reject or fail is not. Any other value, for example `changes_requested`, gives no outcome, and the reason names the value. A review verdict on a reviewer's last attempt counts twice: as the review gate's pass and as the run's success.
- `loopmath runs` and the runs page list the other finished runs with a review verdict, no tests verdict and no known outcome, with the command that moves them: `loopmath outcome --rule review RUN...`. The command refuses a run with a tests verdict, and writes nothing if any run is refused. It re-scores a finished run's receipts at once and refits once. `loopmath outcome --undo-rule RUN...` undoes it, restoring each run's rule exactly as it was; naming that rule (`tests` when it had none) does too, unless the run's own rule was also named review. Runs imported before 0.2.7 carry no mark, so they appear in this list.
- `run finish`, receipts, the runs view and `share` read a run's stored rule as the fit does: a rule with no name is the default rule, `window_days: 0` is the default window, and `excludes_events: []` is the default exclusions. The next fit, in the foreground or the background, first re-scores a receipt scored before 0.2.7 under such a rule when its outcome differs, and adds a note to the run; until then `loopmath report` says how many such receipts there are. A receipt the fit cannot read, re-score or write, a receipts folder it cannot list, or a run file it cannot update is named in a note on stderr, and the next fit does what is left. Each receipt gets one note, and a fit with nothing left writes nothing.
- A store that imports a share with `prior import-shared` now counts each shared run's outcome toward the chance of success, with the q the share carries. When the sender's organization setting changed between two shares, only the copy shared last counts.
- Setting `outcome.q.<tier>` now changes only what depends on the chance of success: the chance, the cost per accepted result and the choice. Predicted costs and token counts are the same as unset. A fit with q set records the q values it used in its meta.json when they differ from the built-in ones.
- `loopmath runs` and the runs page keep the runs `onboard` brought in apart from spend: the total leaves them out, and a separate line gives their cost and count.
- The runs page's opening line gives the dates of the first and last run in local time, as the rows do.
- `loopmath config unset KEY` returns a key to its default. `config set` and `config unset` keep the file's other lines, comments included, also beside a table header with a comment. `fit` says when config.toml cannot be read, and fits with the built-in values.
- Log-axis tick labels, the builder page and the graph legend print money as the terminal does, with an ASCII minus. The builder's strip labels no longer overlap. A recording note whose two times fall in the same minute prints seconds.
- The graph page footer and the posterior page's dropped rows put a plural noun after any count but 1.
- `skill uninstall --scope project` removes the folders it empties.
- `skill doctor` counts every model a session ran among the models it saw, not only the first.
- Adapters write their extension under `dev.loopmath.adapter.<name>`. Documents written by earlier versions still migrate and validate, with a warning.
- The README, the record skill and `run record --session` help say how a sub-agent or Codex child thread, a Codex counter reset and a session that cannot be cut to the window count.

### Upgrading from 0.2.6

There is nothing to run by hand. The first fit after the upgrade, `loopmath fit` or a fit in the background:

- re-scores each receipt 0.2.6 scored under a rule it read otherwise, when the outcome differs, and adds the note "receipt ID re-scored: the run's rule is read as the fit reads it" to its run. The fit prints how many on one line (`receipts_rescored` in `--json`). A store with no such rule reads no receipt and changes nothing;
- judges the runs `onboard` brought in by their latest review, when they have a review or accepted verdict and no tests verdict;
- counts the outcomes of the shares the store imported before, with the q each carries;
- with `outcome.q` set, predicts the same costs and token counts as with it unset.

Runs stored by `run import` before 0.2.7 stay under their rule until you move them: `loopmath runs` lists those with a review verdict, no tests verdict and no known outcome.

## 0.2.6 (2026-10-06)

Codex sessions cut to the recorded window, one money format, a tidier graph page, and `outcome.q` from config in the fit. Predictions for stores with no Codex attempt recorded with a window and no `outcome.q` set are unchanged.

- `run record --session` with a Codex session counts only the usage the session logged inside the window (`--run`, `--since`, `--until`): each token reading's increase counts at the reading's time, and a reading on either edge is included. A resumed thread counts each of its files this way. A thread the session started inside the window counts in full, as a Claude Code sub-agent does. When a reading has no time, or a usage counter goes down inside the window without a reset, the session counts whole and the note says why. Attempts recorded without a window, and stores written by earlier versions, count as before.
- A named session that starts after `--until`, Codex or Claude Code, counts nothing, and the note says so.
- Money reads the same everywhere: `$0`, one significant digit under a cent (`$0.001`), otherwise two decimals with thousands separators (`$0.05`, `$1,234.40`), and `-$2.50` for a negative amount. The `runs` table, the run, estimate, plan and graph pages, and `run record` now print the same string for the same amount, rounding an exact half to even as the terminal does (`$0.125` is `$0.12` on the pages too).
- The graph page's cost axis ends at or above the total, with ticks to match the amounts. The force panel is sized by the number of attempts, and the footer names the page's own attempts.
- `loopmath config set outcome.q.<tier>` now reaches the fit, the `runs` view and `share`, as it already reached `run finish`. Unset, every number is as before. When config.toml cannot be parsed or read, `runs` and `share` keep the built-in q and say so.

### Known issues

- Setting `outcome.q` also changes the predicted costs and token counts, not only the chance of success. The cost and token model is the same; the fit draws new random samples for it, so its numbers move. Fixed in 0.2.7.
- Runs imported with `prior import-shared` do not count toward the chance of success in the store that imports them, and that store does not use the q a share file carries. This has been so since shares were added; it is not new in 0.2.6. Fixed in 0.2.7.

## 0.2.5 (2026-09-28)

Fixes background fits, learning from imported runs and analysis explanations. Predictions for stores without adapter imports are unchanged.

- Background fits stay in the requested store when `--home` or `LOOPMATH_HOME` is relative. This covers imported and finished runs, recorded sessions, late outcomes and explicit background fits, including their inherited exclusions.
- Imported Pi, OpenCode and graph attempts now contribute to the fit through their saved workflow membership. Existing migrated adapter runs can be used by the next `loopmath fit` without rewriting their history. Checks without model usage remain checks.
- Migrating a graph run export preserves its measured dollars, token counts and full model settings. To repair graph runs imported with 0.2.4, migrate the original graph export again with `loopmath ocp migrate`, re-import the migrated document with `loopmath run import --finish`, then run `loopmath fit` explicitly. Re-importing an already-finished run can leave an existing fit unchanged.
- When analysis cannot compare configurations, it names the largest exclusion and its count, separates missing outcome evidence from comparison exclusions, and suggests what would make a comparison possible. Terminal, JSON and HTML explanations agree, including a custom `--min-n`.
- Graph costs that round up to one cent display as `$0.01`. HTML reports omit support notes for spread bands that are not shown.
- Pi and OpenCode exports use current extension names; older exports remain readable.

## 0.2.4 (2026-09-28)

Fixes from a new-user test of 0.2.3. The prior and the belief model are 0.2.3's: the same runs and task give the same cost, chance and score. The results page now predicts for the same task as `recommend`, so its ranges match.

### Adapter output imports

- `run import` accepts the run ids loopmath's own adapters and `graph --format run` write (`pi-adapt:...`, `omp:...`, `loopmath-graph:WORKSPACE`), so `adapt`, `ocp migrate` and `run import --finish` work end to end, and importing a run again replaces it instead of adding a copy. The run document keeps its id; the store names the file with each character outside `A-Z`, `a-z`, `0-9`, `.`, `_` and `-` written as `%XX` (`a:b` is stored as `a%3Ab.ocp.json`). Every id that was valid before keeps its file name. `run import --finish` also takes documents whose producer declares no events or outcome evidence, and records those declarations in the run's `dev.loopmath.import` ext.
- `run import FOLDER` says how many runs were new and how many were already in the store (`--json` adds `new` and `already_in_store`). Importing finished runs into a store that has runs and no fit starts a fit; when no fit starts, it says why.
- `adapt NAME` that finds no sessions ends with an error and writes nothing (`adapt omp` wrote an empty document), and a store an adapter cannot read gives one error line instead of a traceback. The opencode adapter's producer is `loopmath/adapter-opencode`.

### Run times

- Commands show times in your local time with the zone, for example `2026-01-15 09:30 PST`; the store and `--json` keep UTC (`...Z`). `--since` and `--until` without an offset are local time. Times already in a store are read as written and sorted by instant.
- The `runs` table, `runs RUN` and every time on the pages (run details, signals, the run page timeline, the results page's fit and page times, footers) show the same local time with the zone.

### Spend in status

- `status` shows the month's spend, then its parts on one line (your runs, and what onboard's labelling calls cost), then the runs onboard brought in from your logs on their own line, not counted. Labelling spend counts toward the budget cap.
- A run you record with `--source habit` is your run: it counts in spend and in `report`. Only runs `onboard` brought in are history.

### `run record`: commits and session windows

- `run record` looks for the run's commits after its base commit in the folder it runs from, the run's repo, and each session's folder and the repo's folder inside it. When it finds none, it names the folders it searched and says to pass `--commit SHA` (or `--no-commits`).
- With `--run`, a Claude Code session that started before the run counts from the run's start; a Codex session cannot be cut and counts whole, with a note. `--since` and the new `--until` set the window, and `--since` before `run start` moves the run's start back. An attempt no longer starts before its run (OCP `W150`).
- Outside a Claude Code tool shell, `--session self` uses the one Claude Code session in the current folder that was active in the last 10 minutes. The new `loopmath run sessions` lists the recent Claude Code and Codex sessions in a folder with their ids, for `--session ID`.
- `run record` prints every note (it printed only the first), including when a session's log ran another model or effort than the configuration, and when another run of the same slate counts the same session over an overlapping window. `runs --slate SLT` says which run the referee preferred (or that none is recorded yet), as do its rows and `runs RUN` for a slate member.

### Shares

- `prior import-shared` refuses a file this store shared itself: its runs are already yours. The new `prior remove-shared ORG` takes an imported organization out again (then run `loopmath fit`).
- Background refits (after `run finish` or a late `outcome`) keep the sources the last `fit --without` or `fit --no-prior` left out, and `status` names them.
- `share --out FILE.json` writes plain JSON; any other name writes gzip, and `prior import-shared` reads both. `prior build` needs `--out DIR` and never writes into the installed package.

### The graph and page layout

- `graph --out` and `adapt --out` write to any path you name, inside or outside a git repository, and create missing folders; a directory is refused. `graph` prints a short summary and says how many more counts `--verbose` shows.
- The graph page: each model has its own color (the page knew three older model names, so every other model was grey); boxes and time labels stay inside the plot; the tokens cell adds input, cache read, cache write and output, and shows a lower bound (`≥6k`) when one of them is missing; an attempt with no role is named after the tool that ran it and shown as "role not known"; what the page could not show sits behind a collapsed line.
- The results page (`posterior --html`) is titled "What the fit predicts for T tasks in R", and each section says what its estimate rests on: how many rows from your runs, or the shipped prior. The headline labels the median and the mean run, the spend section reads as sentences, each level is labelled once at its own point, and the panels' axes fit the data.
- The recommend page counts its options from the data and names what the pair runs, and its workflow strip fits its box. The run page draws a one-attempt run as a labelled bar in its piece's lane, and the end tick keeps its label. The builder's cost axis reaches builds under a cent, a review gate shows its round limit ("reviewed once ... 1 round, so no repair") instead of a warning, and the model cards fill their rows.

### Small costs

- A cost of exactly zero prints `$0`. A cost above zero and under one cent prints with one significant digit (`$0.002`, and `$0.001` for 0.00096), never `$0.00`. Larger costs keep the precision they had. This holds in `status`, `budget`, `report`, `run record` and `run finish`. JSON keeps the numbers.
- The same rule holds in the `recommend` text and on every page; a change in cost prints `±$0` only when it is exactly zero, and nothing prints `under $0.01`.

### Recommend text

- Your usual workflow comes only from your habit runs (onboarded, or started with `--source habit`): a run of a recommended workflow never becomes the usual, and a tie keeps the workflow that reached the count first. A line under the reference says where it comes from and how to name another.
- Each alternative gives its own chance, run cost and cost per accepted result, then what it changes from the pick, and the exploration lines give the same payback as the summary. A long answer keeps its summary and the rec id, and says how many more alternatives `recommend --json` has.
- `posterior --workflow CFG` without `--type` and `--repo` uses the task type and repo CFG ran on most, and names the others. `recommend` records the task it asked the fit about, and the results page predicts for that task.

### CLI and README

- `verify-receipts` with no argument checks the receipts in your store (`--home`); give a file to check a research receipt ledger, as before. `run artifact` and `outcome` print a one-line summary after the id.
- `fit --full` compiles pytensor's C code in the store's cache instead of `~/.pytensor` (a compile folder set in `PYTENSOR_FLAGS` is kept), works around a macOS clang that reads pytensor's `-ld64` as a library name, and when the compiler still fails ends with one error line and writes no fit.
- `analyze` gives no band for a spread it cannot compute, and says why no configuration qualified and what to try. `analyze --json` prints one object (schema `loopmath.analyze/1`), and `analyze --home PATH` keeps its parse cache in that store. The research verbs move under `loopmath research --help`, and end with one error line on a missing or empty sweep folder.
- `skill install --dry-run` lists every file it would write and writes nothing, and `skill uninstall` says what it removed. `config get` lists every key loopmath reads, set or not, and `config set` refuses a key loopmath does not read. `prices` shows each source without build notes and marks the rates that are not confirmed. The onboard dry run and `doctor` say they keep the log parse cache, and `onboard` records what its labelling calls cost. Sessions in a folder outside git count under `(no repo)`. `workflows list` names the composed workflows onboard may use, and `workflows show` takes them.
- The README says `builder` serves its page on 127.0.0.1 only and no other command opens a network connection, and its commands match `--help`.

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
