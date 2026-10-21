# Does my plugin pay? Reading `claude plugin eval` results

`claude plugin eval` runs a plugin's eval suite: each case runs three times with the plugin and, by default, three times without it, every run is graded, and the result gives each case a score and a delta ([Claude Code docs](https://code.claude.com/docs/en/plugin-evals)). `adapt plugin-eval` reads that result so that loopmath can answer the question the delta leaves open: adopt the plugin, keep the baseline, or run more, with a range on the gain and, when you give each run's cost, on cost per accepted result.

The importer only reads the file. It never runs Claude Code and makes no model calls. Experimental: it is tested on results written from the docs, not yet on real output (see [The layout is provisional](#the-layout-is-provisional)).

## Four commands

```
claude plugin eval . --json result.json
loopmath adapt plugin-eval result.json --out ocp --rows rows.jsonl
loopmath run import ocp --finish
loopmath check-change from-runs --base-label arm:without --cand-label arm:with
loopmath gate --base arm:without --cand arm:with --runs rows.jsonl --rule accept --r-max 24
```

The first is Claude Code's own. The input may also be the `results/<timestamp>/` folder that plugin eval writes, which holds `aggregate-result.json`. `adapt plugin-eval` writes one OCP run per eval run into `ocp/` and, with `--rows`, the same runs as `loopmath.harness-runs/0.1` rows. Its summary ends with the next commands filled in for your suite: `check-change from-runs` gives the verdict (`adopt`, `reject` or `more`) with 80% and 95% ranges, and `gate` gives the action and how many more pairs to run. `--json` prints the summary as one JSON object (`loopmath.adapt.runs/1`), with the result's own numbers under `plugin_eval`.

Exit codes: 0 written; 1 refused, with nothing written; 2 for a usage error or a missing file.

## What becomes a run

- **One run per eval run.** The case name is the task, `arm` is `with` or `without`, and the pair is `<case>#<j>`, the j-th run of each arm. Replicates stay separate runs.
- **The grade** is the run's score: a number from 0 to 1, or the weighted fraction of its graders that passed, leaving out graders with `scored: false` (in a two-arm run plugin eval leaves out a grader that fails in both arms).
- **Accepted** when the grade reaches the threshold: 1.0 by default, as plugin eval's `--threshold`. If the suite ran with `--threshold 0.8`, pass `--accept-at 0.8`; the importer notes when `casesPassed` suggests another threshold. `--outcome score` keeps the grade as the score `plugin-eval` instead; then give `--rule score:plugin-eval` to `check-change from-runs` and `gate` (the summary's commands do).
- **Skipped paid graders.** A run with `skippedPaidGraders` is kept, unscored: the cost ceiling skipped its judge graders, so its grade is not comparable.
- **Aborted.** A run a mock stopped is graded 0, as plugin eval scores it.
- **Errors.** A run with an `error` keeps the grade the file gives it. Since Claude Code 2.1.299 a run that ends on an API error, such as a rate limit, scores 0 with no grader results, so the error counts against the arm it hit: re-run those cases. A run with no score and no grader list is refused; the importer never supplies a grade.
- **Partial suites** (`partial: true`, for example at the `--max-cost-usd` ceiling) import the runs they recorded, with a note.
- **Labels.** The result does not record the model, so `--model M` labels the runs. The plugin's name and version come from the `plugin.json` above the results folder; `--plugin NAME` names it when there is none (for a plugin you evaluate by name, plugin eval writes results under your current folder, so pass `--plugin` there).
- **Kept fields.** The suite's numbers, each case's score and delta, and each run's grade, abort and whether it ended with an error are kept in the run's ext (`docs/adapters.md` lists every member). Fields the importer does not know are kept too, under `com.anthropic.claude-code.plugin-eval`. Free text (prompts, grader names and explanations, error messages) is dropped unless you pass `--keep-text`, and the summary lists what was dropped.

## Which cases are compared

The docs omit a case's delta "when the case ran one arm or the arms aren't comparable". So only a case with both arms and a delta is compared. Any other case of a two-arm suite is withheld: no run of it is written, and the summary's `withheld` list gives the case, why it is not comparable, and its run grades. A missing delta is never read as 0. If no case is comparable, the import is refused. A suite run with one arm (`--ablation none`) imports its runs as records, with nothing to compare.

## Cost

plugin eval records one cost for the whole suite, `costUsd`, a list-price estimate that includes judge calls. The importer shows it and keeps it, and it is never split over runs: an even split would hide the difference between the arms that cost per accepted result exists to measure. Without per-run costs the comparison is on acceptance alone, and the summary says the cost is unknown.

`--cost-from FILE` gives each run its cost from your own records, one JSON line per run:

```
{"case": "fix-null-check", "arm": "with", "run": 1, "usd": 0.41}
{"case": "fix-null-check", "arm": "without", "run": 1, "tokens": {"input": 42000, "cache_read": 180000, "cache_write": 6000, "output": 3100}, "model": "claude-sonnet-5-5"}
```

`run` counts from 1, in file order. Tokens without `usd` are priced from the price table (`loopmath prices`). A line that names no run of the result, or a second line for the same run, refuses the import. Only the file you name is read.

Where the numbers can come from, since the result has none per run (none of these is verified on real output):

- **One run per suite.** `claude plugin eval . --case NAME --runs 1 --ablation none --json r.json` runs one `with` run, and its `costUsd` is that run's list price with its judge calls. The docs give no way to run the `without` arm alone, and each such run is spent again.
- **The kept sandbox.** `--keep-temp` keeps every run's sandbox directory and prints its path. If the run's session transcript is there, the usage it records gives the run's tokens, which the price table prices. The docs do not say what the directory holds.
- **Not OpenTelemetry, as documented.** A run inherits only an allowlist of your shell's environment (`PATH`, locale, proxy, provider and most `ANTHROPIC_*` and `CLAUDE_CODE_*` settings, and `EVAL_*`), and the allowlist names no `OTEL_*` variable.

```
loopmath adapt plugin-eval result.json --out ocp --cost-from costs.jsonl --model claude-sonnet-5-5
```

## How far the answer reaches

The answer is about these cases. Runs of one case share its task, so they are not independent tasks: more runs narrow the range for these cases, and to learn about tasks like them, add cases, not runs. With grades and few cases this binds hard: in the simulation below, with 6 cases a plugin that helps was adopted in 2% to 12% of suites, against 9% to 44% with 30 cases at one run each. When a case has more than one pair, the import's summary says so.

- **Grades** (`--outcome score`) **with more than one run per arm in a case:** `check-change from-runs` compares cases, not runs. Each case is one pair, with each arm's mean grade over its graded runs and its mean cost per run, so the answer is about new cases like these. It prints `unit: case` and why, and `--json` gives the same under `unit`. `--unit run` pairs the runs instead. The cap is chosen for the cases.
- **Acceptance rows** stay paired run by run, and `--unit case` is refused for them. In the simulation below, they stayed under the 5% false adopt bound that from-runs states, even when the plugin's effect differed by case.
- **`gate`** pairs runs. Under `--outcome score` with several runs per case, the import's summary says so under the gate command.

How much this matters was measured in a simulation, not on real suites. Each case's plugin effect was drawn around zero (normal on the logit scale, standard deviation 2), so the plugin helps some cases, hurts others, and on average helps none. With 30 runs per arm, one `check-change from-runs` look adopted the plugin:

- in 0.1% to 0.3% of suites with 30 cases at one run each;
- with 10 cases at 3 runs, in 0.7% to 1.2% read as acceptance, and read as grades (`--outcome score`), in 3.2% paired run by run and in 0.8% paired case by case;
- with 6 cases at 5 runs, in 1.8% to 2.9% read as acceptance (with or without per-run costs), and read as grades, in 7.2% paired run by run and in 0.7% paired case by case.

Paired case by case is what from-runs now does for grades with replicates.

The price is fewer adopts when the plugin does help. With each case's effect drawn around a mean that helps (standard deviation 1; mean 0.5 or 1.0 on the logit scale, a mean gain of 0.09 or 0.18 in a run's grade) and 30 runs per arm, one look adopted the plugin, read as grades:

- with 30 cases at one run each, in 9% or 44% of suites;
- with 10 cases at 3 runs, in 5% or 26%;
- with 6 cases at 5 runs, in 2% or 12%: with 6 cases, a modest gain is almost never adopted.

Nearly every other look said `more`, not `reject`. Paired run by run, the same suites adopted more often (16% or 47% with 6 cases), but that pairing also adopted a plugin with no gain on average in up to 7.2% of suites, as above.

With the same effect on every case, no design adopted in 1% or more. These rates hold for that simulated setting only; they are not a bound for your suite.

`check-change from-runs` sets its cap from the pairs recorded: the smallest calibrated cap above them (6, 12, 24 or 48), so a check that has not decided says `more` rather than keeping the base at the cap. It prints the cap and why, and `--max-pairs` sets another. Suites over 48 pairs get the strictest calibrated threshold and no stated false adopt bound. `gate` has its own cap, `--r-max`; the summary's gate command gives it the same cap.

Each `check-change from-runs` is one look at the runs. Running it again as pairs accrue, with the cap chosen again each time, is a new look, and across cap bands the false adopt bound no longer holds as stated. For a check you will look at again, fix `--max-pairs` once, or use `check-change start`.

## The layout is provisional

The docs describe the result's fields but give no JSON example, and they name neither a run's score member nor its grader list. The importer reads a run's `score` (a number from 0 to 1) or its one grader list (a list of objects with a boolean `passed`, an optional boolean `scored`, and a `weight` at or above 0, 1 by default). It refuses a run with two grader lists, a score its graders contradict, or a member of the wrong type. It checks every case against the file's own numbers: the mean `with` grade must be the case's score, and with minus without its delta. A case that fails the check is refused by name, with the `claudeVersion`. These checks catch a misreading of the means, not of every run: runs graded 0 and 1 have the same mean as two runs graded 0.5. Real plugin eval output is unverified; the tests use results written from the docs. The importer knows `schemaVersion` 1 and refuses any other version by name.
