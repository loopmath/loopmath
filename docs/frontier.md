# Finding a cost and pass-rate frontier (experimental)

`loopmath frontier` looks for the settings (model and effort) that give the best pass rate for their cost on your
own tasks. It works by ask and tell: `next` says which setting to run on which task, `record` takes each result
back, and `report` shows each setting's pass rate, cost per run and P, the model's probability that it is on the
frontier. It is experimental: P is not yet checked for calibration, so do not stop on it.

```sh
export LOOPMATH_HOME=$(mktemp -d)
loopmath frontier init --tasks tasks.jsonl --settings settings.toml --budget 50
loopmath frontier next --json
loopmath frontier record TRL --passed PASSED --usd USD
loopmath frontier report
```

`tasks.jsonl` holds one task per line (`{"id": "t1"}`, with an optional `"weight"`); `settings.toml` holds
`[[settings]]` blocks with `name`, `model` and `effort`. TRL is a trial id that `next` gave, PASSED is 1 for a pass
and 0 for a fail, and USD is what the run cost in dollars.

## Strategies

A frontier keeps the strategy it was made with. `next --strategy NAME` refuses a frontier made with another one.

- **`adaptive-blocks`** is the default for new frontiers. `next` hands out a block: one run, or 8 or 64 runs of one
  setting.
- **`kg+random`** was the only strategy up to 0.4.5. Frontiers made with it resume on the same code and give the same
  trials and the same report as before. Their report adds `strategy` and one line saying that a newer strategy
  exists. `init --strategy kg+random` still makes one.

## How `adaptive-blocks` chooses runs

It buys the run or block of runs that adds the most value per planned dollar.

- **Value.** A user with a budget per run picks the setting with the best estimated pass rate among those whose
  estimated cost per run fits the budget. The value is that pick's pass rate, averaged over a grid of budgets per run
  (36 log-spaced budgets from USD 0.0145 to 26.5 by default; `init --run-budgets LO:HI[:J]`). A candidate's gain is
  the expected rise of the value once its outcomes are in.
- **Candidates.** Every open pair (setting, task) as a single run, and per setting a block of 8 and a block of 64
  runs. A block takes the open tasks that lower the setting's uncertainty most per planned dollar, one run each.
  Blocks let it spend a large budget on a setting that matters without deciding run by run.
- **Pair cap.** A pair takes at most 4 runs (`init --pair-cap N`). Copies of one pair add little about a setting's
  pass rate across your tasks.
- **Optimistic price.** Whether a setting can serve a budget is judged by the 20th percentile of the belief about its
  mean cost per run, so a setting is not priced out before it has been run enough. A run's planned cost is the
  median of that belief.
- **No penalty.** The value that chooses runs has no term for the uncertainty of the pick.
- **Choice.** The largest gain per planned dollar among candidates with a positive gain; if none is positive, the
  largest gain.
- **Probes.** A single run keeps its chosen task 90% of the time. Otherwise its task is drawn at equal chance from
  the setting's open tasks. These probes feed a standing misfit test of the success model. A setting the test flags
  gets its tasks by the balanced rule below.

`init --task-rule balanced` swaps how tasks are chosen and keeps everything else. A candidate is a block of 1, 8 or
64 runs of one setting on its least-run open tasks, or one run on each of its open tasks. Each is scored on that
allocation, and ties among least-run tasks are drawn at random. Every run is then a probe.

The success model is a two-parameter logistic model per task, with learned effort steps and a Laplace
approximation over setting abilities and task difficulties. The cost model is a lognormal model per setting, fitted
apart from it. Both are refitted in full at the end of a block once 20 or more runs have come in since the last
full refit.

## Blocks, records and resumes

- `next` returns every trial of the open block. While any are pending it returns those same trials again, so a
  restart loses nothing. `--k` is for `kg+random` only.
- `record` takes the runs of a block one at a time and in any order. The next block is the same whatever the order
  and whatever reports came in between, because the models take in a block's runs in the order they were issued.
  `report` reads the recorded runs of a partly recorded block too.
- `next` cuts a block before the first run whose expected cost would take the spend over the budget.
- Each trial records `block`, `pos` (its place in the block), `explore` (a probe), `support` (the tasks it was drawn
  from) and `p_pred` (the model's pass chance for the pair when the block was issued). The frontier records the
  strategy's settings in `config.policy` and `config.run_budgets`, and the state of its random streams in `rng`.
- The snapshot that speeds up a resume is keyed by the package version and the model version. A snapshot written by
  other code is ignored, and the state is rebuilt from the recorded runs.

## The report

`report` keeps every line and key of 0.4.5. For `adaptive-blocks` it adds:

- `budgets`: for each budget per run, the pick (from the frontier of the estimates), its estimated pass rate and
  mean cost per run, its runs, and `p_fit`, the model's chance that its mean cost per run is at most the budget. The
  text groups neighbouring budgets with the same pick.
- `misfit`: the misfit test's state. `flag` is on while a setting's results stray from the model's predictions more
  than chance allows, at a 5% false-alarm rate over the whole frontier.

The report reads the models after a full refit of a copy of them. It never changes what `next` hands out.
