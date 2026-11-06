# Run a flow experiment with loopmath

A flow is a fixed multi-step workflow: its steps and the model each one runs, the checks after them, the judges
that score a run, the inputs, and the edits worth trying. loopmath plans which arms to run on which inputs, takes
the finished runs, and says which change to adopt. It runs no model and spends nothing: you run each arm with your
own tools, and loopmath reads only what you record.

Use exactly the commands below, and take their JSON as printed. Their flags, JSON members and exit codes are in the
last section of this guide, "The flow commands". Never run `--help`.

`FLOW` is the flow folder (or its `flow.ocp.json` or `flow.toml`). Every command takes `--home PATH` for the store,
as the other loopmath commands do.

## 1. Set up the flow

Ask the user what the flow does and fill in the eight answers below, then:

```sh
loopmath flow init DIR --answers FILE --json
```

The answers are one JSON object with exactly these eight keys, in any order. `hypothesis` and `allow` take null
when empty; the key itself is still required. Each record takes its own members only (required, then optional):

- `purpose`: `id`, `title`, `example_input`.
- `steps`: in run order, each `id`, `role`, `model`, `harness`, `effort`, `reads`, `writes` (ids, or
  `{"id", "kind"}`); optional `width`, `context_policy`, `options`.
- `checks`: `budget_rounds` (the round budget), `rescue` (null gives none); optional `gates`
  (each `after`, optional `id`, `rule`, `on_fail`) and `checks`. A second gate after one step needs its own `id`.
- `judging`: `judges` (each `id` and `kind`: `model` with a `model` and a `rubric`, `person` with a `rate`, the
  share of runs a person scores; optional `score`, `scale` `fraction` or `linear` with a `range`) and `accept`, the
  acceptance rule (`name`, `definition`, `requires`, the checks it needs, and `score`).
- `objective`: `score`; optional `kind` (default `cheapest_within_margin`), `margin` (default 0.10, the quality a
  change may lose and still be adopted), `reference` (default `base`), `cost_cap_usd`.
- `hypothesis`: `text` and `edits` (each `kind`, optional `piece`, `key`, `value`); null for none. Each edit must
  be one that `allow` permits.
- `allow`: what may change: `pieces` (per step, for example `models`, `efforts`, `widths`, `drop`), `budgets` and
  `merges`; null for nothing.
- `inputs`: `ref` (the inputs file, beside the flow), `type`, `ids`, `holdout` (the ids kept for the checks);
  optional `n`.

The example flow, a question turned into a one-page brief:

```json
{"purpose": {"id": "lit-brief", "title": "Question to a one-page brief", "example_input": "Stub question 1."},
 "steps": [
  {"id": "scout", "role": "worker", "model": "claude-sonnet-5-5", "harness": "api", "effort": "default",
   "reads": ["question"], "writes": ["sources"]},
  {"id": "extract", "role": "worker", "model": "claude-opus-5-5", "harness": "api", "effort": "high",
   "reads": ["sources"], "writes": ["claims"]},
  {"id": "synthesize", "role": "worker", "model": "gpt-6-astra", "harness": "api", "effort": "high",
   "reads": ["claims", "question"], "writes": ["brief"]},
  {"id": "critic", "role": "reviewer", "model": "claude-haiku-5-5", "harness": "api", "effort": "default",
   "reads": ["brief", "claims"], "writes": [{"id": "verdict", "kind": "verdict"}]}],
 "checks": {"gates": [{"after": "extract", "rule": "command:claims-check", "on_fail": "extract"},
                      {"after": "critic", "on_fail": "synthesize"}],
            "budget_rounds": 2, "rescue": "redo_usual"},
 "judging": {
  "judges": [{"id": "j-claude", "kind": "model", "model": {"id": "claude-sonnet-5-5"}, "rubric": "rubric.md#v1"},
             {"id": "j-astra", "kind": "model", "model": {"id": "gpt-6-astra"}, "rubric": "rubric.md#v1"},
             {"id": "person", "kind": "person", "rate": 0.2}],
  "accept": {"name": "quality>=0.7+claims-check",
             "definition": "quality at least 0.7 and the claims check passes", "requires": ["claims-check"],
             "score": {"name": "quality", "target": 0.7, "better": "higher", "scale": "fraction"}}},
 "objective": {"score": "quality"},
 "hypothesis": {"text": "extract can run on Sonnet without hurting the brief; the critic adds little",
                "edits": [{"kind": "model", "piece": "extract", "value": {"id": "claude-sonnet-5-5"}},
                          {"kind": "drop", "piece": "critic"}]},
 "allow": {
  "pieces": {"extract": {"models": [{"id": "claude-sonnet-5-5"}, {"id": "claude-haiku-5-5"}]},
             "synthesize": {"efforts": ["medium", "high"]},
             "critic": {"drop": true},
             "scout": {"widths": [1, 2], "select": {"harness": "api", "model": {"id": "claude-haiku-5-5"}}}},
  "budgets": [1, 2, 3],
  "merges": [{"id": "extract+synthesize", "replace": ["extract", "synthesize"],
              "into": {"id": "extract_synthesize", "role": "worker"},
              "setting": {"harness": "api", "model": {"id": "claude-opus-5-5"}, "effort": "high",
                          "options": {"prompt": "merged-v1"}}}]},
 "inputs": {"ref": "questions.jsonl", "type": "research",
            "ids": ["q01", "q02", "q03", "q04", "q05", "q06", "q07", "q08", "q09", "q10",
                    "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20"],
            "holdout": ["q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20"]}}
```

It writes `flow.ocp.json`, the inputs file when it is not there yet (one line per id: `id`, `split` and `text`,
empty but for the first train input, which holds the example input; the user fills in the rest; an inputs file
already there is kept), and the first `plan.json`. `--toml` also
writes `flow.toml`, the same flow as a TOML view. The JSON has `schema` (`loopmath.flow-init/1`), `dir`, `written`,
`kept`, `flow`, `digest` and `plan`.

A missing or unknown key is refused by name at any depth, with its path, and nothing is written:
"steps[2].modle: unknown key; steps[2] takes id, role, model, harness, effort, reads, writes, width,
context_policy, options". Fix what it names and run it again. With no `--answers` and no terminal it exits 2 and
names the keys still missing. A margin the planner cannot reach is refused too: one whose check would need more
than 1000000 pairs ("margin M: needs N check pairs at sd_pair S, above the limit of 1000000").

### From runs you already have

When the user already has runs of the workflow, as OCP run documents or as rows, `flow init` takes the steps from
them, and the answers give only the rest. Put `DIR` first, since `--from-ocp` takes every file after it:

```sh
loopmath flow init DIR --from-ocp FILE FILE --json
loopmath flow init DIR --from-ocp FILE FILE --answers FILE --json
```

`--from-rows FILE` gives a rows file, in place of `--from-ocp` or beside it. Every run is read and checked before
anything else: a refused file or row is named, the exit is 1, and nothing is written. Run it first without
`--answers`. It prints what it took from the runs on stderr, then exits 2 naming the answers still missing (in
`error` with `--json`), and writes nothing.

A run declares its configuration when it names a flow (`run.flow`) or its configuration's `source` is not
`habit`. A row logged without a flow converts to a `habit` configuration, which declares nothing.

- When every run declares the same configuration, the flow takes it exactly, with its checks and budget. The
  answers give every key but `steps` and `checks`; giving either is refused ("the runs declare it"). A later run
  of that configuration with other members is named in a `warning:` line on stderr, and the first run's is kept.
- When no run declares one, the steps are inferred: in the order they first appear, each with the role, width and
  setting most runs used, a tie going to the run read first. The answers still give all eight keys, but each step
  is only `{"id", "reads", "writes"}`, one per step of the runs and in their order; a member the runs give is
  refused ("the runs give it"). Checks are never taken from runs; the most rounds the runs saw is proposed for
  `budget_rounds`.
- Also refused by name: a mix, where "some runs declare their configuration and some do not"; runs that declare
  two configurations; a run that declares its workflow by catalog reference; a row that names a flow (record it
  with `loopmath flow record FLOW --rows FILE`); a file with no run; and one run given twice.

What it took, for two logged runs of the example's four steps, one of them with a second round (`RUN` stands for
each run id):

```text
prefilled from 2 runs (how inferred, confidence 0.95): RUN, RUN
  scout: worker, claude-sonnet-5-5 on api, effort default
  extract: worker, claude-opus-5-5 on api, effort high
  synthesize: worker, gpt-6-astra on api, effort high
  critic: reviewer, claude-haiku-5-5 on api, effort default
  the runs saw at most 2 rounds; budget_rounds 2 is proposed
```

Declared runs end with `configuration ID as declared, with its checks and budget` in place of the rounds line.
This prints before the exit 2, and after the `wrote` lines without `--json`; with `--json` a written flow prints
only the init JSON. `confidence` is a heuristic figure of how far the runs agree, not a probability; declared runs
give 1.0. The written flow keeps it in `flow.ext["dev.loopmath.prefill"]`: `from` (the run ids in read order),
`how` (`declared` or `inferred`) and `confidence`. That member is outside the digest, so prefill changes neither
the flow's digest nor its configuration id.

## 2. Plan

```sh
loopmath flow plan FLOW --json
```

The plan holds the experiments, each against the base. With 3 or more factors (the edits to try), the screen runs
first: arms in pairs on the train inputs, each pair one input. Then each merge and the subset (the factors the
screen kept) are checked on the holdout inputs, one arm against the base, up to its pair cap. The margin table
gives the pairs and runs each margin needs, at assumed spreads until 8 scored pairs exist, then at fitted ones. The
price is runs times the mean dollars of the flow's recorded runs (base runs when there are any), unknown until a
recorded run has dollars.

The JSON has `schema` (`loopmath.flow-plan-report/1`), `plan` (the plan as saved: `schema`
(`loopmath.flow-plan/1`), `id`, `flow`, `seed`, `margin`, `arms`, `experiments`, `queued`, `excluded`), `written`,
`replaced`, `margin_table`, `fitted_table`, `price_usd`, `price_unknown` (only while a price is unknown) and
`wait`.

Two prices: each `plan.experiments[].design.price` (`usd`, `per_run`, `from_runs`) is the price when the plan was
made, rounded to four places, and a kept plan keeps it. `price_usd`, per experiment, is the price from the runs
recorded now, unrounded. While one is null, `price_unknown` says why, per experiment; once every price is known,
the report has no `price_unknown`. Quote `price_usd` for what the runs cost now.

`--margin M` and `--budget-usd X` change the plan. A new plan gives new run ids, so once runs of the current plan
are recorded, a new plan is refused and the refusal says what would change. With the user's word, `--replace`
writes it and keeps the old one as `plan.<its id>.json` in the flow folder; nothing is deleted. A margin near the
pair limit makes a large plan: about 115 MB, and about 5 seconds to make, with 2 model judges.

## 3. Take the next runs

```sh
loopmath flow next FLOW --n 2 --seed 0 --json
```

`--n N` is how many runs (default 2, one pair); the same plan state and `--seed S` give the same runs. The JSON
has `schema` (`loopmath.flow-next/1`), `plan`, `runs`, `wait` and `adopted`. Each item of `runs` holds `run_id`,
`input`, `arm`, `configuration_id`, `settings` (per piece: `harness`, `model` and, where set, `effort`,
`context_policy` and `options`),
`experiment`, `split`, `slate`, `judges`, `person`, `flow` (`id`, `version`, `digest`, `arm`, `plan`, `split`),
`record` and `slate_members`.

The two runs of a slate are a pair: one input, two arms. Run both. A run with `person` true needs a person's score
(step 7). When `runs` is empty, `adopted` names the arm a finished plan adopts; otherwise every run given so far
is recorded and the plan waits for their outcomes (`wait`, when set, names the runs). Record them, then ask again.

## 4. Run each arm

For each run: take the input (its `text` in the inputs file) and run the arm's steps on it with the run's settings:
each piece's harness, model and effort. The arm's pieces, in order, with their roles and widths, are in
`plan.json`: the item of `arms` whose `id` is the run's `arm`, under `configuration.workflow.pieces` (`id`,
`role`, `width`). An arm may drop a step, merge two, or widen one and add a select piece that picks one of its
outputs. `configuration.workflow.control` says which steps a check follows, the rule of each check, which step a
failure sends back to, and the rounds allowed. Keep each step's cost.

Then judge: each judge in the run's `judges` scores the output on the flow's score (a model judge: that model with
the judge's rubric). Do not show a judge which arm made the output.

## 5. Write what ran

Write one harness-runs row per run, one JSON object per line, in a rows file. This row is the example flow's
`run_pln_97f47f3321db.screen.001.a`, shown over several lines:

```json
{"schema": "loopmath.harness-runs/0.1", "id": "run_pln_97f47f3321db.screen.001.a", "task_id": "q01",
 "arm": "set-1-3", "status": "done",
 "flow": {"id": "lit-brief", "version": 1, "digest": "flw_774ab7a14c50", "arm": "set-1-3",
          "plan": "pln_97f47f3321db", "split": "train"},
 "slate": {"id": "slt_pln_97f47f3321db.screen.001",
           "members": ["run_pln_97f47f3321db.screen.001.a", "run_pln_97f47f3321db.screen.001.b"]},
 "stages": [
  {"role": "worker", "piece": "scout", "model": "claude-sonnet-5-5", "cost": {"usd": 0.31}},
  {"role": "worker", "piece": "extract", "model": "claude-sonnet-5-5", "effort": "high", "cost": {"usd": 0.42},
   "checks": [{"id": "extract", "verdict": "pass"}]},
  {"role": "worker", "piece": "synthesize", "model": "gpt-6-astra", "effort": "medium", "cost": {"usd": 0.38}},
  {"role": "reviewer", "piece": "critic", "model": "claude-haiku-5-5", "cost": {"usd": 0.04},
   "checks": [{"id": "critic", "verdict": "pass"}]}],
 "outcome": {"judges": [{"id": "j-claude", "score": 0.78, "observed_at": "2026-11-05T10:12:00-08:00"},
                        {"id": "j-astra", "score": 0.74, "observed_at": "2026-11-05T10:13:00-08:00"}]},
 "started_at": "2026-11-05T10:00:00-08:00", "ended_at": "2026-11-05T10:11:00-08:00"}
```

- `id` is the run's `run_id`, kept as its run id; `task_id` is its `input`; `arm` and `flow` are the run's, as
  given; `slate` is `{"id": slate, "members": slate_members}`.
- `status` is `done` or `failed`.
- `stages`: one per session, in the order they ran, each with its piece's `role` and `piece` from `plan.json`, its
  `model` and `effort`, and its `cost` (`{"usd": X}`, or tokens priced with its model). A step sent back runs
  again as `"round": 2`, and so does the check after it. A piece of width 2 is two stages in one round.
- A stage after which a check ran gives `checks`: `{"id", "verdict"}` items, `pass` or `fail`. The id is the
  check's id in the arm's `control`, or, for a gate, the step it follows. A stage that reports no check says
  nothing about it.
- `outcome.judges`: one `{"id", "score", "observed_at"}` per judge that scored the run.
- `started_at` and `ended_at`, times with an offset. A run states its own end: one with none is refused.

An OCP 0.4 run document works too: `run.flow` as the run's `flow`, `run.task.id` its input, `run.slate`, the
configuration whose `id` is its `configuration_id`, and `run.ended_at` (or each attempt's `ended_at`).

## 6. Record

```sh
loopmath flow record FLOW --rows FILE --json
```

`--ocp FILE ...` records run documents instead. The JSON has `schema` (`loopmath.flow-record/1`), `flow`,
`digest`, `recorded` (each `run`, `arm`, `input`, `import`, `ended_at_from`), `findings`, `advisories`, `fit` and
`plan` (`id`, `advanced`, `skipped`, `problem`). Every item is checked first: one refused item records nothing, and
the JSON then has `refused` (each `item`, `run`, `reason`). Tell the user each finding and each refusal as it is.
The refusals name what is wrong, for example:

- "states no end (ended_at, or its last stage's ended_at); flow record stores a run finished at its own end";
- "this row names flow 'lit-brief' version 2, but the flow given is flow 'lit-brief' version 1";
- "task.id 'q99' is not among the flow's inputs.ids";
- "stages.0.piece: 'nope' is not a piece of flow 'lit-brief' arm 'set-1-3' (pieces: scout, extract, synthesize,
  critic)";
- "flow record does not replace a recorded run", for a run already recorded with other content.

A run document is refused the same way, and also when its `run.flow.digest` is not the flow's ("the run was made
for another revision of the flow") or its acceptance rule is not the flow's.

A run the plan did not ask for needs `--allow-unplanned`. A simulated run (one whose `run.ext` holds
`dev.loopmath.simulate`) needs `--allow-simulated`, on every import, even of a run already stored; never give it
for runs that really ran. Recording the same file again is safe: each run is "already recorded", `import`
`unchanged`, and no refit starts.

After the store write, `flow record` brings `plan.json` up to date (`plan.advanced`). If that fails, the runs stay
recorded, it exits 1 and `plan.problem` says why; `loopmath flow next FLOW` brings the plan up to date once the
cause is fixed.

## 7. Scores after the run

A run with `person` true: show the user the input and the output, without the arm, and ask for a score on the
flow's scale. Then:

```sh
loopmath flow score FLOW --run RUN --judge ID VALUE --json
```

`ID` is the flow's person judge (`person` in the example); `flow next` prints this line for each such run.

- `VALUE` is a number: in [0, 1] for a `fraction` judge, within the range of a `linear` judge that has one (a
  bound itself is in range). A judge that is not the flow's is refused, naming the judges.
- `--observed-at TS` is when the score was given, ISO 8601 with an offset (default now). It is stored in UTC; a
  time with no offset is refused.
- A second score of one run by one judge is refused: "--replace adds this one as the newest reading". With
  `--replace` the new score is added and the earlier ones are kept; its time must be later than the newest
  reading. Per run and judge, the reading is the newest by `observed_at`.

The JSON has `schema` (`loopmath.flow-score/1`), `run`, `judge`, `signal`, `value`, `replaced`, `state`, `fit` and
`plan`. A recorded run is finished, so its score is `late`: the run's receipts are re-scored and one refit starts
in the background. A receipt is written at the run's finish with the judges fit of then; a late score is how it
catches up. Like `flow record`, `flow score` brings `plan.json` up to date after the write.

## 8. Read the state and decide

```sh
loopmath flow status FLOW --json
loopmath flow check FLOW --arm ARM --json
```

`flow status` has `schema` (`loopmath.flow-status/1`), `flow`, `plan`, `plan_problem`, `runs`, `other_revisions`
(runs of other revisions of the flow, counted apart), `cost` (per step and model: `step`, `model`, `attempts`,
`priced`, `usd_known`, `usd`, `reason`), `checks` (per check and arm: `check`, `arm`, `runs`, `pass`, `fail`,
`other`, `missing`, `rate`), `quality` (per arm: `arm`, `runs`, `scored`, `pooled_mean`), `judges` (`calibrated`,
`n_person`, `judges`, `s_run`, `refused`, `anchor`), `experiments` (each `id`, `kind`, `arms`, `state`, `verdict`,
`pairs`, `pairs_left`, `effects`, `next_pair_value`), `wait`, `adopted`, `dollars` (`runs`, `priced_runs`,
`attempts`, `priced`, `usd_known`, `usd`, `reason`) and `notes`.

`flow check` judges one change against the base: `schema` (`loopmath.flow-check/1`), `arm`, `reference`, `pairs`,
`p_quality`, `p_cost`, `threshold`, `margin`, `pair_cap`, `verdict`, `more_pairs` and `extra` (the experiment, its
rule, the reasons). `verdict` is `adopt`, `reject` or `more` (`more_pairs` more pairs are needed), and every
verdict exits 0. Exit 1 means the check cannot be made, for example for an arm that runs only in the screen, for
the base, or with a plan made for another revision of the flow. So branch on `verdict`, never on the exit code.
`extra["excluded"]` lists each pair left out because a run is not the one its pair planned (`pair`, `run`,
`reason`). Only when there are any, `extra` also has `nonfinite` (`cost:<run id>`, a run whose dollars no float
holds, read as no cost), `cost_range` (compare's reason when it states no cost range, so the dollars are read as
missing) and `unreadable` (`signals:<run id>`, a run read with no signals).
Screen arms have no check of their own: `flow status` gives the screen's state, and its effects once read.

How to read the numbers:

- `quality` in `flow status` is pooled: the judges fit reads each run under its arm, its input and the run noise,
  so its quality is pulled toward its arm and input level. Every other reader of a run (its outcome, the checks)
  uses its unpooled estimate, from its own judge scores with their biases taken out. Neither is always the wider:
  the unpooled spread can sit below the pooled one.
- Under the score name of a run's own acceptance rule, a run's quality is the judges fit's value, never a raw
  judge score. A judge that writes under its own `score` name keeps those readings raw.
- The judges fit takes at most 13 judges with readings per score, the person included; declared judges with no
  reading do not count. Above that the fit is refused, those runs read no score (`too-many-judges:<n>`), and
  `flow status` and `flow check` print a warning on stderr.
- `notes` are lines on what was not read, each ending in its reason: `unranged:<judge>` (a linear judge's score
  has no range), `outside-range:<judge>` (the judge's newest score is outside its range, [0, 1] for a judge with
  no scale), `nonfinite:<judge>` and `nonnumeric:<judge>` (a score of that judge that is not finite, or not a
  number, was seen; the judge is still read from its other numbers, and a count is of scores skipped),
  `unreadable:signals` (a run whose `run.signals` is not a list) and `cost:<run id>` (a run whose dollars are past
  a float). Judge scores that name no judge share one line naming each `unattributed:<signal id>`. Record judge
  scores with `flow score`, which names the judge.
- Until 6 runs carry a person's score the judges are uncalibrated: their biases are relative to the judges' mean.
  From then on the person anchors the scale.

When `verdict` is `adopt`, or `adopted` names an arm, tell the user. Changing the flow to that arm is the user's
call.

## 9. Change the flow

```sh
loopmath flow show FLOW --json
loopmath flow export FLOW --json
loopmath flow import FILE DIR --json
```

`flow show` prints the checker's `findings` first, then the flow: `schema` (`loopmath.flow-show/1`), `path`,
`findings`, `id`, `version`, `title`, `digest`, `steps`, `checks`, `budget`, `rescue`, `judges`,
`acceptance_rule`, `objective`, `hypothesis`, `allow`, `inputs`, `plan` and `unread`. A folder with both
`flow.ocp.json` and `flow.toml` reads `flow.ocp.json` (`path` says which), so after editing `flow.toml`, run
`flow import` to write `flow.ocp.json` from it; a file that does not validate is refused and nothing is written.
`flow export` writes `flow.toml` beside the flow, or the file `--out FILE` names. Saving writes the TOML view in a
standard form, so comments and shorthand spellings are not kept, and gates that need check features are saved as
checks; the flow itself does not change. Option values keep their JSON values both ways: an option written as
`true` in `flow.toml` is imported as `true`, and one TOML cannot hold, such as a run's `null`, is saved as a
`[[document.keep]]` entry. `flow init` keeps the option values the runs give; in the answers' own steps option
values are strings.

A changed flow has a new digest. A row may state it in `flow.digest`, as each spec of `flow next` does, and a row
stating another revision's digest is refused, as a run document of another digest is. A row without one names the
flow by `id` and `version` only, so raise `version` when you change the flow: a row of the old version is then
refused. Runs of other revisions are counted apart in `flow status`. Plan again after a change; when runs of the
old plan are recorded, `flow plan` refuses and names `--replace`.

## Never

- Do not invent a score, a cost or a time: record what ran, as it ran.
- Do not give `--allow-simulated` for runs that really ran, or `--replace` a plan without the user's word.
- Do not edit `plan.json`, `flow.ocp.json` or files in the store by hand.

## The flow commands

Every flow command takes `FLOW`, `--home PATH` and `--json`, except where named. Exit codes: 0 ok, with every
verdict of `flow check`; 1 a refusal (the message names what to fix), or a check that cannot be made; 2 usage (a
flag missing or wrong, or no answers with no terminal). A refusal prints `error:` lines on stderr. Under `--json`
stdout holds one object either way, so read the exit code before the object. A command refused before its report
prints nothing else, so the object is the error: `schema` (`loopmath.error/1`), `ok` (false), `command`, `error`
and `exit`. These exit 1 with their own report in its place:

- `flow plan`, refusing a new plan over recorded runs: the plan report, `written` false and `replaced` null;
- `flow record`, refusing what it was given: `flow` and `refused` (section 6), with nothing recorded;
- `flow record` and `flow score`, when bringing `plan.json` up to date fails after the store write: their whole
  report, `plan.problem` saying why;
- `flow show`, for a document with an error finding (nothing on stderr) or a catalog workflow it cannot find:
  `path` and `findings`.

`flow status` and `flow check` write nothing.

- `loopmath flow init DIR --answers FILE --from-ocp FILE --from-rows FILE --toml --json`: `DIR` defaults to
  here; `--from-ocp` takes one or more run documents, and `--from-rows` one rows file.
- `loopmath flow show FLOW --json`.
- `loopmath flow export FLOW --out FILE --json`: `--out` defaults to `flow.toml` beside the flow.
- `loopmath flow import FILE DIR --json`: `FILE` is a `flow.toml`; `DIR` defaults to beside it.
- `loopmath flow plan FLOW --margin M --budget-usd X --replace --json`.
- `loopmath flow next FLOW --n N --seed S --json`: `--n` defaults to 2, `--seed` to 0.
- `loopmath flow record FLOW --rows FILE --allow-unplanned --allow-simulated --json`, or `--ocp FILE ...` in
  place of `--rows FILE`.
- `loopmath flow score FLOW --run RUN --judge ID VALUE --observed-at TS --replace --json`; `--at-attempt ATT`
  names the attempt scored (default: the run's last).
- `loopmath flow status FLOW --json`.
- `loopmath flow check FLOW --arm ARM --json`.
