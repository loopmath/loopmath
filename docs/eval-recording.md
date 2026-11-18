# Recording runs and judge outputs for `loopmath eval`

`loopmath eval` reads scores you have already recorded. It runs no model and calls no judge. This page says
exactly what to write down, in which file and in which form, so that `eval scores check`, `eval fit`,
`eval judges`, `eval value` and `eval labels` work on your data.

> **Experimental.** `loopmath eval` is experimental in 0.4.5. The file formats and the numbers may change.
> Read the [limits](eval.md#limits) before you rely on a result.

What the commands do and how to read their numbers is in [eval.md](eval.md). This page is about the input.

Every example here is a real file under [`docs/examples/eval-recording/`](examples/eval-recording/), and a test
runs them against the code. All names and numbers in them are made up, and every run in them is marked
`"simulated": true`. That is why the example commands carry `--allow-simulated`. Leave both out for real data.

| Example | What it is |
|---|---|
| `study/`, `study-reviewer.csv`, `study-reviewer-pairs.csv` | A small study of 2 arms on 16 inputs with 2 model judges and 1 reviewer. `record_study.py` writes it. |
| `scales/` | A hand-written study of 2 runs that uses every scale kind, steps, check items, a reviewer who scores in sittings, and corrections. |
| `ocp/` | Two OCP run documents with score signals, for the import. |

## Contents

1. [The short version](#the-short-version)
2. [A first study, start to end](#a-first-study-start-to-end)
3. [The words](#the-words)
4. [The study folder](#the-study-folder)
5. [Recording judge outputs](#recording-judge-outputs): the verdict, the call id, check items, repeats, corrections
6. [Recording comparisons](#recording-comparisons): one verdict per request, and a limit of the current fit
7. [Recording from your pipeline](#recording-from-your-pipeline): three functions to copy
8. [Human reviewers](#human-reviewers): call mode, blind ids, sheets, labels plans
9. [Importing from other tools](#importing-from-other-tools): CSV and OCP
10. [Costs](#costs)
11. [What makes a study fit well](#what-makes-a-study-fit-well)
12. [Before you fit: a short list](#before-you-fit-a-short-list)

Reference, to look things up:

- [Reference: the files, member by member](#reference-the-files-member-by-member)
- [Reference: CSV columns](#reference-csv-columns)
- [Reference: what the check refuses, and the fix](#reference-what-the-check-refuses-and-the-fix)

## The short version

1. Write `rubric.json` (what is scored) and `judges.json` (who scores) by hand, once.
2. Each time your system produces an output, append one line to `runs.jsonl`.
3. Each time a judge answers a request, append one line per score to `scores.jsonl`. Every line of that answer
   carries the id of the request in `call`.
4. Each time a judge compares two outputs, append one line to `pairs.jsonl`.
5. Run `loopmath eval scores check`. Fix what it refuses. Then fit.

## A first study, start to end

This is the whole way once, on the example. `record_study.py` plays the part of your pipeline. It calls no
model: its "system" and its "judges" are random numbers from fixed formulas, so it writes the same files every
time. The shape of every record is the real one. The script needs only Python. In a checkout of the repository
it is at `docs/examples/eval-recording/record_study.py`.

```
$ python docs/examples/eval-recording/record_study.py --out my-study
```

<!-- output: record -->
```
wrote my-study
```

It wrote the folder `my-study` and, beside it, two sheets a reviewer filled in. One output is one line of
`runs.jsonl`:

<!-- file: study/runs.jsonl -->
```json
{"schema": "loopmath.eval-runs/1", "run": "r001", "input": "q15", "arm": "baseline", "steps": [{"step": "final", "model": {"id": "model-small", "family": "beta"}, "effort": "low", "variant": "prompt-a"}], "cost_usd": 0.0108, "simulated": true, "refs": {"input": "inputs/q15.md", "output": "outputs/r001.md"}}
```

One score is one line of `scores.jsonl`. These are a request to `judge-main` for the depth of `r001`, a second
request for the same thing (a repeat, so a new `call`), and one request to `judge-cheap` that scored all four
components at once (one `call`, four lines). Its depth was stored wrong, so a fifth line corrects it under the
same `call`:

<!-- file: study/scores.jsonl -->
```json
{"schema": "loopmath.eval-scores/1", "id": "c0003.depth", "run": "r001", "component": "depth", "judge": "judge-main", "call": "c0003", "value": 5.3, "rubric": "answer@1", "observed_at": "2026-03-02T09:01:00+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0005.depth", "run": "r001", "component": "depth", "judge": "judge-main", "call": "c0005", "value": 4.9, "rubric": "answer@1", "observed_at": "2026-03-02T09:01:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0161.correct", "run": "r001", "component": "correct", "judge": "judge-cheap", "call": "c0161", "value": 1, "rubric": "answer@1", "observed_at": "2026-03-02T09:53:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0161.clarity", "run": "r001", "component": "clarity", "judge": "judge-cheap", "call": "c0161", "value": "poor", "rubric": "answer@1", "observed_at": "2026-03-02T09:53:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0161.depth", "run": "r001", "component": "depth", "judge": "judge-cheap", "call": "c0161", "value": 0.67, "rubric": "answer@1", "observed_at": "2026-03-02T09:53:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0161.overall", "run": "r001", "component": "overall", "judge": "judge-cheap", "call": "c0161", "value": 7.2, "rubric": "answer@1", "observed_at": "2026-03-02T09:53:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0161.depth.fix1", "run": "r001", "component": "depth", "judge": "judge-cheap", "call": "c0161", "value": 6.7, "rubric": "answer@1", "observed_at": "2026-03-02T09:53:40+00:00", "replaces": "c0161.depth"}
```

One comparison is one line of `pairs.jsonl`:

<!-- file: study/pairs.jsonl -->
```json
{"schema": "loopmath.eval-pairs/1", "id": "c0201", "a": "r027", "b": "r009", "judge": "judge-main", "call": "c0201", "shown_first": "a", "verdict": "a", "rubric": "answer@1", "observed_at": "2026-03-02T10:07:00+00:00"}
```

Check the three files written by hand:

```
$ loopmath eval rubric check my-study/rubric.json --judges my-study/judges.json --value my-study/value.json
```

<!-- output: rubric-check -->
```
rubric.json: ok (answer@1, 4 components, 4 dimensions)
judges.json: ok (3 judges, anchor: reviewer-1)
value.json: ok (2 weights, 1 gates)
```

Bring in the reviewer's sheets and check the study. Always give the check both files:

```
$ loopmath eval scores import --csv my-study-reviewer.csv --pairs-csv my-study-reviewer-pairs.csv --rubric my-study/rubric.json --out my-study
$ loopmath eval scores check my-study --rubric my-study/rubric.json --judges my-study/judges.json
```

<!-- output: check -->
```
study my-study: 32 runs, 378 readings (376 read), 22 pairs (22 read), 0 items
rubric answer@1
mixed_arm: none

design (what it can identify):
term           judge        dimension      count  met   needs
run_vs_input   -            -              16     yes   2 or more runs per input
...
sigma          reviewer-1   final/depth    4      yes   2 or more calls by the judge on the same target and component
tau            reviewer-1   final/depth    12     yes   repeats by the judge, and 2 or more judges on the same targets
judge_input    reviewer-1   final/depth    6      no    2 or more runs per input scored by the judge, on 10 or more inputs
sigma          judge-main   final/depth    16     yes   2 or more calls by the judge on the same target and component
tau            judge-main   final/depth    32     yes   repeats by the judge, and 2 or more judges on the same targets
judge_input    judge-main   final/depth    16     yes   2 or more runs per input scored by the judge, on 10 or more inputs
sigma          judge-cheap  final/depth    8      yes   2 or more calls by the judge on the same target and component
tau            judge-cheap  final/depth    32     yes   repeats by the judge, and 2 or more judges on the same targets
judge_input    judge-cheap  final/depth    16     yes   2 or more runs per input scored by the judge, on 10 or more inputs
anchor         -            final/depth    12     weak  6 targets scored by an anchor judge, 30 for a firm anchor
...
family         -            -              2      yes   judges from 2 or more families, each scoring outputs from 2 or more families
halo           judge-cheap  -              8      yes   repeated panel calls on the same targets
...
ok
```

The first line counts 378 readings and reads 376. Two lines were stored wrong and each has a correction: one
score of `judge-cheap` and one of the reviewer. The table says what this design can measure, one row per term,
judge and component, with what each term needs. On `depth` every judge was asked twice on some outputs, so
`sigma` is met. The reviewer scored 12 outputs: enough for an anchor, short of the 30 for a firm one, so the
anchor is `weak`. The last line is `ok` and the exit code is 0. When a row says `no`, the last column says what
to record. See [what makes a study fit well](#what-makes-a-study-fit-well).

Fit. The default engine needs nothing more; `--engine pymc` needs the modelling extra,
`pip install 'loopmath[bayes]'`. The runs of this example are made up and say so, and `eval fit` refuses such a
study unless you allow it:

```
$ loopmath eval fit my-study --rubric my-study/rubric.json --judges my-study/judges.json
```

<!-- stderr: fit-refused -->
```
error: runs.jsonl: run "r001": the run is simulated; pass --allow-simulated to fit a simulated study
```

So the example needs `--allow-simulated`. A study of real runs does not.

```
$ loopmath eval fit my-study --rubric my-study/rubric.json --judges my-study/judges.json --allow-simulated
```

<!-- recorded: fit -->
```
design: 32 runs of 2 arms on 16 inputs, 4 dimensions, 3 judges
readings: 376 of 378 used; pairs: 22 of 22 used; 2 rows not used
anchor: person
  final/correct: weak on 12 anchored targets
...
fit: my-study/fit.json (1000 draws in fit.draws.npz)
converged: yes (R-hat 1.016, effective sample size 334, 0 divergences, 8 s)
```

The fit took 8 seconds on a laptop. Before it fits it prints the design table again and what it will fit.
`eval fit --check-only` prints that part and stops. The default engine does not use MCMC, so it always reports 0
divergences.

Read the judges:

```
$ loopmath eval judges my-study/fit.json
```

<!-- recorded: judges -->
```
eval judges (experimental): fit.json, 3 judges, 1000 draws
Figures are posterior means on the fitting scale; intervals are 90%. A dash is not measured.
On final/correct: bias, tau, sigma and mse are in the anchors' total noise of one reading. The choices use the scale the engine fits, named on each choice line.

judge reviewer-1 (person, per_component, anchor)
  dimension                       bias     tau   sigma   mse(1)   mse(3)     r2  eff(3)
  final/correct                 +0.000       -       -        -        -      -       -
  final/clarity                 +0.000       -       -        -        -      -       -
  final/depth                   +0.000   0.019   0.037   0.0021   0.0011   0.85    2.22
  final/overall                 +0.000   0.071   0.020   0.0063   0.0060   0.66    1.09
  halo: shared share of output disagreement 0.42 (0.06 to 0.84), of call noise does not apply (the judge reads one component per call)
  pairs: position bias 0.01 (-0.21 to 0.28), tie width not measured

judge judge-main (model, model-large, effort high, per_component, 0.03 USD per call)
  dimension                       bias     tau   sigma   mse(1)   mse(3)     r2  eff(3)
  final/correct                 +1.148       -       -        -        -      -       -
  final/clarity                 +0.119       -       -        -        -      -       -
  final/depth                   +0.090   0.043   0.034   0.0138   0.0130   0.71    1.26
  final/overall                 +0.047   0.061   0.038   0.0107   0.0097   0.55    1.22
  final/overall pairs: tie rate 0.23 (0.10 to 0.43), picks the better output 0.67 (0.51 to 0.82)
  halo: shared share of output disagreement 0.32 (0.06 to 0.69), of call noise does not apply (the judge reads one component per call)
  pairs: position bias -0.00 (-0.05 to 0.05), tie width 0.06 (0.03 to 0.10)
  own family (alpha) on final/correct: 0.08 (-0.47 to 0.78)
  own family (alpha) on final/clarity: 0.05 (-0.09 to 0.20)
  own family (alpha) on final/depth: 0.01 (-0.03 to 0.05)
  own family (alpha) on final/overall: -0.03 (-0.11 to 0.03)

judge judge-cheap (model, model-small, effort low, panel, 0.004 USD per call)
  dimension                       bias     tau   sigma   mse(1)   mse(3)     r2  eff(3)
  final/correct                 +1.794   1.152   0.890   7.0325   6.3054   0.14    1.42
  final/clarity                 +0.168   0.198   0.217   0.1557   0.1208   0.21    1.57
  final/depth                   +0.146   0.055   0.097   0.0385   0.0317   0.45    1.90
  final/overall                 +0.091   0.066   0.066   0.0222   0.0191   0.39    1.44
...
```

A dash is a figure this design does not measure. `reviewer-1` and `judge-main` have dashes on `correct` and
`clarity`, where each was asked once per output. `judge-cheap` has all its figures, because its panel requests
were repeated. The reviewer's bias is 0 by definition: the reviewer is the anchor. On `correct`, a yes or no
component, the figures are in the anchors' total noise of one reading, as the third line says. The output goes
on with the other components and a list of what is not measured. What the columns mean is in
[eval.md](eval.md#what-the-numbers-mean).

The same fit answers the question the study was run for:

```
$ loopmath eval value my-study/fit.json --value my-study/value.json
```

<!-- recorded: value -->
```
eval value (experimental): 2 arms, reference baseline, objective best_value
  baseline: value 0.524 [0.161, 0.940]  p_pass 0.389 [0.125, 0.686]  cost $0.01107
      gate final/correct alone: 0.389 [0.125, 0.686]
      final/correct: 0.389 [0.125, 0.686]
      final/clarity: 0.490 [0.340, 0.644]
      final/depth: 0.438 [0.384, 0.492]
  candidate: value 1.311 [0.786, 1.730]  p_pass 0.742 [0.449, 0.947]  cost $0.09735
      gate final/correct alone: 0.742 [0.449, 0.947]
      final/correct: 0.742 [0.449, 0.947]
      final/clarity: 0.638 [0.484, 0.787]
      final/depth: 0.568 [0.515, 0.624]
  candidate against baseline: value 0.787 [0.313, 1.252]  p_pass 0.353 [0.080, 0.641]  p_better 0.998  adopt (p_better 0.998 is at least 0.95)
```

And it plans the next labels for the reviewer:

```
$ loopmath eval labels next my-study/fit.json --mode pilot --n 6 --out my-study/labels-001.json --form labels-001.csv
```

<!-- recorded: labels -->
```
eval labels next (experimental): pilot mode, 6 assignments for reviewer-1 (5 pilot, 1 repeat), 30 minutes, budget 6 assignments
  candidates: 20 in 20 offers
  wrote plan: my-study/labels-001.json
  wrote form: labels-001.csv
```

Your numbers may differ a little from these on another machine. With `--engine pymc` they differ more: its
sampler does not give the same draws on every machine.

The rest of the page is what you need to write these files for your own system.

## The words

| Word | Meaning |
|---|---|
| input | One thing you give your system: a question, a document, a task. Its id stays the same across arms. |
| arm | One configuration of your system: a fixed choice of model, effort and prompt for every step. `eval` compares arms. |
| run | One output: what one arm produced for one input, one time. Running the same arm on the same input again is a new run with a new id. |
| step | A stage of your pipeline whose output can be scored, such as `draft` and `final`. A system with one stage has the single step `final`. |
| target | What a judge looks at: the output of one step of one run. |
| component | One thing the rubric scores, such as `correct` or `clarity`. Each component has one scale. |
| judge | Whoever gives scores: a person, or a model with one prompt, one effort and one way of being asked. Change any of those and it is another judge. |
| call | One request to a judge about one target, or about one pair, and the answer to it. One call gives one score, or several: one per component when the judge scores several components at once, and one per check item for a list of checks. |
| reading | One score: one judge, one call, one target, one component, one value. It is one line of `scores.jsonl`. |
| repeat | The same judge asked again about the same target and component, in a new call. |
| correction | A line that takes the place of a line that was stored wrong. It names the old line in `replaces` and keeps its call. |
| pair | One judge comparing two outputs of the same input and saying which is better, or that they tie. |
| anchor | The judges whose scores set the scale that every other judge's bias is measured against. Usually your reviewers. |

## The study folder

A study is one folder.

```
my-study/
  rubric.json      you write it      the components and their scales
  judges.json      you write it      every judge: person or model, prompt, effort, mode, cost per call
  value.json       you write it      weights, gates and the objective (needed from eval value on)
  runs.jsonl       your pipeline     one line per output
  scores.jsonl     your judges       one line per reading
  pairs.jsonl      your judges       one line per comparison (optional)
  items.jsonl      you write it      the check items of an items component (only with such a component)
  fit.json, fit.draws.npz            eval fit writes them
  labels-001.json                    eval labels next writes it
```

The commands take `rubric.json`, `judges.json` and `value.json` by path (`--rubric`, `--judges`, `--value`), so
those three can live anywhere. Keeping them in the folder is the simplest.

Two things have no file of their own:

- **Calls.** A call is the `call` member of each line of `scores.jsonl` and `pairs.jsonl`.
- **Costs.** They are members of other files. See [Costs](#costs).

Rules that hold for every file:

- `rubric.json`, `judges.json` and `value.json` each hold one JSON document. A `.jsonl` file holds one JSON
  object per line, with no line break inside an object.
- Every document and every line has a `schema` member that names its format.
- A member the format does not know is refused. Put your own data under `ext`, an object that is allowed in
  every object and never read: `"ext": {"prompt_tokens": 812}`.
- Leave out an optional member you have no value for. Do not write `null` in its place. `null` is refused in
  some places where leaving the member out is fine: `"mode": null` and `"blinded": null` in `judges.json` are
  two.
- The ids of judges, components, steps, rubrics, rubric versions, items and model families may not hold a
  space, `/` or `@`. Run, input, arm, call, reading and pair ids can be any text that is not empty.
- Ids are compared letter for letter. `Depth` is not `depth`.
- A time is ISO 8601 with an offset: `2026-03-02T09:00:00+00:00`.
- An id appears once per `.jsonl` file: a run id in `runs.jsonl`, a reading id in `scores.jsonl`, a pair id in
  `pairs.jsonl`, an item id in `items.jsonl`.
- Mark a run `"simulated": true` only when it is made up. `eval fit` and the import of runs then ask for
  `--allow-simulated`, so that made-up scores do not slip into a real study.

Every member of every file is in the [reference](#reference-the-files-member-by-member). In short:

| File | One entry is | It must name |
|---|---|---|
| `rubric.json` | a component | its id and its scale |
| `judges.json` | a judge | its id and kind; for a model its model id and family; its `mode` when one request scores all components |
| `runs.jsonl` | an output | the run id, the input, the arm, and the model of each step |
| `scores.jsonl` | a score | its own id, the run, the component, the judge, the call, the value, the rubric and the time |
| `pairs.jsonl` | a comparison | its own id, the two runs, the judge, the call, the verdict, the rubric and the time |
| `items.jsonl` | a check item | its id, the input and the component |

**The identity of a judge is its entry in `judges.json`.** A reading carries only the judge id, so the model, the
prompt, the effort, the mode and the reference of a judge must not change under one id. When you change the
judge prompt, add a new judge (`judge-main-v3`) and keep the old one. The fit then tells the two apart instead
of mixing them. Give each reviewer their own id (`reviewer-1`, `reviewer-2`). One shared id for all people
hides how much they differ.

All runs of one arm should have the same `steps`. `eval scores check` lists an arm whose runs differ in a model,
an effort or a variant under `mixed_arm`.

## Recording judge outputs

The examples in this section are lines of `scales/scores.jsonl`. Its rubric and judges are shown in the
[reference](#reference-the-files-member-by-member).

### The raw verdict on its scale

Write down what the judge said, on the scale the rubric declares. Do not average repeats. Do not round. Do not
turn words into numbers. Do not turn around a scale where less is better: declare `better: lower` instead.

| Scale kind | `value` | Not accepted |
|---|---|---|
| `binary` | The number `1` for yes or pass, `0` for no or fail. | `true`, `false`, `"pass"`, `"yes"` |
| `levels` | Exactly one of the declared levels. A word level as text (`"good"`), a number level as a whole number (`4`). | `"Good"`, `"4"`, `4.0` |
| `linear` | A number inside the declared range. | A number outside it, a word |
| `fraction` | A number from 0 to 1. | A percentage such as `80` |
| `items` | Per check item: `item` and `value` 0 or 1. Or one count line: `value` passed out of `of`. | Both forms in one call |

One yes or no reading, one word level, one number level, one score on 0 to 10 with the cost of its call, one
score of the `draft` step, one fraction and one count of errors:

<!-- file: scales/scores.jsonl -->
```json
{"schema": "loopmath.eval-scores/1", "id": "c0001.correct", "run": "r101", "component": "correct", "judge": "judge-main", "call": "c0001", "value": 1, "rubric": "report@2", "observed_at": "2026-03-02T09:00:00+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0002.clarity", "run": "r101", "component": "clarity", "judge": "judge-main", "call": "c0002", "value": "good", "rubric": "report@2", "observed_at": "2026-03-02T09:00:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0003.tone", "run": "r101", "component": "tone", "judge": "judge-main", "call": "c0003", "value": 4, "rubric": "report@2", "observed_at": "2026-03-02T09:00:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0004.depth", "run": "r101", "component": "depth", "judge": "judge-main", "call": "c0004", "value": 7.5, "rubric": "report@2", "observed_at": "2026-03-02T09:01:00+00:00", "cost_usd": 0.031}
{"schema": "loopmath.eval-scores/1", "id": "c0005.depth", "run": "r101", "step": "draft", "component": "depth", "judge": "judge-main", "call": "c0005", "value": 6, "rubric": "report@2", "observed_at": "2026-03-02T09:01:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0006.sourced", "run": "r101", "component": "sourced", "judge": "judge-main", "call": "c0006", "value": 0.8, "rubric": "report@2", "observed_at": "2026-03-02T09:01:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0007.errors", "run": "r101", "component": "errors", "judge": "judge-main", "call": "c0007", "value": 2, "rubric": "report@2", "observed_at": "2026-03-02T09:02:00+00:00"}
```

### One request, one call id

The `call` member is how `eval` knows which scores came out of the same request. The rule has no exception:
**one request to a judge gets one call id, and every line written from its answer carries that id.** A new
request gets a new id. How many lines one request gives depends on what was asked.

**One component per request** (`mode: per_component`, the default). The answer is one score, so the call has one
line. The seven lines above are seven requests.

**All components in one request** (`mode: panel`). The answer has one score per component, so the call has
several lines. Scores from one request tend to move together, and the shared id is what lets the fit measure
that. This is one panel request to `judge-cheap` about run `r102`:

<!-- file: scales/scores.jsonl -->
```json
{"schema": "loopmath.eval-scores/1", "id": "c0011.correct", "run": "r102", "component": "correct", "judge": "judge-cheap", "call": "c0011", "value": 0, "rubric": "report@2", "observed_at": "2026-03-02T09:03:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0011.clarity", "run": "r102", "component": "clarity", "judge": "judge-cheap", "call": "c0011", "value": "ok", "rubric": "report@2", "observed_at": "2026-03-02T09:03:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0011.depth", "run": "r102", "component": "depth", "judge": "judge-cheap", "call": "c0011", "value": 5.5, "rubric": "report@2", "observed_at": "2026-03-02T09:03:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0011.sourced", "run": "r102", "component": "sourced", "judge": "judge-cheap", "call": "c0011", "value": 0.5, "rubric": "report@2", "observed_at": "2026-03-02T09:03:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0011.covers", "run": "r102", "component": "covers", "judge": "judge-cheap", "call": "c0011", "value": 2, "of": 3, "rubric": "report@2", "observed_at": "2026-03-02T09:03:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0011.overall", "run": "r102", "component": "overall", "judge": "judge-cheap", "call": "c0011", "value": 3, "rubric": "report@2", "observed_at": "2026-03-02T09:03:20+00:00"}
```

**A list of checks in one request** (a component of kind `items`). The checks of each input are kept in
[`items.jsonl`](#itemsjsonl). A judge that answers each check gives one line per check item. They came out of
one request, so they share one call id, also for a `per_component` judge. This is one request to `judge-main`
about the component `covers`, with three items:

<!-- file: scales/scores.jsonl -->
```json
{"schema": "loopmath.eval-scores/1", "id": "c0008.covers.deadline", "run": "r101", "component": "covers", "judge": "judge-main", "call": "c0008", "value": 1, "item": "doc-01.deadline", "rubric": "report@2", "observed_at": "2026-03-02T09:02:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0008.covers.owner", "run": "r101", "component": "covers", "judge": "judge-main", "call": "c0008", "value": 0, "item": "doc-01.owner", "rubric": "report@2", "observed_at": "2026-03-02T09:02:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0008.covers.cost", "run": "r101", "component": "covers", "judge": "judge-main", "call": "c0008", "value": 1, "item": "doc-01.cost", "rubric": "report@2", "observed_at": "2026-03-02T09:02:20+00:00"}
```

The fit reads these three lines as one reading: 2 of 3 passed. Three call ids would pass the check, and the fit
would read three requests of one item each, which is another model of the same answer. A judge that only says
how many checks passed writes one count line. This one says 2 of 3:

<!-- file: scales/scores.jsonl -->
```json
{"schema": "loopmath.eval-scores/1", "id": "c0011.covers", "run": "r102", "component": "covers", "judge": "judge-cheap", "call": "c0011", "value": 2, "of": 3, "rubric": "report@2", "observed_at": "2026-03-02T09:03:20+00:00"}
```

A judge has one mode. If you ask the same model both ways, declare two judges. People have a mode too: see
[Human reviewers](#human-reviewers).

Use an id the request already has: the request id the API returns, or a UUID you make when you send the
request. A call belongs to one judge and one target. `eval scores check` warns when a call id is shared across
judges or targets.

### The id of a line is not the id of the call

Every line has its own `id`, and no two lines of the file share one. `call` is shared by the lines of one
request. Build the line id from the call, the component and the check item: `c0011.depth`,
`c0008.covers.deadline`. Then the same score gets the same id every time you export, and a line written twice
is caught as a duplicate.

### Repeats

A **repeat** is the same judge asked again about the same target and component. It is a new request, so it is a
new line with a new call id. Do not overwrite the first reading and do not average the two. This is a second
request for the depth of `r101`, after call `c0004` above:

<!-- file: scales/scores.jsonl -->
```json
{"schema": "loopmath.eval-scores/1", "id": "c0010.depth", "run": "r101", "component": "depth", "judge": "judge-main", "call": "c0010", "value": 7, "rubric": "report@2", "observed_at": "2026-03-02T09:03:00+00:00"}
```

Repeat a panel request as a whole: ask for all components again, under one new call id.

### Corrections

A **correction** takes the place of a line that was stored wrong: a typing slip of a reviewer, a parser that
misread an answer. The judge was not asked again. So the new line gets

- its own `id`,
- the **same `call`** as the line it corrects,
- that line's id in `replaces`.

Keep the old line. Here a reviewer typed 5 and meant 3:

<!-- file: scales/scores.jsonl -->
```json
{"schema": "loopmath.eval-scores/1", "id": "h0001", "run": "r102", "component": "tone", "judge": "reviewer-1", "call": "h0001", "value": 5, "rubric": "report@2", "observed_at": "2026-03-02T14:10:00+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "h0001.fix1", "run": "r102", "component": "tone", "judge": "reviewer-1", "call": "h0001", "value": 3, "rubric": "report@2", "observed_at": "2026-03-02T14:12:00+00:00", "replaces": "h0001"}
```

Keeping the call matters most for a request with several lines. Here `reviewer-2` scored three components of
`r101` in one sitting, call `h0002`. The depth was typed as 1.5 and is corrected to 7.5. All lines keep
`h0002`, so the sitting is still one request with three scores:

<!-- file: scales/scores.jsonl -->
```json
{"schema": "loopmath.eval-scores/1", "id": "h0002.tone", "run": "r101", "component": "tone", "judge": "reviewer-2", "call": "h0002", "value": 4, "rubric": "report@2", "observed_at": "2026-03-02T15:00:00+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "h0002.depth", "run": "r101", "component": "depth", "judge": "reviewer-2", "call": "h0002", "value": 1.5, "rubric": "report@2", "observed_at": "2026-03-02T15:00:00+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "h0002.overall", "run": "r101", "component": "overall", "judge": "reviewer-2", "call": "h0002", "value": 4, "rubric": "report@2", "observed_at": "2026-03-02T15:00:00+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "h0002.depth.fix1", "run": "r101", "component": "depth", "judge": "reviewer-2", "call": "h0002", "value": 7.5, "rubric": "report@2", "observed_at": "2026-03-02T15:00:00+00:00", "replaces": "h0002.depth"}
```

The same holds for a check item. This line corrects the `owner` item of call `c0008` above, which now reads 3
of 3:

<!-- file: scales/scores.jsonl -->
```json
{"schema": "loopmath.eval-scores/1", "id": "c0008.covers.owner.fix1", "run": "r101", "component": "covers", "judge": "judge-main", "call": "c0008", "value": 1, "item": "doc-01.owner", "rubric": "report@2", "observed_at": "2026-03-02T09:02:20+00:00", "replaces": "c0008.covers.owner"}
```

A correction under a new call id passes the check, and it is wrong: the fit then sees two requests where there
was one, and the scores of the answer no longer move together in the model.

`eval` counts every line and reads only the ones that are not replaced. That is why the report on this folder
says "25 readings (22 read)".

Two lines for one judge, target and component are a repeat unless one names the other in `replaces`. Nothing is
read as a correction because it is newer. When an answer was unusable, cut off or not parseable, and you asked
again, record only the answer you used.

## Recording comparisons

A pair is a judge saying which of two outputs of the same input is better. `a` and `b` are run ids. The verdict
is `a`, `b` or `tie`. These are the lines of `scales/pairs.jsonl`:

<!-- file: scales/pairs.jsonl -->
```json
{"schema": "loopmath.eval-pairs/1", "id": "p0001", "a": "r101", "b": "r102", "judge": "reviewer-1", "call": "p0001", "shown_first": "b", "verdict": "a", "rubric": "report@2", "observed_at": "2026-03-02T14:20:00+00:00"}
{"schema": "loopmath.eval-pairs/1", "id": "c0012", "a": "r101", "b": "r102", "step": "draft", "component": "depth", "judge": "judge-main", "call": "c0012", "shown_first": "a", "verdict": "tie", "rubric": "report@2", "observed_at": "2026-03-02T09:04:00+00:00", "cost_usd": 0.05}
{"schema": "loopmath.eval-pairs/1", "id": "h0003.depth", "a": "r102", "b": "r101", "component": "depth", "judge": "reviewer-2", "call": "h0003", "shown_first": "a", "verdict": "b", "rubric": "report@2", "observed_at": "2026-03-02T15:10:00+00:00"}
{"schema": "loopmath.eval-pairs/1", "id": "h0003.overall", "a": "r102", "b": "r101", "component": "overall", "judge": "reviewer-2", "call": "h0003", "shown_first": "a", "verdict": "tie", "rubric": "report@2", "observed_at": "2026-03-02T15:10:00+00:00"}
{"schema": "loopmath.eval-pairs/1", "id": "h0003.overall.fix1", "a": "r102", "b": "r101", "component": "overall", "judge": "reviewer-2", "call": "h0003", "shown_first": "a", "verdict": "b", "rubric": "report@2", "observed_at": "2026-03-02T15:10:00+00:00", "replaces": "h0003.overall"}
```

The first is a reviewer comparing two outputs overall. The second is a model judge comparing the `draft` step
on `depth`. Each is one request with one verdict. That is the form to collect for the fit.

Which run is `a` says nothing about the order shown. Pick `a` and `b` at random, so that `a` is not always the
same arm, and record which one the judge saw first in `shown_first`. To measure the lean to the first one
shown, ask about the same pair twice, once in each order. That is two calls and two lines.

The last three lines are a record-format example. They show how to write down a request that compared on
several components, and they come with a limit. `reviewer-2` compared the same two outputs on two components
in one sitting: one call, one line per component, and the line id is the call and the component. The last
line corrects the fourth: its own id, the same call, and `replaces`.

### One pair request, several verdicts: a limit of the current fit

The shared call id keeps the record true to what was asked. But the current fit does not model noise shared
by the verdicts of one pair request. For scores it does: the scores of one panel request share a term in the
model. For pairs there is no such term in loopmath 0.4.x. The fit reads the two verdicts of call `h0003` as
two independent requests, although verdicts given in one sitting tend to move together. The check passes
either way and says nothing.

So:

- **For a study meant for the current fit, ask for one verdict per pair request.** Either make one request
  per compared component, or ask for one overall verdict per pair request.
- **Answers you already have keep their true call id.** If a judge compared several components in one
  request, record it that way. Giving each verdict a call id of its own does not fix the model: the fit comes
  out the same, and the record of what was asked is lost. A later version that models the shared noise can
  only use it if the call ids are true.
- Read results from such pairs with care. Two verdicts from one sitting can carry less information than two
  separate requests, and the current fit counts them as two.

## Recording from your pipeline

When you own the code that calls the system and the judges, write the lines yourself, at the moment each thing
happens. Three small functions are enough. These are from `record_study.py`:

<!-- file: record_study.py -->
```python
def _append(path: Path, record: dict) -> None:
    """Append one JSON line. A member with no value is left out, never written as null."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({k: v for k, v in record.items() if v is not None}) + "\n")


def record_run(study: Path, *, run: str, input: str, arm: str, steps: list[dict], cost_usd: float | None = None,
               refs: dict | None = None, simulated: bool | None = None) -> None:
    """One output of the pipeline: which input it answers, under which arm, and what each step ran on.
    Pass `simulated=True` only for made-up runs. Leave it out for the runs of a real system."""
    _append(study / "runs.jsonl", {"schema": "loopmath.eval-runs/1", "run": run, "input": input, "arm": arm,
                                   "steps": steps, "cost_usd": cost_usd, "simulated": simulated, "refs": refs})


def record_reading(study: Path, *, run: str, component: str, judge: str, call: str, value, observed_at: str,
                   rubric: str = RUBRIC_REF, step: str | None = None, item: str | None = None,
                   of: int | None = None, cost_usd: float | None = None, reading_id: str | None = None,
                   replaces: str | None = None, assignment: str | None = None) -> str:
    """One score out of one request to a judge. Returns the id of the line.

    `call` is the id of the request. Every line that came out of one request has the same `call`: the
    components of a panel answer, and the check items of an items answer. A new request is a new `call`.

    The id of the line is the call, the component and the item, so it is the same on every rerun. To correct
    a line that was stored wrong, write a new line with its own `reading_id`, the SAME `call` (no new request
    was made) and the id of the wrong line in `replaces`."""
    reading_id = reading_id or ".".join(part for part in (call, component, item) if part)
    _append(study / "scores.jsonl", {
        "schema": "loopmath.eval-scores/1", "id": reading_id, "run": run, "step": step, "component": component,
        "judge": judge, "call": call, "value": value, "item": item, "of": of, "rubric": rubric,
        "observed_at": observed_at, "cost_usd": cost_usd, "replaces": replaces, "assignment": assignment})
    return reading_id


def record_pair(study: Path, *, a: str, b: str, judge: str, call: str, verdict: str, shown_first: str,
                observed_at: str, rubric: str = RUBRIC_REF, component: str | None = None,
                step: str | None = None, cost_usd: float | None = None, pair_id: str | None = None,
                replaces: str | None = None) -> str:
    """One comparison of two outputs of the same input. `verdict` is "a", "b" or "tie". Returns the id.

    The id of the line is the call, and the component when one is named. A correction is a new line with its
    own `pair_id`, the same `call` and the id of the wrong line in `replaces`.

    Make one request per compared component, or ask for one overall verdict per pair request. A request that
    compared on several components is recorded as one line per component, all with the same `call`. That
    keeps the record true to what was asked, but the current fit does not model noise shared by the verdicts
    of one pair request: it reads them as independent. Keep the true `call` all the same. A call id per
    verdict does not change the fit, and it loses the record of what was asked."""
    pair_id = pair_id or ".".join(part for part in (call, component) if part)
    _append(study / "pairs.jsonl", {
        "schema": "loopmath.eval-pairs/1", "id": pair_id, "a": a, "b": b, "step": step, "component": component,
        "judge": judge, "call": call, "shown_first": shown_first, "verdict": verdict, "rubric": rubric,
        "observed_at": observed_at, "cost_usd": cost_usd, "replaces": replaces})
    return pair_id
```

`call` and the id of the line are separate arguments. Left alone, the line id is the call, the component and
the item. A correction passes its own `reading_id` and the call of the line it corrects.

A judge that is asked for one component per request, with a second request on some outputs:

<!-- file: record_study.py -->
```python
    for n, run in enumerate(runs):
        for component in components:
            record_reading(out, run=run, component=component, judge="judge-main", call=made.call_id(),
                           value=made.ask("judge-main", run, [component])[component], observed_at=made.now())
        if n % 2 == 0:
            for component in ("depth", "overall"):
                record_reading(out, run=run, component=component, judge="judge-main", call=made.call_id(),
                               value=made.ask("judge-main", run, [component])[component], observed_at=made.now())
```

A judge that is asked for all components in one request. One score of the first answer was stored with the
point one place off, and the fix follows it:

<!-- file: record_study.py -->
```python
    for n, run in enumerate(runs):
        for _ in range(2 if n % 4 == 1 else 1):
            call = made.call_id()
            answer, at = made.ask("judge-cheap", run, components), made.now()
            for component in components:
                slip = n == 0 and component == "depth"                     # stored with the point one place off
                record_reading(out, run=run, component=component, judge="judge-cheap", call=call,
                               value=round(answer[component] / 10, 2) if slip else answer[component],
                               observed_at=at)
            if n == 0:
                # The fix for the slip: a new line with its own id, the SAME call (the judge was not asked
                # again) and the id of the wrong line in `replaces`. The panel request stays one request.
                record_reading(out, run=run, component="depth", judge="judge-cheap", call=call,
                               value=answer["depth"], observed_at=at, reading_id=f"{call}.depth.fix1",
                               replaces=f"{call}.depth")
```

A run of a real system is recorded without `simulated`. The example passes `simulated=True` because its runs
are made up.

If several processes judge at the same time, let each write its own file and join the files into `scores.jsonl`
afterwards. Then no two writers share a file.

## Human reviewers

A reviewer is a judge of kind `person`. Their scores are readings and their comparisons are pairs, in the same
two files as everyone else's. Name the reviewers in `anchor`: the fit then measures every model judge against
them.

### A reviewer has a call mode too

Decide how each reviewer works, and record it the same way a model judge is recorded.

- **One component at a time.** The reviewer scores all outputs on one component, then moves to the next. Each
  score is a request of its own: `mode` is `per_component` (the default) and every row has its own call.
  `reviewer-1` of the example study works this way.
- **One sitting per output.** The reviewer opens an output and scores all components before moving on. That is
  a panel request: declare `"mode": "panel"` and give the rows of one sitting the same call. `reviewer-2` of
  `scales/` works this way, as shown under [Corrections](#corrections).

A second look at an output is a repeat. For a reviewer who works in sittings, the second look is a new sitting
with a new call id.

### Keeping reviewers blind

A reviewer who can tell which arm wrote an output will lean, without meaning to. Three things keep them blind.

1. **Neutral run ids.** `r001`, not `candidate-017`. Hand the ids out in shuffled order. If the first half of the
   numbers is one arm, the number gives the arm away. `record_study.py` shuffles before it numbers the runs.
2. **Neutral refs.** The paths in `refs` are what a reviewer opens. `outputs/r001.md`, not
   `outputs/candidate/q01.md`.
3. **Show the output, not the record.** A reviewer sees the input and the output. They do not see `runs.jsonl`,
   the arm, the model or any judge's score.

`eval labels next` checks the first two. It refuses to plan when a run id or a ref holds an arm name, a model
id or a model family. This is what a study of real runs gets:

<!-- recorded: not-blind -->
```
error: fit.json: runs: 32 of 32 candidates have a run id or a ref that spells out an arm, a model id or a model family, so the plan would not be blind; give the runs ids and refs that do not name the configuration (r001, outputs/r001), import the scores under those ids and fit again
```

When every run is marked simulated, as in the examples of this page, the plan is written with a note that it
is not blind.

The check looks for those names anywhere in the text. An input file called `alpha-particles.md` trips it when a
model family is called `alpha`. Nothing checks this earlier, so choose neutral ids on the first day. Changing
run ids later means writing all three `.jsonl` files again.

For pairs, pick at random which output is shown first, and record it in `shown_first`.

### From a sheet

Reviewers often score in a spreadsheet. Save it as CSV and import it. This is the head of
`study-reviewer.csv`. The reviewer worked one component at a time, so every row has its own `call`:

<!-- file: study-reviewer.csv -->
```
id,call,run,component,judge,value,observed_at,replaces
rv.r026.correct,rv.r026.correct,r026,correct,reviewer-1,0,2026-03-02T10:12:20+00:00,
rv.r024.correct,rv.r024.correct,r024,correct,reviewer-1,0,2026-03-02T10:12:40+00:00,
rv.r019.correct,rv.r019.correct,r019,correct,reviewer-1,1,2026-03-02T10:13:00+00:00,
```

A reviewer who looks at an output a second time adds a row with a new `call`. That is a repeat. A reviewer who
fixes a slip adds a row with a new `id`, the `call` of the wrong row and its id in `replaces`. That is a
correction. The first score of the depth of `r026`, the second look and the fix:

<!-- file: study-reviewer.csv -->
```
id,call,run,component,judge,value,observed_at,replaces
rv.r026.depth,rv.r026.depth,r026,depth,reviewer-1,8.1,2026-03-02T10:20:20+00:00,
rv.r026.depth.2,rv.r026.depth.2,r026,depth,reviewer-1,4.7,2026-03-02T10:28:20+00:00,
rv.r026.depth.fixed,rv.r026.depth,r026,depth,reviewer-1,5.1,2026-03-02T10:31:00+00:00,rv.r026.depth
```

A row with no `call` column is read as a request of its own. That is right for a reviewer who works one
component at a time and wrong for one who works in sittings: for sittings, add the `call` column and give the
rows of one sitting the same value.

For comparisons a reviewer can answer `left`, `right` or `tie` in a `choice` column. With `shown_first`, the
import turns `left` into the output that was shown first. This is the head of `study-reviewer-pairs.csv`:

<!-- file: study-reviewer-pairs.csv -->
```
a,b,judge,shown_first,choice,observed_at
r026,r024,reviewer-1,a,right,2026-03-02T10:31:20+00:00
r007,r019,reviewer-1,b,right,2026-03-02T10:31:40+00:00
r023,r004,reviewer-1,a,left,2026-03-02T10:32:00+00:00
```

```
$ loopmath eval scores import --csv my-study-reviewer.csv --pairs-csv my-study-reviewer-pairs.csv --rubric my-study/rubric.json --out my-study
```

<!-- output: import-reviewer -->
```
study my-study
runs: 0 new, 0 unchanged, 0 refused
readings: 57 new, 0 unchanged, 0 refused
pairs: 6 new, 0 unchanged, 0 refused
items: 0 new, 0 unchanged, 0 refused
wrote scores.jsonl, pairs.jsonl
```

Importing the same file again changes nothing:

<!-- output: import-reviewer-again -->
```
study my-study
runs: 0 new, 0 unchanged, 0 refused
readings: 0 new, 57 unchanged, 0 refused
pairs: 0 new, 6 unchanged, 0 refused
items: 0 new, 0 unchanged, 0 refused
nothing new: no file changed
```

The columns the import reads are in the [reference](#reference-csv-columns).

### Labels that `eval` asks for

Once a fit exists, `eval labels next` picks the outputs a reviewer should score next. It writes a plan and,
with `--form`, a CSV form. The form names the run, the step and the component of each score to give, never the
arm. The reviewer fills the `value` column. The filled form comes back through the same import, with the plan:

```
$ loopmath eval labels next my-study/fit.json --mode pilot --n 6 --out my-study/labels-001.json --form labels-001.csv
$ loopmath eval scores import --csv labels-001.csv --labels my-study/labels-001.json --rubric my-study/rubric.json --out my-study
```

The form has no `call` column, so every filled row comes in as a request of its own. For a reviewer who works
in sittings, add a `call` column to the filled form, with one value per sitting, before you import it.

Each row of the form carries its `assignment`. With `eval scores check --require-assignment`, a reading by a
person that names no assignment is refused. Use that once all your labels come from plans.

## Importing from other tools

`eval scores import` builds or extends a study folder from scores kept somewhere else. It checks every row
first. If one row is refused, nothing is written. Rows it already has are left as they are, so you can import
the same export every day.

| Your scores are in | Use |
|---|---|
| your own pipeline code | Write the `.jsonl` lines yourself, as [above](#recording-from-your-pipeline). No import is needed. |
| a spreadsheet, or any table you can save as CSV | `--csv` for readings, `--pairs-csv` for comparisons. The columns are in the [reference](#reference-csv-columns). |
| OCP run documents | `--ocp`. Runs and scores come from the same documents. |
| rows written by a loopmath harness, with `outcome.judges` | `--rows` |
| a list of check items | `--items` adds an `items.jsonl` to the study. |

Always pass `--rubric`. The import then reads values on the right scale and stamps each row with the rubric.

A CSV row has no call unless you give it one. For a judge that scores several components in one request, add a
`call` column and give the rows of one request the same value.

### OCP run documents

If your runs are already written as OCP documents, the same documents can carry the scores. A score is a signal
of the run. The parts `eval` needs that OCP has no place for go in the signal's `ext`, under `dev.loopmath.eval`.
These are two requests to one judge about one output, from `ocp/r201.ocp.json`:

<!-- file: ocp/r201.ocp.json -->
```json
   {"id": "c0901.depth", "kind": "score", "name": "depth", "value": 5.5, "tier": "reported",
    "observed_at": "2026-03-03T09:00:00+00:00", "source": {"kind": "judge", "ref": "judge-main"},
    "ext": {"dev.loopmath.eval": {"call": "c0901", "rubric": "answer@1"}}},
   {"id": "c0902.depth", "kind": "score", "name": "depth", "value": 6.0, "tier": "reported",
    "observed_at": "2026-03-03T09:00:20+00:00", "source": {"kind": "judge", "ref": "judge-main"},
    "ext": {"dev.loopmath.eval": {"call": "c0902", "rubric": "answer@1"}}},
```

| In the OCP document | Becomes |
|---|---|
| `run.id` | the run id |
| `run.task.id` | the input |
| `run.flow.arm`, else `run.labels.arm`, else the id of the configuration | the arm |
| `run.configuration.settings` | the steps: model id, family and effort of each |
| the sum of `cost.usd` over the attempts | the run's `cost_usd` |
| `run.ext["dev.loopmath.simulate"]`, an object | the run is marked `simulated` |
| a signal of kind `score`: `name`, `value`, `observed_at`, `id` | a reading: component, value, time, id |
| the signal's `source.ref` | the judge |
| `ext["dev.loopmath.eval"]`: `call`, `step`, `rubric`, `replaces`, `assignment`, `item`, `of` | the members of the same names |
| `ext["dev.loopmath.eval"].level`, with `value` null | the value of a `levels` component with word levels |

The two example documents are made up and carry the simulate mark, so the import needs `--allow-simulated`.
Documents of real runs carry no mark and need no flag.

```
$ loopmath eval scores import --ocp docs/examples/eval-recording/ocp/r201.ocp.json docs/examples/eval-recording/ocp/r202.ocp.json --rubric my-study/rubric.json --out ocp-study --allow-simulated
```

<!-- output: import-ocp -->
```
study ocp-study
runs: 2 new, 0 unchanged, 0 refused
readings: 6 new, 0 unchanged, 0 refused
pairs: 0 new, 0 unchanged, 0 refused
items: 0 new, 0 unchanged, 0 refused
not used: signal 's3' (r202.ocp.json): "latency_s" is not a component of the rubric
wrote runs.jsonl, scores.jsonl
```

A signal whose name is not a component of the rubric is listed as not used. That is how other signals of the
run, such as a latency, stay out.

**Repeats need the `dev.loopmath.eval` member.** `r202.ocp.json` is written the way another tool would write it,
without that member. It has two `depth` signals from one judge. Without a call id, `eval` cannot know that they
are two requests. It reads the newer one as a correction of the older one, so one reading is left:

<!-- output: check-ocp -->
```
study ocp-study: 2 runs, 6 readings (5 read), 0 pairs (0 read), 0 items
...
```

Six readings came in and five are read. If the signals are repeats, write `call` in the member, as in
`r201.ocp.json`.

## Costs

There is no cost file. Record costs in three places.

| Where | Member | Who reads it |
|---|---|---|
| each line of `runs.jsonl` | `cost_usd`: what it cost to produce this output | The mean over an arm's runs is the arm's cost in `eval value` and `eval steps`. `eval labels next --mode active` needs it under a `best_under_cost` objective. |
| each judge in `judges.json` | `cost_usd_per_call`: what one call of the judge costs | `eval judges`, to price repeats and to pick a judge under `--budget-usd`. |
| each line of `scores.jsonl` and `pairs.jsonl` | `cost_usd`: what this call cost | Nothing in 0.4.5. It is kept, so record it when you have it. |

Record the cost of a run when the run ends. It is hard to rebuild later.

## What makes a study fit well

`eval scores check` prints a table of what your design can measure, with what each term needs. Read it before
you spend on judges. These are the needs, as the table words them:

| To measure | Record |
|---|---|
| how much a judge's score changes from call to call (`sigma`) | 2 or more calls by the judge on the same target and component |
| how far a judge is off on a given output, every time it looks (`tau`) | repeats by the judge, and 2 or more judges on the same targets |
| a judge's bias against people (`anchor`) | targets scored by an anchor judge: 6 at least per dimension, 30 for a firm anchor |
| the true pass or fail behind a yes or no component (`binary_status`) | repeats by the anchor, or 2 or more judges on the same targets |
| whether a judge favours its own model family (`family`) | judges from 2 or more families, each scoring outputs from 2 or more families |
| whether the scores of one call move together (`halo`) | repeated panel calls on the same targets |
| a lean to the output shown first (`position`) | pairs judged in both orders, or `shown_first` balanced across pairs |
| the noise of comparisons and the width of a tie (`sigma_pair`, `tie_width`) | repeats of pairs by the judge, and ties. One verdict per pair request: the current fit has no term for noise shared by the verdicts of one request |
| the difference between arms apart from the inputs (`run_vs_input`) | 2 or more runs per input |

A step and a component together are a **dimension**, written `final/depth`. The table and the fit work per
dimension.

In practice:

1. **Run every arm on the same inputs.** Then an arm is compared on like for like.
2. **Use at least two judges, and let every judge score outputs of every arm.** If one judge scores only one
   arm, a lenient judge and a good arm look the same.
3. **Repeat on a share of the targets.** Pick them at random, not where a score looked odd. In the example,
   `judge-main` gets a second request on half of the outputs and `judge-cheap` on a quarter. With
   `--engine pymc`, repeats on yes or no and levels components have a known problem: the fit often ends marked
   as not converged. See the 0.4.4 entry of the changelog. The default engine converged on every such fit
   measured for 0.4.5. The example has 8 such repeated requests and converged. If your fit does not, keep one
   request per output on those components and repeat on the number scales.
4. **Get some human labels on the same outputs the model judges scored.** 6 outputs per dimension is the least
   that counts. 30 makes a firm anchor. Let a reviewer score a few outputs twice.
5. **Keep an arm fixed.** One arm, one setting of every step.
6. **Record costs** on runs and on judges. The cost side of `eval value` and `eval judges` is empty without them.
7. **Keep the scale wide enough.** When most scores sit at the top or the bottom of a scale, the fit reads them
   badly. See the [limits](eval.md#limits).

## Before you fit: a short list

- Every output has one line in `runs.jsonl`, with a neutral run id, its input, its arm and its cost. Only
  made-up runs are marked `simulated`.
- Every judge has an entry in `judges.json` with its mode, and a changed prompt or model is a new judge.
- Every request to a judge has one call id, and every line of its answer carries it: all components of a panel
  answer, all items of a list of checks.
- Every line has its own id. A repeat has a new call id. A correction has a new line id, the old call id and
  `replaces`.
- Every pair request asked for one verdict: one component, or the output overall.
- Values are raw and on the rubric's scale. A member with no value is left out, not written as `null`.
- A share of the targets has repeats, two or more judges score every arm, and a reviewer has scored some of the
  same outputs.
- `eval scores check` with `--rubric` and `--judges` ends with `ok`.

## Reference: the files, member by member

In the tables, `[]` marks a member of each entry of a list. A member with a default is optional. A member of an
optional object is required only when the object is there.

### `rubric.json`

<!-- file: scales/rubric.json -->
```json
{
 "schema": "loopmath.eval-rubric/1",
 "id": "report",
 "version": "2",
 "steps": ["draft", "final"],
 "components": [
  {"id": "correct", "title": "The facts are right", "scale": {"kind": "binary"}},
  {"id": "clarity", "title": "Clear to a new reader", "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]}},
  {"id": "tone", "title": "Tone fits the reader", "scale": {"kind": "levels", "levels": [1, 2, 3, 4, 5]}},
  {"id": "depth", "title": "Depth", "scale": {"kind": "linear", "range": [0, 10]}, "applies_to": ["draft", "final"]},
  {"id": "sourced", "title": "Share of claims with a source", "scale": {"kind": "fraction"}},
  {"id": "errors", "title": "Errors left in the text", "scale": {"kind": "linear", "range": [0, 20]}, "better": "lower"},
  {"id": "covers", "title": "Covers the required points", "scale": {"kind": "items"}, "review_minutes": 3},
  {"id": "overall", "title": "Overall", "role": "overall", "scale": {"kind": "linear", "range": [1, 5]}}
 ]
}
```

<!-- fields: loopmath.eval-rubric/1 -->
| Member | Type | Required | Meaning |
|---|---|---|---|
| `schema` | text | yes | `loopmath.eval-rubric/1` |
| `id` | id | yes | The name of the rubric. |
| `version` | id or whole number | yes | Change it when a wording or a scale changes. Every reading names the rubric as `id@version`. |
| `steps` | list of ids | no, default `["final"]` | The steps whose output is scored, in pipeline order. The last one must be `final`. |
| `components` | list | yes | One entry per component, at least one. |
| `components[].id` | id | yes | The name readings use in `component`. |
| `components[].title` | text | no, default the id | A line for people. |
| `components[].role` | `criterion` or `overall` | no, default `criterion` | `overall` marks the overall judgement. One per step at most. Pairs that name no component are read as comparisons of it. |
| `components[].scale` | object | yes | The scale of the component. |
| `components[].scale.kind` | `binary`, `levels`, `linear`, `fraction` or `items` | yes | The scale. See [the raw verdict on its scale](#the-raw-verdict-on-its-scale). |
| `components[].scale.levels` | list of words or whole numbers | for `levels` only | 3 or more levels, lowest first. For 2 levels use `binary`. |
| `components[].scale.range` | `[low, high]` | for `linear` only | The ends of the scale. |
| `components[].better` | `higher` or `lower` | no, default `higher` | `lower` for a scale where less is better, such as a count of errors. |
| `components[].applies_to` | list of steps | no, default `["final"]` | The steps at which the component is scored. |
| `components[].anchors` | text or object | no | Where the example outputs for each score are kept. Stored, not read. |
| `components[].review_minutes` | number above 0 | no, default 1 | Minutes a person needs to score it. `eval labels next` counts them against its budget. |
| `components[].compare_minutes` | number above 0 | no, default half of `review_minutes` | Minutes a person needs to compare two outputs on it. |
| `components[].human_only` | true or false | no, default false | `eval rubric check` warns when a judge that is not a person is listed for it. |

### `judges.json`

<!-- file: scales/judges.json -->
```json
{
 "schema": "loopmath.eval-judges/1",
 "anchor": ["reviewer-1", "reviewer-2"],
 "min_anchor_targets": 6,
 "judges": [
  {"id": "reviewer-1", "kind": "person", "blinded": true},
  {"id": "reviewer-2", "kind": "person", "mode": "panel", "blinded": true},
  {"id": "judge-main", "kind": "model",
   "model": {"id": "model-large", "family": "alpha", "provider": "example-cloud"},
   "effort": "high", "prompt": "rubric-prompt-v2", "mode": "per_component",
   "reference": {"shown": true, "author": {"kind": "person"}, "version": "ref-2026-03"},
   "blinded": true, "cost_usd_per_call": 0.03},
  {"id": "judge-cheap", "kind": "model", "model": {"id": "model-small", "family": "beta"},
   "effort": "low", "prompt": "rubric-prompt-v2", "mode": "panel", "blinded": true, "cost_usd_per_call": 0.004,
   "components": ["correct", "clarity", "depth", "sourced", "covers", "overall"]}
 ]
}
```

<!-- fields: loopmath.eval-judges/1 -->
| Member | Type | Required | Meaning |
|---|---|---|---|
| `schema` | text | yes | `loopmath.eval-judges/1` |
| `judges` | list | yes | One entry per judge. |
| `judges[].id` | id | yes | The name readings use in `judge`. |
| `judges[].kind` | `model`, `person` or `command` | yes | `command` is a script or a program that scores. |
| `judges[].model` | object | for kind `model` only | The judge model. A judge of another kind has none. |
| `judges[].model.id` | text | yes | The id of the model. |
| `judges[].model.family` | id | yes | The model family. Use the same family names as in `runs.jsonl`: the fit measures whether a judge favours outputs of its own family. |
| `judges[].model.provider` | text | no | Who serves the model. |
| `judges[].effort` | text | no | The effort or reasoning setting of the judge. |
| `judges[].prompt` | text | no | The name or version of the judge prompt. |
| `judges[].mode` | `per_component` or `panel` | no, default `per_component` | `per_component`: one request scores one component. `panel`: one request scores all of them. For a person: one component at a time, or one sitting per output. Leave it out for the default; `null` is refused. |
| `judges[].reference` | object | no | The reference answer the judge saw, if any. |
| `judges[].reference.shown` | true or false | no, default false | Whether the judge saw a reference answer. |
| `judges[].reference.author` | object | when `shown` is true | Who wrote the reference. |
| `judges[].reference.author.kind` | `model`, `person` or `verified` | yes | `verified`: a model wrote it and a person checked it. |
| `judges[].reference.author.model` | object | for author kinds `model` and `verified` | The model that wrote the reference. |
| `judges[].reference.author.model.id` | text | yes | Its id. |
| `judges[].reference.author.model.family` | id | yes | Its family. |
| `judges[].reference.version` | text | no | The version of the reference set. |
| `judges[].blinded` | true or false | no, default false | True when the judge could not see which arm wrote the output. It is kept with the fit. No number depends on it in 0.4.5. Leave it out for the default; `null` is refused. |
| `judges[].cost_usd_per_call` | number, 0 or more | no | What one call of this judge costs. `eval judges` prices repeats with it. |
| `judges[].components` | list of component ids | no, default all | The components this judge scores. |
| `anchor` | list of judge ids | no | The anchor judges. They must be of kind `person`. Without an anchor, a judge's bias is measured against the mean of the judges. |
| `min_anchor_targets` | whole number, 1 or more | no, default 6 | With fewer targets scored by an anchor judge on a dimension, the anchor is not used there: bias is then measured against the mean of the judges. Below 30 targets the design table calls the anchor `weak`. |

### `runs.jsonl`

<!-- file: scales/runs.jsonl -->
```json
{"schema": "loopmath.eval-runs/1", "run": "r101", "input": "doc-01", "arm": "short-prompt", "split": "dev", "steps": [{"step": "draft", "model": {"id": "model-small", "family": "beta"}, "effort": "low", "variant": "prompt-short"}, {"step": "final", "model": {"id": "model-large", "family": "alpha"}, "effort": "high"}], "cost_usd": 0.21, "simulated": true, "refs": {"input": "inputs/doc-01.md", "draft": "outputs/r101.draft.md", "output": "outputs/r101.md"}}
{"schema": "loopmath.eval-runs/1", "run": "r102", "input": "doc-01", "arm": "long-prompt", "split": "dev", "steps": [{"step": "draft", "model": {"id": "model-small", "family": "beta"}, "effort": "low", "variant": "prompt-long"}, {"step": "final", "model": {"id": "model-large", "family": "alpha"}, "effort": "high"}], "cost_usd": 0.26, "simulated": true, "refs": {"input": "inputs/doc-01.md", "draft": "outputs/r102.draft.md", "output": "outputs/r102.md"}}
```

<!-- fields: loopmath.eval-runs/1 -->
| Member | Type | Required | Meaning |
|---|---|---|---|
| `schema` | text | yes | `loopmath.eval-runs/1` |
| `run` | text | yes | The id of this output. Keep it neutral: see [keeping reviewers blind](#keeping-reviewers-blind). |
| `input` | text | yes | The id of the input it answers. Runs of different arms on the same input share it. |
| `arm` | text | yes | The configuration that wrote it. |
| `steps` | list | yes | One entry per step of the pipeline: what it ran on. |
| `steps[].step` | id | yes | The step, named as in the rubric. |
| `steps[].model` | object | yes | The model of the step. |
| `steps[].model.id` | text | yes | Its id. |
| `steps[].model.family` | id | yes | Its family. |
| `steps[].effort` | text | no | The effort setting of the step. |
| `steps[].variant` | text | no | Anything else that sets this arm apart: a prompt version, a tool set. |
| `split` | text | no | A label such as `dev` or `test`. Kept with the run. |
| `cost_usd` | number, 0 or more | no | What it cost to produce this output. |
| `simulated` | true or false | no, default false | True only for made-up runs. `eval fit` refuses them, and the import refuses to bring them in, unless you pass `--allow-simulated`. |
| `refs` | object of texts | no | Where the input and the output are kept, for example `{"input": "...", "output": "..."}`. A labels plan copies them so that a reviewer can open the output. |

### `scores.jsonl`

<!-- fields: loopmath.eval-scores/1 -->
| Member | Type | Required | Meaning |
|---|---|---|---|
| `schema` | text | yes | `loopmath.eval-scores/1` |
| `id` | text | yes | The id of this line. No two lines share one. Build it from the call, the component and the item. A correction has its own. |
| `run` | text | yes | The run scored. It must be in `runs.jsonl`. |
| `step` | id | no, default `final` | The step whose output was scored. |
| `component` | id | yes | The rubric component. |
| `judge` | id | yes | The judge, as named in `judges.json`. |
| `call` | text | yes | The id of the request to the judge. All lines of one answer share it: the components of a panel answer, the items of a list of checks. A repeat has a new one. A correction keeps the one of the line it corrects. |
| `value` | number or level | yes | The raw verdict on the component's scale. |
| `item` | id | `items` components only | The check item this line answers. The value is then 0 or 1. |
| `of` | whole number, 1 or more | `items` components only | The count form: `value` items passed out of `of`. Never together with `item`. |
| `rubric` | `id@version` | yes | The rubric the score was given under. |
| `observed_at` | time | yes | When the judge answered. |
| `replaces` | text | no | The id of the line this one corrects. |
| `assignment` | text | no | The id of the `eval labels next` assignment this reading answers. |
| `cost_usd` | number, 0 or more | no | What this call cost. Kept with the reading. `eval judges` prices from `judges.json`, not from here. |

### `pairs.jsonl`

<!-- fields: loopmath.eval-pairs/1 -->
| Member | Type | Required | Meaning |
|---|---|---|---|
| `schema` | text | yes | `loopmath.eval-pairs/1` |
| `id` | text | yes | The id of this line. Build it from the call, and the component when one call compared on several. |
| `a` | text | yes | One run. |
| `b` | text | yes | The other run. It must answer the same input as `a`. |
| `step` | id | no, default `final` | The step whose outputs were compared. |
| `component` | id | no, default the overall component of the step | What they were compared on. |
| `judge` | id | yes | The judge. |
| `call` | text | yes | The id of the request to the judge. A correction keeps it. Several verdicts of one request share it, but the current fit does not model noise shared by them: see [the limit](#one-pair-request-several-verdicts-a-limit-of-the-current-fit). |
| `shown_first` | `a` or `b` | no | Which output the judge saw first. Record it: without it the fit cannot measure a lean to the first one shown. |
| `verdict` | `a`, `b` or `tie` | yes | Which output is better. |
| `rubric` | `id@version` | yes | The rubric. |
| `observed_at` | time | yes | When the judge answered. |
| `replaces` | text | no | The id of the line this one corrects. |
| `assignment` | text | no | The id of the `eval labels next` assignment it answers. |
| `cost_usd` | number, 0 or more | no | What this call cost. |

### `items.jsonl`

Only a rubric with an `items` component needs this file. Each line is one check for one input.

<!-- file: scales/items.jsonl -->
```json
{"schema": "loopmath.eval-items/1", "id": "doc-01.deadline", "input": "doc-01", "component": "covers", "label": "States the deadline", "source": {"kind": "expert"}}
{"schema": "loopmath.eval-items/1", "id": "doc-01.owner", "input": "doc-01", "component": "covers", "label": "Names who is responsible", "source": {"kind": "expert"}}
{"schema": "loopmath.eval-items/1", "id": "doc-01.cost", "input": "doc-01", "component": "covers", "label": "Gives the cost", "source": {"kind": "model_verified", "model": "model-large"}}
```

<!-- fields: loopmath.eval-items/1 -->
| Member | Type | Required | Meaning |
|---|---|---|---|
| `schema` | text | yes | `loopmath.eval-items/1` |
| `id` | id | yes | The name readings use in `item`. |
| `input` | text | yes | The input the check belongs to. |
| `component` | id | yes | The `items` component it belongs to. |
| `label` | text | no | The check, in words. |
| `source` | object | no | Who wrote the check. |
| `source.kind` | `expert`, `model` or `model_verified` | yes | A person, a model, or a model with a person's check. |
| `source.model` | text | for kinds `model` and `model_verified` | The model that wrote it. |

### `value.json`

`eval value` needs it, the recording does not. It is here so that the list of files is complete. How it is read
is in [eval.md](eval.md).

<!-- file: study/value.json -->
```json
{
 "schema": "loopmath.eval-value/1",
 "weights": {
  "clarity": 1,
  "depth": 2
 },
 "gates": [
  {
   "component": "correct"
  }
 ],
 "objective": {
  "kind": "best_value",
  "reference": "baseline"
 }
}
```

<!-- fields: loopmath.eval-value/1 -->
| Member | Type | Required | Meaning |
|---|---|---|---|
| `schema` | text | yes | `loopmath.eval-value/1` |
| `weights` | object | one weight or one gate at least | A weight, 0 or more, per component id. |
| `gates` | list | one weight or one gate at least | The components an output must pass. |
| `gates[].component` | id | yes | The component. |
| `gates[].at_least` | number or level | no | The threshold on the component's own scale. Not needed for a `binary` component. |
| `gates[].step` | id | no, default `final` | The step. |
| `gated_value` | number | no, default 0 | The value of an output that fails a gate. |
| `objective` | object | no, default kind `best_value` | What to optimise. |
| `objective.kind` | `best_value`, `cheapest_within_margin`, `best_under_cost` or `cost_per_pass` | yes | The kind of objective. |
| `objective.margin` | number, 0 or more | for `cheapest_within_margin` | The value an arm may give up. |
| `objective.reference` | text | no | The arm every other arm is compared against. |
| `objective.cost_cap_usd` | number above 0 | for `best_under_cost` | The cost cap, in USD. |

## Reference: CSV columns

<!-- columns: scores -->
| Column of `--csv` | Required | Meaning |
|---|---|---|
| `run` | yes | The run scored. |
| `component` | yes | The component. |
| `judge` | yes | The reviewer's judge id. |
| `value` | yes | The score. A row with an empty value is a blank line of a form and is skipped. |
| `step` | no | Default `final`. |
| `call` | no | The request or sitting the row came out of. Without it every row is its own call, except that item rows of one run, step, component and judge with the same `observed_at` share one. |
| `item` | no | The check item, for an `items` component. |
| `of` | no | The count form, for an `items` component. |
| `observed_at` | no | Without it the time the file was last changed is used. |
| `id` | no | Without it the id is made from the content of the row, so the same file gives the same ids. |
| `replaces` | no | The id of the row this one corrects. Give the row the `call` of the row it corrects. |
| `assignment` | no | The `eval labels next` assignment the row answers. |
| `rubric` | no | Default the rubric given with `--rubric`. |
| `cost_usd` | no | What the call cost. |

<!-- columns: pairs -->
| Column of `--pairs-csv` | Required | Meaning |
|---|---|---|
| `a` | yes | One run. |
| `b` | yes | The other run. |
| `judge` | yes | The reviewer's judge id. |
| `verdict` | this or `choice` | `a`, `b` or `tie`. |
| `choice` | this or `verdict` | `left`, `right` or `tie`. `left` is the output shown first, so the row needs `shown_first`. |
| `shown_first` | no | `a` or `b`. |
| `step` | no | Default `final`. |
| `component` | no | Default the overall component. |
| `call` | no | Without it every row is its own call. |
| `observed_at` | no | Without it the time the file was last changed is used. |
| `id` | no | Without it the id is made from the content of the row. |
| `replaces` | no | The id of the row this one corrects. |
| `assignment` | no | The `eval labels next` assignment the row answers. |
| `rubric` | no | Default the rubric given with `--rubric`. |
| `cost_usd` | no | What the call cost. |

A column with another name is not read. The import says so in a line that starts with `ignored columns`.

A sheet without ids has a trap. The id of a row is then made from its content. Change a value in the sheet and
import it again, and the changed row comes in as a new row beside the old one: a repeat, not a correction. Give
the rows ids, and correct with a new row and `replaces`.

## Reference: what the check refuses, and the fix

Run the check with both files. Without `--judges` it cannot tell a known judge from a misspelt one.

```
$ loopmath eval scores check my-study --rubric my-study/rubric.json --judges my-study/judges.json
```

It prints every problem on its own line and exits 1. Each message below is the real one, made by breaking
`scales/` in one place.

### A call recorded twice

Two readings of the same component under one call id:

<!-- refusal: call-twice -->
```
error: scores.jsonl: call "c0004": two readings of one call for the component "depth" ("c0004.depth" and "c0004.depth.again"); give a repeat its own call, or name the reading it corrects in replaces
```

Fix: if the judge was asked again, give the second reading the id of the second call. If the second line
corrects the first, add `replaces` and keep the call. If your exporter ran twice and wrote the line twice, the
ids are equal too, and you see this instead:

<!-- refusal: id-twice -->
```
error: scores.jsonl: line 26: id: "c0004.depth" is already on line 4; give every line its own id
```

Fix: write each reading once. Ids built from the call and the component make a second copy easy to see.

### Missing ids

<!-- refusal: no-call, no-id, no-rubric, no-time -->
```
error: scores.jsonl: line 4: missing member "call"; add it
error: scores.jsonl: line 4: missing member "id"; add it
error: scores.jsonl: line 4: missing member "rubric"; add it
error: scores.jsonl: line 4: missing member "observed_at"; add it
```

Fix: keep the id of each request when you make it, and write it with the score. If old scores have no call id,
give each reading its own, such as the reading's id. Then every reading is read as a request of its own, which
is right for a judge that scores one component per request and wrong for a panel judge and for check items.

A time needs an offset:

<!-- refusal: bad-time -->
```
error: scores.jsonl: line 4: observed_at: "2026-03-02T09:01:00" is not an ISO 8601 time with an offset; write it like 2026-01-15T09:30:00-08:00
```

### Mixed scales

A component has one scale. A value in another form is refused:

<!-- refusal: binary-word, level-case, level-text, fraction-percent, linear-range -->
```
error: scores.jsonl: id "c0001.correct": value: "pass" is not 0 or 1; the component "correct" is binary: give 0 or 1
error: scores.jsonl: id "c0002.clarity": value: "Good" is not a level of "clarity"; use one of poor, ok, good
error: scores.jsonl: id "c0003.tone": value: "4" is not a level of "tone"; use one of 1, 2, 3, 4, 5
error: scores.jsonl: id "c0006.sourced": value: 80 is outside the scale of "sourced"; give a number from 0 to 1
error: scores.jsonl: id "c0004.depth": value: 75 is outside the scale of "depth"; give a number from 0 to 10
```

Fix: convert where you record, in one place. Write `1` and `0` for pass and fail. Write a level as it is spelt
in the rubric. A level that is a number must be a whole number in the file: `4`, not `"4"` and not `4.0`. Write
a fraction as 0.8, not 80.

If two judges answer on different number ranges, such as 1 to 5 and 0 to 10, ask both for the range in the
rubric. For old scores, convert with one straight-line rule and keep the judge's own number under `ext`. If a
judge answers in words and another in numbers, they are scoring two different components. Declare two.

The check cannot see a score that is on the wrong scale and still inside the range, such as 0.7 written under
a 0 to 10 component. Look at the lowest and highest value per judge and component before you fit.

### Names that do not match

<!-- refusal: unknown-judge, unknown-component, unknown-run, rubric-version -->
```
error: scores.jsonl: id "c0004.depth": judge: unknown judge "judge-new"; use one of reviewer-1, reviewer-2, judge-main, judge-cheap, or add the judge to the judges file
error: scores.jsonl: id "c0004.depth": component: unknown component "Depth"; use one of correct, clarity, tone, depth, sourced, errors, covers, overall
error: scores.jsonl: id "c0004.depth": run: no run "r999" in runs.jsonl; import the run, or fix the reading
error: scores.jsonl: id "c0004.depth": rubric: unknown rubric version "report@1"; this check reads "report@2": give the rubric the row was scored under
```

Fix: add the judge to `judges.json`, spell the component as the rubric does, and write the run line before the
readings of the run. A reading scored under an older rubric version stays under that version. Check and fit it
with that version of the rubric file.

### Members the format does not know

<!-- refusal: extra-member -->
```
error: scores.jsonl: line 4: model: unknown member; remove it, or move it under "ext"
```

Fix: the judge's model belongs in `judges.json`. Anything of your own goes under `ext`.

### Check items

<!-- refusal: items-both, unknown-item -->
```
error: scores.jsonl: call "c0008": the call has item lines and a count line for the component "covers" ("c0008.covers.deadline" and "c0008.covers"); write item lines or one count line, never both
error: scores.jsonl: id "c0008.covers.deadline": item: unknown item "doc-01.none"; add it to items.jsonl (eval scores import --items)
```

Fix: per call, write either one line per item or one count line. Every item named must be in `items.jsonl`.

### Pairs and corrections

<!-- refusal: pair-inputs, replaces-unknown, replaces-self -->
```
error: pairs.jsonl: id "p0001": the runs "r101" and "r103" have different inputs ("doc-01" and "doc-02"); compare two runs of one input
error: scores.jsonl: id "c0010.depth": replaces: "c0000.depth" names no reading; name the id of the reading it corrects
error: scores.jsonl: line 12: replaces: a reading replaces itself; name the earlier reading it corrects
```

Fix: compare only outputs of the same input. In `replaces`, name the id of another line that is in the file. A
correction needs an id of its own: a line that names its own id in `replaces` gets the last message.

### What the check does not refuse

These pass the check and still hurt. Look for them yourself.

- **A score on the wrong scale inside the range.** See above.
- **A call structure that does not match the request.** A `panel` judge whose readings each have their own
  call id passes. So do the check items of one request under one call id each, and a correction under a new
  call id. In each case the fit sees more requests than were made.
- **Several verdicts under one pair call.** The record is right when that is what was asked, but the current
  fit reads the verdicts as independent. See
  [the limit](#one-pair-request-several-verdicts-a-limit-of-the-current-fit).
- **Run ids that name the arm.** The check passes them. `eval labels next` refuses them later.
- **A misspelt judge, when you leave out `--judges`.**
- **An arm that mixes settings.** It is listed under `mixed_arm` and the check still says `ok`.
- **Made-up runs with no `simulated` mark.** Nothing can tell them from real ones.

`eval fit` is more forgiving than the check. A reading with an unknown judge or an unknown run is left out and
counted in a line such as "1 rows not used". So run the check first, every time.
