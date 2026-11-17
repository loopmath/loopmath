# Read rubric scores with loopmath eval (experimental)

`loopmath eval` is for outputs that no test can pass or fail: reports, briefs, summaries, answers. They are scored
on a rubric by model judges and by people. A judge's score is part output and part judge: a judge can be lenient,
can disagree with a person about one output in a way that repeats, and can give a different score on a second
call. `loopmath eval` separates the three, so that the difference between two arms comes with a range that counts
the judges' error, and so that the user can see which judge is worth its price and where a person's time helps
most.

It runs no pipeline and calls no judge. The user brings the scores; each command reads named files and writes one.

It is experimental: it works and has tests, but it has been checked on simulated studies only, on far fewer of them
than its acceptance checks ask for, and its files and output may change. Say so once when you first show a result,
and give the limits in the "Limits" section below before the user relies on a number.

Use exactly the commands below. Their flags and exit codes are in the last section of this guide, "The eval
commands". Never run `--help`. To set up a pipeline so that each run and each judge call is written in the files
below, run `loopmath guide eval-record` and follow it.

## The study folder

A study is a folder. The user writes the first three files; their pipeline and judges fill the next ones.

- `rubric.json`: the components, each with an id, a title and a scale: `binary` (yes or no), `levels` (named
  levels, lowest first), `linear` (a range), `fraction` (0 to 1) or `items` (check items per input).
- `judges.json`: who scores: each judge's id, `kind` (`person` or `model`), for a model its id and family, and
  the `mode` (`per_component`, one call per component, or `panel`, one call for all), and the cost per call.
- `value.json`: a weight per component, the gates an output must pass, and the objective. Needed from step 6.
- `runs.jsonl`: one line per output: its run id, the input it answers, its arm.
- `scores.jsonl`: one line per reading. Two readings of one judge on one output are two lines with two call ids.
- `pairs.jsonl`: one line per comparison: outputs `a` and `b`, which was shown first, and the verdict.
- `items.jsonl`: the check items of an `items` component, per input.
- `fit.json` and `fit.draws.npz`: what `eval fit` writes. Every later command takes `fit.json`.
- `labels-NNN.json`: a plan from `eval labels next`: what to ask a reviewer.

## 1. Check the rubric

Before anything is scored, check the first three files against each other:

```sh
loopmath eval rubric check rubric.json --judges judges.json --value value.json
```

It names what does not fit. Fix each finding with the user before going on.

## 2. Get the scores in

Build or extend the study folder from what the user has: OCP run documents with score signals and preferences,
harness-runs rows, or plain tables of readings and of comparisons.

```sh
loopmath eval scores import --ocp runs/*.json --rubric rubric.json --out study/
loopmath eval scores import --rows FILE --rubric rubric.json --out study/
loopmath eval scores import --csv FILE --pairs-csv FILE --rubric rubric.json --out study/
```

Importing the same rows again changes nothing. `loopmath eval scores add` adds one reading or one pair by hand.

## 3. Check the study

```sh
loopmath eval scores check study/ --rubric rubric.json --judges judges.json
```

It names unknown components, values off their scale and arms that mix settings, and prints, term by term, what the
design can identify and what each term needs. For example, a judge's call noise needs 2 or more calls by that judge
on the same output and component. Tell the user which terms the design cannot identify and what would fix them.

## 4. Fit

```sh
loopmath eval fit study/ --rubric rubric.json --judges judges.json --check-only
loopmath eval fit study/ --rubric rubric.json --judges judges.json
```

`--check-only` prints which terms the design identifies and stops; it writes nothing. The fit itself needs the
optional modelling extra: if it prints the install line for it, tell the user and stop, and install nothing
without their agreement. A fit takes minutes: say so before it starts (the "Limits" section has measured times).

The fit samples, checks convergence and writes `study/fit.json`. If it prints "the fit did not converge", the fit
is written but every later command refuses it. Tell the user; fit again with more `--draws` if they agree. Pass
`--allow-unconverged` to a later command only when the user asks for it, and say that its numbers come from a fit
that did not converge.

## 5. The judges

```sh
loopmath eval judges study/fit.json
```

Per judge and component it prints the bias against the anchor (the people, when there are any), the per-output
disagreement, the call noise, the error of a mean of 1 to 5 calls, and for a yes or no component the false pass and
false fail rates. Add `--target-error X` to list the cheapest judge and repeats that reach an error, or
`--budget-usd USD` for the best judge and repeats within a budget per output.

## 6. Value, and what each step adds

```sh
loopmath eval value study/fit.json --value value.json
loopmath eval steps study/fit.json --value value.json
```

`eval value` gives each arm a value from the weights and gates: the chance that an output passes every gate, the
weighted quality of those that pass, and each arm against the reference with a range and P(better).

`eval steps` reports what each step of a pipeline adds. A measured entry is a swap that was run: two arms that
differ in one step. A hint comes from how step scores and the result move together, and says `correlational`.
Never present a hint as an effect; suggest running the swap first.

## 7. Ask a reviewer

```sh
loopmath eval labels next study/fit.json --mode pilot --n 30 --out study/labels-001.json --form labels-001.csv
loopmath eval scores import --csv labels-001.csv --labels study/labels-001.json --rubric rubric.json --out study/
```

The first command draws a random, blind batch: the plan names outputs and components, never the arm, the model or a
judge's score. Give the user the form to fill in. When it comes back filled, import it with the second command,
then fit again (step 4). After a reviewer batch, `eval value --check design --labels PLAN` adds an estimate that
does not rely on the model, and flags arms where the two disagree; a flag is a reason to look, not a test.

The plan is refused when run ids or refs spell out an arm, a model or a model family, because it would not be
blind. Tell the user to give the runs neutral ids (`r001`), import the scores under those ids and fit again.

`--mode active --minutes M --value value.json` in place of `--mode pilot --n N` picks the outputs where a label is
expected to move the decision most per minute. It is opt-in: offer it only after a pilot batch.

## Try it without data

`eval simulate` writes a practice study with known truth, so the walk can be tried before there are scores:

```sh
loopmath eval simulate --list
loopmath eval simulate linear --out study/ --seed 0
loopmath eval fit study/ --rubric study/rubric.json --judges study/judges.json --allow-simulated
```

The catalog has 17 specs, one per question the group answers; `gated`, `overall-vs-criteria` and `steps-swap` also
write a `value.json`. A simulated study is marked in every run, and `eval fit` and `eval scores import` take it only
with `--allow-simulated`. Never pass `--allow-simulated` for the user's own scores.

## What the numbers mean

- Quality is on 0 to 1 for every scale: the place on a `linear` range, the pass rate of a `binary` component, the
  expected level of a `levels` component, the share of items passed.
- Bias is how much higher a judge scores than the anchor on average. Without a person in `judges.json`, biases are
  relative to the mean of the judges, and the report says so.
- Per-output disagreement (`tau`) repeats when the same output is scored again: more calls do not remove it,
  another judge or a person does. Call noise (`sigma`) changes from call to call: repeats average it away.
- `p_pass` is the chance that one output of an arm passes every gate, from the joint fit; the product of the single
  pass rates, printed beside it as `p_pass_if_independent`, is too low when gated components move together.
- Every range is a 90% interval unless the line says otherwise.

## Limits

Give the ones that touch the user's study before they rely on a number:

- The checks that a fit recovers what a study was simulated from ran on far fewer simulated studies than they ask
  for, and no real study has been through the group yet.
- `linear` and `fraction` readings at the ends of the scale are read as if they were not cut off there. With many
  readings at 0 or at the top, expect arm means pulled toward the middle and noise read too small.
- When a judge scores one output more than once on a yes or no or levels component, the fit often ends not
  converged. One call per output on those components avoids it.
- Fits with yes or no components, levels or comparisons are slow: 12 to 15 minutes on 2 cores for 80 to 120
  outputs at 2 chains of 500 draws, against 1 to 3 minutes with linear scales only. The defaults are 4 chains of
  1000 draws. A linear study of 20,000 readings took about 17 minutes.
- `eval fit --halo full` and `--cuts free` are not built and are refused.
- `eval steps` estimates no interactions between steps, and `eval judges` costs have no range.
- `eval labels next --mode active` beat a random batch by a small margin only in simulation, and is refused under a
  value file whose objective is `cheapest_within_margin` or `cost_per_pass`.

## Never

- Never invent, edit or fill in a score, a label or a fit file. Only the user's judges, reviewers and loopmath write
  them.
- Never run the user's pipeline or a judge to get scores without the user's choice.
- Never present a hint from `eval steps`, or a number from a fit that did not converge, as settled.

## The eval commands

Every eval command takes `--json`, and then prints one object on stdout with a `schema` member, for example
`loopmath.eval-judges-report/1`. Exit codes: 0 ok, also for a fit written as not converged; 1 a refusal (an
`error:` line on stderr names what to fix); 2 usage (a flag missing or wrong). Under `--json` a refusal prints
`schema` (`loopmath.error/1`), `ok` (false), `command`, `error` and `exit`. `FIT` is a fit file; `STUDY` and `DIR`
are study folders.

- `loopmath eval rubric check RUBRIC --judges FILE --value FILE --items FILE --json`; `--fit FILE` adds a report
  on the rubric itself from a fit, with `--allow-unconverged` to take one that did not converge.
- `loopmath eval scores import --out DIR --rubric FILE --json` with one or more of `--ocp FILE ...`, `--rows FILE`,
  `--csv FILE`, `--pairs-csv FILE` and `--items FILE`; `--labels PLAN` for a filled form, `--pair-component ID`
  for a preference that names no component, `--allow-simulated` for simulated runs.
- `loopmath eval scores check STUDY --rubric FILE --judges FILE --require-assignment --json`.
- `loopmath eval scores add STUDY --rubric FILE --judge ID --run RUN --value V --json` for a reading, or
  `--pair RUN_A RUN_B --verdict VERDICT --shown-first FIRST` in place of `--run RUN --value V` for a pair
  (`VERDICT` is `a`, `b` or `tie`, `FIRST` is `a` or `b`); also `--step STEP` (default `final`),
  `--component ID`, `--item ID`, `--of N`, `--call ID`, `--observed-at TIME`, `--replaces ID`,
  `--assignment ID` and `--id ID`.
- `loopmath eval scores export STUDY --run RUN --json`: `--run` is repeatable; without it, the whole study.
- `loopmath eval simulate SPEC --out DIR --seed S --json`, or `loopmath eval simulate --list`.
- `loopmath eval fit STUDY --rubric FILE --judges FILE --check-only --allow-simulated --out FILE --json`; also
  `--seed S`, `--draws N`, `--chains N`, `--keep-draws N`, `--components ID ...`, `--structure FORM`,
  `--terms NAME=on`, `--halo factor` and `--cuts fixed`.
- `loopmath eval judges FIT --target-error X --budget-usd USD --json`.
- `loopmath eval value FIT --value FILE --reference ARM --check design --labels PLAN ... --seed S --json`.
- `loopmath eval steps FIT --value FILE --reference ARM --pool --runs FILE --seed S --json`.
- `loopmath eval labels next FIT --mode MODE --n N --minutes M --value FILE --out FILE --form FILE --json`:
  `MODE` is `pilot` (the default) or `active`; also `--ask ASK` (`scores`, the default, `pairs` or `auto`),
  `--reviewer ID`, `--overlap ID`, `--components ID ...`, `--arms ARM ...`, `--uniform-share X`,
  `--repeat-share X`, `--seed S` and `--explain ID`.

`eval judges`, `eval value`, `eval steps` and `eval labels next` also take `--study DIR` (default: the folder of
`FIT`), `--strict` (refuse when the study files changed since the fit; the default warns) and `--allow-unconverged`.
