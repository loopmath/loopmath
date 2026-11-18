# Reading rubric scores from judges and people (experimental)

`loopmath eval` is experimental in 0.4.5: it works and has tests, but it has been checked on simulated studies
only, not all of its recovery checks pass yet, and its files and output may change. The limits are listed at the
end of this page; read them before you rely on a number.

Use it when the outputs you compare have no test to pass: reports, briefs, summaries, answers. Such outputs are
scored on a rubric, by model judges and by people. A judge's score is part output and part judge: a judge can be
lenient, can disagree with a person about one output in a way that repeats, and can give a different score on a
second call. `loopmath eval` separates the three, so that the difference between two arms comes with a range that
counts the judges' error, and so that you can see which judge is worth its price and where a person's time helps
most.

It does not run your pipeline and it does not call a judge. You bring the scores; each command reads named files
and writes one.

## The files

A study is a folder. You write the first three files, your pipeline and judges fill the next ones.

| File | What it holds |
|---|---|
| `rubric.json` | The components of the rubric: an id, a title and a scale each. A scale is `binary` (yes or no), `levels` (named levels, lowest first), `linear` (a range, for example 0 to 10), `fraction` (0 to 1) or `items` (a list of check items per input). `better: lower` turns a scale around. A component has the role `criterion`, or `overall` for an overall judgement. A rubric with `steps` scores each step of a pipeline. |
| `judges.json` | Who scores: each judge's id, `kind` (`person` or `model`), for a model its id and family, effort and prompt version, the `mode` (`per_component`: one call per component, or `panel`: one call for all), whether a reference answer was shown and who wrote it, and the cost per call. |
| `value.json` | What you want: a weight per component, the gates (a component an output must pass, with `at_least` for a scale with levels or numbers), and the objective. Optional until `eval value`. |
| `runs.jsonl` | One line per output: its run id, the input it answers, its arm, and the model and effort of each step. |
| `scores.jsonl` | One line per reading: run, step, component, judge, call and value. Two readings of one judge on one output are two lines with two call ids. |
| `pairs.jsonl` | One line per comparison: outputs `a` and `b`, which was shown first, and the verdict (`a`, `b` or `tie`). |
| `items.jsonl` | The check items of an `items` component, per input. |
| `fit.json`, `fit.draws.npz` | What `eval fit` writes. Every later command reads these two. |
| `labels-NNN.json` | A plan `eval labels next` writes: what to ask a reviewer. |

`loopmath eval rubric check rubric.json --judges judges.json --value value.json` reads the first three against each
other before anything is scored, and names what does not fit.

## The walk

```
$ loopmath eval rubric check rubric.json --judges judges.json --value value.json
$ loopmath eval scores import --ocp runs/*.json --rubric rubric.json --out study/
$ loopmath eval scores check study/ --rubric rubric.json --judges judges.json
$ loopmath eval fit study/ --rubric rubric.json --judges judges.json
$ loopmath eval judges study/fit.json
$ loopmath eval labels next study/fit.json --mode pilot --n 30 --out study/labels-001.json --form labels-001.csv
$ loopmath eval scores import --csv labels-001.csv --labels study/labels-001.json --rubric rubric.json --out study/
$ loopmath eval fit study/ --rubric rubric.json --judges judges.json
$ loopmath eval value study/fit.json --value value.json
$ loopmath eval steps study/fit.json --value value.json
$ loopmath eval labels next study/fit.json --value value.json --mode active --minutes 60 --out study/labels-002.json
```

1. **Get the scores in.** `eval scores import` builds or extends the study folder from OCP run documents with score
   signals and preferences (`--ocp`), from harness-runs rows (`--rows`), or from plain tables (`--csv` for readings,
   `--pairs-csv` for comparisons). Importing the same rows again changes nothing. `eval scores add` adds single
   readings by hand, and `eval scores export` prints the study as OCP signals and preferences: all of it, one run
   (`--run`), or the readings and comparisons of named judges (`--judges`).
2. **Check the study.** `eval scores check` reads the folder against the rubric and the judges: unknown components,
   values off their scale, arms that mix settings. It also prints a table of what the design can identify, term by
   term, with what each term needs: for example, a judge's call noise needs 2 or more calls by that judge on the
   same output and component.
3. **Fit.** `eval fit` uses the Laplace engine by default, an approximation that needs no extra. `--engine pymc`
   samples with PyMC instead and needs the optional modelling extra: `pip install 'loopmath[bayes]'`. It first
   prints which terms the design identifies (`--check-only` prints that and stops), then fits, checks convergence
   and writes `fit.json`. A fit that did not converge is written and marked; later commands refuse it unless you pass
   `--allow-unconverged`.
4. **Read the judges.** `eval judges` prints, per judge and component: the bias against the anchor (the people,
   when there are any), the per-output disagreement, the call noise, the error of a mean of 1 to 5 calls, and for
   a yes or no component the false pass and false fail rates. With `--target-error` or `--budget-usd` it lists
   which judge and how many repeats reach an error or fit a budget. On a yes or no component that error is on the
   scale the engine fits, the anchors' call noise (see the hidden quality under What the numbers mean).
5. **Ask a reviewer.** `eval labels next --mode pilot` draws a random, blind batch: the plan names outputs and
   components, never the arm, the model or a judge's score. `--form` writes a table to fill in. The filled table
   comes back with `eval scores import --csv FORM --labels PLAN --rubric rubric.json`. Then fit again.
6. **Decide.** `eval value` turns the fit and the value file into a value per arm: the chance that an output passes
   every gate, the weighted quality of those that pass, and each arm against the reference with a range and
   P(better). `eval steps` reports what each step of a pipeline adds. `eval labels next --mode active` plans the
   next reviewer batch where a label is expected to move the decision most per minute.

## Trying it without data

`eval simulate` writes a whole study from a spec with known truth, so you can run the walk before you have scores:

```
$ loopmath eval simulate --list
$ loopmath eval simulate linear --out study/ --seed 0
$ loopmath eval fit study/ --rubric study/rubric.json --judges study/judges.json --allow-simulated
```

The catalog has 17 specs, one per question the group answers (`linear`, `binary-checks`, `levels-5`, `gated`,
`pairs`, `own-family`, `panel-vs-per-component`, `steps-swap` and others); `gated`, `overall-vs-criteria` and
`steps-swap` also write a `value.json`. A simulated study is marked as such in
every run, `truth.json` holds the parameters it was drawn from, and `eval fit` and `eval scores import` take it only
with `--allow-simulated`, so that simulated runs never mix into real ones by accident.

## What the numbers mean

- **Reported quality** is on 0 to 1 for every scale: the place on a `linear` range, the pass rate of a `binary`
  component, the expected level of a `levels` component, the share of items passed. Reports also print it on the
  component's own scale where that reads better.
- **Bias** is how much higher a judge scores than the anchor on average. The anchor is the people in
  `judges.json`; without a person, biases are relative to the mean of the judges and the report says so.
- **Per-output disagreement** (`tau`) is the part of a judge's error that repeats when the same output is scored
  again. More calls by the same judge do not remove it. Every judge has its own, people too. Scoring the same
  outputs with other independent judges or people helps the fit tell quality from disagreement and narrows the
  ranges, but it does not remove any judge's disagreement.
- **Call noise** (`sigma`) is the part that changes from call to call. Repeats average it away.
- **Hidden quality of a yes or no component.** Behind each pass rate the fit keeps a hidden quality per output,
  in units of the anchors' total noise of one reading: how far one reading strays from the output's hidden
  quality, counting the per-output disagreement, the call noise and the part that repeats across the outputs of
  one input. The anchors are the people in the `anchor` list of `judges.json` who scored the component; when none
  did, every judge who scored it. With several, the unit is the geometric mean of their total noises. So for one
  reading by a judge with no bias, no family or position effect, and a total noise equal to the unit (one anchor
  alone, for instance), an output with hidden quality 1 passes about 84% of the time; another judge's readings
  pass at a rate set by its own bias and noise. A component with comparisons only uses the anchors' total noise
  of one comparison instead (the people in the `anchor` list who compared on it, else every judge who did): for
  such a judge, a gap of 1 between two outputs is called right about 84% of the time, before ties and the order
  they were shown in. Pass rates, reported quality and every decision the reports make (verdicts, judge choices,
  flags, label picks) do not depend on this unit. It shows in the hidden numbers: the `fitting` figures, and the
  bias, per-output disagreement, call noise and error of the judges table, in plain output as under `--json`,
  which on such a component are in the anchors' total noise. The judge choices of `--target-error` and
  `--budget-usd` are made on the scale the engine fits, the anchors' call noise (their call noise of one
  comparison for comparisons only); `--target-error X` is read on that scale and each choice line says so. Fits written
  before 0.4.5 used the anchors' call noise alone as the unit; they are still read, and each dimension's `unit`
  in `fit.json` says which unit it holds.
- **Halo** is the part of a judge's error shared across components: by output (`halo_target`) or, for a `panel`
  judge, by call (`halo_call`).
- **Family effect** is the extra a judge gives to outputs written by its own model family, or by the family that
  wrote the reference answer it was shown. It is reported with a range and flagged when the range excludes 0.
  On a yes or no component the flag reads the range on the scale the engine fits, which can differ a little
  from the range shown; the marker then says "(in the anchors' call noise)".
- **Relative** marks a component that has comparisons but no scores. Its arms can be ordered but not placed on the
  scale, so a gate on it is refused.
- **p_pass** is the chance that one output of an arm passes every gate. It is computed from the joint fit: gated
  components usually move together, and the product of the single pass rates (printed beside it as
  `p_pass_if_independent`) is then too low.
- Every range is a 90% interval unless the line says otherwise.

## Reading `eval steps`

`eval steps` prints two things and keeps them apart. **Measured** entries are swaps that were run: two arms that
differ in one step. Their value difference is an effect. **Hint** entries come from how step scores and the final
value move together across runs; a step can correlate with the result without driving it, and each hint line says
`correlational`. Run the swap before you act on a hint.

## Reading `eval value --check design`

After a reviewer batch from `eval labels next`, `eval value --check design --labels PLAN` prints, per arm and
component, a second estimate of the reviewer's mean that does not trust the model: the judges' mean plus the
weighted mean gap between reviewer and judge on the labelled outputs, weighted by the chance each output had of
being drawn. Where the two estimates differ by more than their ranges allow, the line is flagged, and the model's
number for that arm should not be trusted yet.

## Limits

`loopmath eval` is new, and these are the things to know before relying on it.

- **The recovery checks do not all pass.** The checks that a fit recovers what a study was simulated from ask for
  200 simulated studies per spec. With the default engine they ran at that count on the 4 specs that test
  interval coverage. One passes (`one-component`); `linear`, `binary-checks` and `levels-3` fail. No class of
  parameters had coverage below the band, but for a few classes (judge bias, arm differences, correlations) 90%
  intervals held the truth in 95% to 98% of studies, against a band of 85% to 95%. On `binary-checks` the hidden
  arm means of most arms came out low. Pass rates computed from the posterior means of the parameters (plug-in
  estimates) were close to the truth on average: over 20 studies, the mean error of each arm and component was
  under 0.02. That does not bound the error in one study, which was often larger. The other specs of the
  catalog are covered in the 0.4.5 entry of the changelog. No real study has been through the group yet.
- **The default engine is an approximation.** It was compared with `--engine pymc` on simulated studies of
  `binary-checks`, `pairs`, `linear` and `one-component`, 20 of each. Plug-in pass rates and reported quality,
  computed from each engine's posterior means of the parameters, agreed to within 0.03 per arm and component.
  The pass rates and reported quality the reports print are means over the draws; how far the engines differ on
  those was not measured. Its hidden arm means on yes or no components came out about 5% below PyMC's on
  average, and its noise and correlation terms differ more. For readings that are not numbers it uses
  expectation propagation (EP). When one group of readings holds every reading of its unknowns, EP gives the
  exact marginal likelihood (the chance of the readings given the noise and correlation terms), up to the error
  of numerical integration; when groups share unknowns, it is an approximation without a bound. Either way, its
  draws of the hidden quality behind such readings are Gaussian approximations. On a design with very little
  call noise EP may not converge, and the fit is then marked as not converged. A study of 20,000 number readings
  fits in about 18 seconds on 2 cores. Readings that are not numbers take longer, and a study of that size with
  such readings has not been timed on this release. To check a fit, fit the same study again with `--engine pymc`.
- **Scores at the ends of a scale.** `eval simulate` clips `linear` and `fraction` readings at the ends of the
  scale, while the fit treats them as unclipped normal readings. What the checks say about recovery holds for
  studies whose readings sit inside the scale. With many readings at 0 or at the top, expect arm means pulled
  toward the middle and noise read too small. A censored likelihood is not built yet.
- **With `--engine pymc`, fits with yes or no components, levels or comparisons are slow.** On 2 cores, with 2
  chains of 500 draws,
  simulated studies of 80 to 120 outputs took 12 to 15 minutes with such components and 1 to 3 minutes with
  linear scales only. A linear study of 500 outputs, 5 components, 4 judges and 2 repeats (20,000 readings) took
  about 30 minutes with 2 chains of 1000 draws, against a target of 10.
- **Priors for two noise terms.** The call noise of levels components and of comparisons has a Gamma prior, not
  the half-normal prior the other noise terms have: with a half-normal prior the sampler diverged.
- **Not built:** `eval fit --halo full` and `--cuts free` are refused by name. A reference answer's family has no
  parameter of its own: it is read as a family effect of the judge that saw the reference.
- **`eval steps` estimates no interactions**: each swap is read alone, against the reference arm.
- **`eval judges` costs** come from the cost per call in `judges.json`, with no range.
- **`eval value --check design`** treats every labelled output as drawn alone: it has no variance term for
  stratified plans and does not use comparisons. Its ranges have been measured through the whole
  chain on one simulated study only: over 25 plans of 30 labels each they held the truth in 90% of cases under
  pilot plans and in 94% under active plans, against a stated 90%. Read a flag from it as a reason to look, not
  as a test.
- **`eval labels next --mode active`** is opt-in; `pilot` is the default. In the simulated comparison, active mode
  had a lower decision loss than pilot and uniform sampling by a small margin. Its scores read yes or no and
  levels components as plain numbers and do not count what a label teaches about a judge. A plan never goes over
  its budget and can come in somewhat under it. Under a value file whose objective is `cheapest_within_margin`
  or `cost_per_pass`, active mode is refused; it runs without `--value`.
- **Blind plans depend on your run ids.** A plan never names an arm, but a run id such as `strong-017` does.
  `eval labels next` refuses to plan when run ids or refs spell out an arm, a model or a model family: give the
  runs neutral ids (`r001`), import the scores under those ids and fit again. `eval simulate` writes neutral run
  ids and refs (`r0001`, `outputs/r0001`) in an order drawn from the seed, so its plans pass this check. A
  simulated study whose run ids still spell out an arm, such as one simulated by loopmath 0.4.4, is let through
  with a note that its plan is not blind.

## JSON

Every command takes `--json` and then prints one JSON object on stdout with a `schema` member, for example
`loopmath.eval-judges-report/1`. A refusal is a `loopmath.error/1` object and exit code 1.
