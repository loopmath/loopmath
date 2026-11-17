# Record runs and judge outputs for loopmath eval

The user runs an AI system in several configurations and scores its outputs with model judges and some human reviewers. They want `loopmath eval` to read those scores. `loopmath eval` runs no model and calls no judge: it reads files. Your job is to make the user's own code write those files, one line per output and one line per score, and to prove with the check that the files are right. Stop there. What comes after a checked study (the fit, the judges table, the value, the reviewer plans) is in `loopmath guide eval`.

`loopmath eval` is experimental in this version: the file formats and the numbers may change. Say that to the user once, before you change their code.

Use exactly the commands below. Their flags are in `loopmath guide eval`. Never run `--help`.

## 1. Look, then ask once

Look through the code before you ask anything. Find where an output is produced, where a judge is called, what the judge returns, and where scores are kept today.

Then ask the user, in one message, only what the code does not tell you:

- Which settings make a configuration: model, effort, prompt version, tools? Each fixed combination is one arm. What is each arm called?
- What does each judge score, and on which scale: yes or no, named levels, a number range, a share from 0 to 1, or a list of checks?
- Which judges are people and which are models? For a model: its id, its family, the effort and the prompt version.
- How is each judge asked: does one request score one criterion, or all of them? Ask this for people too: does a reviewer go through all outputs one criterion at a time, or score every criterion of an output in one sitting?
- Do reviewers see which configuration wrote an output? They should not.
- What does one run cost and what does one judge call cost, when known?
- Are there scores already, in a sheet or in another tool?
- Is any of the data made up: a dry run, test data, outputs of a stand-in system?

Never guess a scale, an arm or a cost. A member you have no value for stays out of the line. Never write `null` in its place: some members refuse it.

## 2. Write the rubric and the judges

`STUDY` below is the study folder. Use the folder the user names, or `eval-study` in the project.

Write `STUDY/rubric.json`: one entry per thing scored.

```json
{
 "schema": "loopmath.eval-rubric/1",
 "id": "answer",
 "version": "1",
 "components": [
  {"id": "correct", "title": "The facts are right", "scale": {"kind": "binary"}},
  {"id": "clarity", "title": "Clear to a new reader", "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]}},
  {"id": "depth", "title": "Depth", "scale": {"kind": "linear", "range": [0, 10]}},
  {"id": "covers", "title": "Covers the required points", "scale": {"kind": "items"}},
  {"id": "overall", "title": "Overall", "role": "overall", "scale": {"kind": "linear", "range": [0, 10]}}
 ]
}
```

- `scale.kind` is `binary`, `levels` (3 or more, lowest first), `linear` (with `range`), `fraction` (0 to 1) or `items` (a list of checks per input).
- Add `"better": "lower"` to a component where less is better, such as a count of errors.
- Give the overall judgement the role `overall`. Comparisons that name no component are read as comparisons of it.
- A pipeline with scored stages lists them in `steps`, ending with `final`, and each component names its stages in `applies_to`.
- Change `version` when a wording or a scale changes.

Write `STUDY/judges.json`: one entry per judge.

```json
{
 "schema": "loopmath.eval-judges/1",
 "anchor": ["reviewer-1", "reviewer-2"],
 "judges": [
  {"id": "reviewer-1", "kind": "person", "blinded": true},
  {"id": "reviewer-2", "kind": "person", "mode": "panel", "blinded": true},
  {"id": "judge-main", "kind": "model", "model": {"id": "model-large", "family": "alpha"}, "effort": "high",
   "prompt": "rubric-prompt-v2", "mode": "per_component", "blinded": true, "cost_usd_per_call": 0.03},
  {"id": "judge-cheap", "kind": "model", "model": {"id": "model-small", "family": "beta"}, "effort": "low",
   "prompt": "rubric-prompt-v2", "mode": "panel", "blinded": true, "cost_usd_per_call": 0.004,
   "components": ["correct", "clarity", "depth", "overall"]}
 ]
}
```

- A judge is one model with one prompt, one effort and one mode. When any of those changes, add a new judge with a new id. Never change an entry that already has scores.
- `mode` is `per_component` when one request scores one component and `panel` when one request scores all of them. `per_component` is the default.
- A person has a mode too. A reviewer who goes through the outputs one component at a time is `per_component` (`reviewer-1`). A reviewer who scores every component of an output in one sitting is `panel` (`reviewer-2`).
- Each reviewer is a judge of kind `person` with their own id. Name the reviewers in `anchor`.
- `cost_usd_per_call` is the cost of one call of that judge. Leave it out when unknown.
- `components` lists what the judge scores when it is not everything.
- Ids of judges, components and families hold no space, no `/` and no `@`.

A rubric with an `items` component needs `STUDY/items.jsonl`: one line per check of each input.

```json
{"schema": "loopmath.eval-items/1", "id": "q01.deadline", "input": "q01", "component": "covers", "label": "States the deadline"}
{"schema": "loopmath.eval-items/1", "id": "q01.owner", "input": "q01", "component": "covers", "label": "Names the owner"}
{"schema": "loopmath.eval-items/1", "id": "q01.cost", "input": "q01", "component": "covers", "label": "Gives the cost"}
```

Check the files before anything is scored. Leave `--items` out when there is no `items` component:

```sh
loopmath eval rubric check STUDY/rubric.json --judges STUDY/judges.json --items STUDY/items.jsonl --json
```

Exit code 0 and `ok` true is a pass. Show the user each entry of `warnings`. On exit 1, fix what the `error:` line on stderr names and run it again.

## 3. Write one line per output

In the place where the user's system finishes an output, append one line to `STUDY/runs.jsonl`:

```json
{"schema": "loopmath.eval-runs/1", "run": "r001", "input": "q01", "arm": "baseline", "steps": [{"step": "final", "model": {"id": "model-small", "family": "beta"}, "effort": "low", "variant": "prompt-a"}], "cost_usd": 0.011, "simulated": true, "refs": {"input": "inputs/q01.md", "output": "outputs/r001.md"}}
{"schema": "loopmath.eval-runs/1", "run": "r002", "input": "q01", "arm": "candidate", "steps": [{"step": "final", "model": {"id": "model-large", "family": "alpha"}, "effort": "high", "variant": "prompt-a"}], "cost_usd": 0.094, "simulated": true, "refs": {"input": "inputs/q01.md", "output": "outputs/r002.md"}}
```

- These two example runs are made up, so they carry `"simulated": true`. Leave `simulated` out of every line a real system writes. Write it only for made-up runs. A study with such runs is fitted only with `--allow-simulated`, so made-up scores cannot pass for real ones.
- `run` is a new id for every output. Keep it neutral: `r001`, never the arm, the model or the family in the id or in `refs`. Hand ids out in shuffled order, so that the number does not give the arm away. A reviewer plan is refused later when a run id or a ref names the configuration.
- `input` is the same id for every arm that answers the same input. Run every arm on the same inputs.
- `arm` is the configuration. All runs of one arm have the same `steps`.
- `steps` has one entry per stage: the stage name, the model with `id` and `family`, and `effort` and `variant` when they apply. Use the same family names as in the judges file.
- `cost_usd` is what this output cost. Write it now: it is hard to rebuild later.
- A member the format does not know is refused. Anything of the user's own goes under `ext`.

## 4. Write one line per score

In the place where a judge answers, append one line per score to `STUDY/scores.jsonl`. Keep the id of the request to the judge: the request id of the API, or a new UUID made when the request is sent.

```json
{"schema": "loopmath.eval-scores/1", "id": "c0001.correct", "run": "r001", "component": "correct", "judge": "judge-main", "call": "c0001", "value": 1, "rubric": "answer@1", "observed_at": "2026-03-02T09:00:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0002.depth", "run": "r001", "component": "depth", "judge": "judge-main", "call": "c0002", "value": 6.5, "rubric": "answer@1", "observed_at": "2026-03-02T09:00:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0003.depth", "run": "r001", "component": "depth", "judge": "judge-main", "call": "c0003", "value": 7.0, "rubric": "answer@1", "observed_at": "2026-03-02T09:01:00+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0004.correct", "run": "r002", "component": "correct", "judge": "judge-cheap", "call": "c0004", "value": 1, "rubric": "answer@1", "observed_at": "2026-03-02T09:01:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0004.clarity", "run": "r002", "component": "clarity", "judge": "judge-cheap", "call": "c0004", "value": "good", "rubric": "answer@1", "observed_at": "2026-03-02T09:01:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0004.depth", "run": "r002", "component": "depth", "judge": "judge-cheap", "call": "c0004", "value": 0.8, "rubric": "answer@1", "observed_at": "2026-03-02T09:01:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0004.overall", "run": "r002", "component": "overall", "judge": "judge-cheap", "call": "c0004", "value": 7.5, "rubric": "answer@1", "observed_at": "2026-03-02T09:01:20+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0004.depth.fix1", "run": "r002", "component": "depth", "judge": "judge-cheap", "call": "c0004", "value": 8.0, "rubric": "answer@1", "observed_at": "2026-03-02T09:01:20+00:00", "replaces": "c0004.depth"}
{"schema": "loopmath.eval-scores/1", "id": "c0005.covers.q01.deadline", "run": "r001", "component": "covers", "judge": "judge-main", "call": "c0005", "value": 1, "item": "q01.deadline", "rubric": "answer@1", "observed_at": "2026-03-02T09:01:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0005.covers.q01.owner", "run": "r001", "component": "covers", "judge": "judge-main", "call": "c0005", "value": 0, "item": "q01.owner", "rubric": "answer@1", "observed_at": "2026-03-02T09:01:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "c0005.covers.q01.cost", "run": "r001", "component": "covers", "judge": "judge-main", "call": "c0005", "value": 1, "item": "q01.cost", "rubric": "answer@1", "observed_at": "2026-03-02T09:01:40+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "h0001.depth", "run": "r002", "component": "depth", "judge": "reviewer-1", "call": "h0001", "value": 3.0, "rubric": "answer@1", "observed_at": "2026-03-02T14:10:00+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "h0001.depth.fix1", "run": "r002", "component": "depth", "judge": "reviewer-1", "call": "h0001", "value": 8.0, "rubric": "answer@1", "observed_at": "2026-03-02T14:12:00+00:00", "replaces": "h0001.depth"}
{"schema": "loopmath.eval-scores/1", "id": "h0002.clarity", "run": "r001", "component": "clarity", "judge": "reviewer-2", "call": "h0002", "value": "ok", "rubric": "answer@1", "observed_at": "2026-03-02T15:00:00+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "h0002.depth", "run": "r001", "component": "depth", "judge": "reviewer-2", "call": "h0002", "value": 6.0, "rubric": "answer@1", "observed_at": "2026-03-02T15:00:00+00:00"}
{"schema": "loopmath.eval-scores/1", "id": "h0002.overall", "run": "r001", "component": "overall", "judge": "reviewer-2", "call": "h0002", "value": 6.5, "rubric": "answer@1", "observed_at": "2026-03-02T15:00:00+00:00"}
```

- One line is one score: one judge, one run, one component, and one check item when the component is a list of checks.
- `call` is the id of the request. One request gets one call id, and every line written from its answer carries it. There is no exception.
  - A `per_component` judge answers with one score, so the call has one line (`c0001`, `c0002`).
  - A `panel` judge answers with one score per component, so the call has one line per component (`c0004`). The same holds for a person who scores an output in one sitting (`h0002`).
  - A list of checks answered in one request has one line per check, each with `item` and a `value` of 0 or 1, all under one call id (`c0005`). This holds for a `per_component` judge too. `c0005` is one reading, 2 of 3 passed. Three call ids would pass the check and be fitted as three requests of one check each, which is wrong. A judge that only gives the number passed writes one line with the count in `value` and the total in `of`, and no `item`.
- `id` is the id of the line, not of the call. No two lines share one. Build it from the call, the component and the item (`c0004.depth`, `c0005.covers.q01.owner`). Then the same score always gets the same id.
- A repeat is the same judge asked again about the same output and component. It is a new request: a new line with a new `call` (`c0003` after `c0002`). Never overwrite and never average. Repeat a panel request as a whole, under one new call id.
- A correction fixes a line that was stored wrong: a typing slip, a parser that misread the answer. The judge was not asked again. Write a new line with its own `id`, the same `call` as the wrong line, and the wrong line's id in `replaces` (`c0004.depth.fix1`, `h0001.depth.fix1`). Keep the wrong line. A correction under a new call id passes the check and is wrong: a panel answer then counts as two requests.
- Without `replaces`, a second line is a repeat. Nothing is taken as a correction because it is newer.
- `value` is what the judge said, on the rubric's scale: `1` or `0` for `binary` (not true, false or a word); a level exactly as the rubric spells it for `levels`, with a number level as a whole number; a number inside the range for `linear`; 0 to 1 for `fraction`.
- Do not round, rescale or turn words into numbers. If a judge answers on another number range, convert in one place and keep the judge's own number under `ext`.
- `rubric` is the rubric id, `@` and its version. `observed_at` is the time of the answer with an offset.
- `step` names the stage scored when it is not `final`.
- If several processes judge at once, let each write its own file and join the files afterwards.

When the user's code has one function that writes a score line, give it the call id and the line id as two separate arguments. A function that derives the line id from the call alone cannot write a correction.

## 5. Write one line per comparison

Only when a judge compares two outputs. Append to `STUDY/pairs.jsonl`:

```json
{"schema": "loopmath.eval-pairs/1", "id": "c0006", "a": "r002", "b": "r001", "judge": "judge-main", "call": "c0006", "shown_first": "a", "verdict": "tie", "rubric": "answer@1", "observed_at": "2026-03-02T09:02:00+00:00"}
{"schema": "loopmath.eval-pairs/1", "id": "h0003.depth", "a": "r001", "b": "r002", "component": "depth", "judge": "reviewer-2", "call": "h0003", "shown_first": "b", "verdict": "b", "rubric": "answer@1", "observed_at": "2026-03-02T15:10:00+00:00"}
{"schema": "loopmath.eval-pairs/1", "id": "h0003.overall", "a": "r001", "b": "r002", "component": "overall", "judge": "reviewer-2", "call": "h0003", "shown_first": "b", "verdict": "b", "rubric": "answer@1", "observed_at": "2026-03-02T15:10:00+00:00"}
```

- `a` and `b` are two runs of the same input. Pick at random which run is `a`.
- `verdict` is `a`, `b` or `tie`. `shown_first` is the one the judge saw first. Always write it.
- The same pair asked again in the other order is a second call and a second line.
- A request that compares on several components writes one line per component under one call id. The line id is then the call and the component (`h0003.depth`).
- A correction works as for scores: its own `id`, the same `call`, and `replaces`.

## 6. Bring in scores that already exist

Scores in a sheet: save it as CSV with the columns `run`, `component`, `judge`, `value`, and when known `call`, `step`, `item`, `observed_at`, `id`, `replaces`. Comparisons need `a`, `b`, `judge` and `verdict`, or `choice` (`left`, `right`, `tie`) with `shown_first`.

```sh
loopmath eval scores import --csv FILE --rubric STUDY/rubric.json --out STUDY --json
loopmath eval scores import --pairs-csv FILE --rubric STUDY/rubric.json --out STUDY --json
```

Runs kept as OCP documents with score signals:

```sh
loopmath eval scores import --ocp FILE --rubric STUDY/rubric.json --out STUDY --json
```

A run that is made up is refused by this import unless the command has `--allow-simulated`. Add the flag only when the user says the runs are made up. Never add it to get past a refusal on real data: tell the user which runs carry the mark.

Take `readings.new`, `readings.unchanged`, `readings.refused`, the same three for `runs` and `pairs`, `refused[]`, `not_used[]` and, when it is there, `ignored_columns`. When a row is refused nothing is written: show the refused rows and fix the source. Importing the same file again changes nothing.

Three traps, all about `call`:

- A CSV row without a `call` column is its own call. For a `panel` judge, and for a reviewer who scores in sittings, add the column and give the rows of one request the same value. The same goes for a filled reviewer form: it has no `call` column.
- A sheet without an `id` column gets ids made from the content of each row. A value changed in the sheet then comes in as a second score beside the old one, a repeat. For a correction, add a row with its own `id`, the `call` of the wrong row and `replaces`.
- An OCP score signal without the `dev.loopmath.eval` member under `ext` has no call. Two such signals from one judge on one output are then read as a score and its correction, not as two repeats. When they are repeats, write `call` in that member.

## 7. Check the study

Run this after the first few outputs are scored, not after the whole batch:

```sh
loopmath eval scores check STUDY --rubric STUDY/rubric.json --judges STUDY/judges.json --json
```

Always pass `--judges`: without it a misspelt judge passes. Exit code 0 and `ok` true is a pass. Take `counts` (`runs`, `readings`, `readings_read`, `pairs`, `pairs_read`, `items`), `problems[]` (`message`, `code`), `warnings[]`, `mixed_arm[]` and `design[]`. When a line is not a valid record at all, the command prints `error` and no `problems[]`: fix that line first.

Fix the cause in the code that writes the lines, not the lines by hand, then write the files again and check again.

| It prints | Do |
|---|---|
| `two readings of one call for the component` | A repeat needs its own `call`. A correction needs `replaces` and keeps the call. |
| `is already on line` | The same line was written twice. Write each score once. |
| `missing member` | Add the member named: most often `call`, `id`, `rubric` or `observed_at`. |
| `is not an ISO 8601 time with an offset` | Write the time with its offset, such as `+00:00`. |
| `is not 0 or 1` | A `binary` score is the number 1 or 0. |
| `is not a level of` | Write the level as the rubric spells it. A number level is a whole number, not text. |
| `is outside the scale of` | The value is on another scale, such as a percentage. Convert where the score is recorded. |
| `unknown judge` | Add the judge to `judges.json`, or fix the id. |
| `unknown component` | Spell the component as the rubric does. Case counts. |
| `in runs.jsonl; import the run` | Write the run's line before its scores. |
| `unknown rubric version` | The score names another version than the rubric file has. Check it with the version it was scored under. |
| `unknown member` | Move the member under `ext`. |
| `have different inputs` | A comparison is between two runs of one input. |
| `names no reading` | `replaces` must name the id of a line in the file. |
| `a reading replaces itself` | A correction needs an `id` of its own. |
| `the call has item lines and a count line` | Per call, write one line per check or one count line, not both. |

The check passes some mistakes. Look for them yourself:

- A value on the wrong scale that is still inside the range, such as 0.7 under a 0 to 10 component. Compare the lowest and highest value per judge and component with the rubric.
- A call structure that is not what was asked: a `panel` judge or a sitting whose lines each have their own `call`, the checks of one request under a `call` each, a correction under a new `call`. Count the distinct `call` values per judge and compare with the number of requests made.
- Made-up runs with no `simulated` mark.
- A run id or a ref that names the arm, the model or the family.
- An entry in `mixed_arm`: an arm whose runs differ in a setting. Split it into two arms or fix the run lines.
- An entry in `warnings` about a call shared by two judges or two outputs.

## 8. Tell the user what the design can measure

Each row of `design` has `term`, `judge`, `dimension`, `count`, `met` and `needs`. For every row with `met` false, tell the user what `needs` says, in their words. The usual gaps:

- No repeats: ask for a second call on a share of the outputs, picked at random. Start with the number scales.
- One judge only, or judges that each score one arm: every judge should score outputs of every arm.
- No reviewer scores, or fewer than 6 outputs per component: ask for a small batch scored by a person. 30 makes a firm anchor.
- Comparisons in one order only: ask each pair in both orders.
- No costs: `cost_usd` on runs and `cost_usd_per_call` on judges.

Do not start a fit, install anything or call a judge to fill a gap. The user decides what to spend. When the study holds runs marked `simulated`, say so: the fit will refuse it without `--allow-simulated`.

## 9. Summary

End with at most 6 lines, then stop:

> Recording is in place: `src/pipeline.py` writes runs, `src/judge.py` writes scores, to `eval-study/`.
> Check passed: 32 runs, 378 readings (376 read), 22 pairs, 3 judges.
> The design does not measure yet: call noise of judge-main on correct (no repeats), a firm anchor (12 of 30 reviewed outputs).
> Not recorded: the cost of a judge call for judge-cheap.
> `loopmath eval` is experimental: formats may change.
> Next: fit and read the judges ("fit this study").

## Never

- Do not write a score without the id of its call, and do not give the lines of one answer different call ids.
- Do not give a correction a new call id.
- Do not write `null` for a value you do not have, and do not mark a real run `simulated`.
- Do not overwrite, average or drop a repeat.
- Do not put the arm, the model or the family in a run id or a ref.
- Do not change a judge's entry after it has scores: add a new judge.
- Do not guess a scale, a cost, a model family or an arm.
- Do not edit `fit.json` or a labels plan, and do not fit before the check passes.
- Do not fix refused lines by hand in the study folder when the code that writes them is still wrong.
