# Screening harness edits (experimental)

`loopmath screen` is experimental in 0.3.2: it works and has tests, but it has had one real run so far, and its
output may change.

Use it when you have several small, separate edits to a harness (lines of a CLAUDE.md, a hook, a setting) and want
to know which are worth keeping, without testing each one alone. It ranks the edits and suggests a subset. Use
`check-change` when you have one change to adopt or reject: it decides. The usual path is a screen, then a
`check-change` of its best subset against the base on other tasks.

## How it works

- Each edit is a folder laid over the base harness tree: `PATH` copies a file, `PATH.append` appends to one,
  `PATH.remove` removes lines.
- A screen is a set of pairs, one per task. In each pair, arm A runs the task with some edits on and arm B with the
  others on (a fold-over row of a Plackett-Burman design). Every edit is on in one arm and off in the other, so task
  difficulty cancels within the pair, and every pair says something about every edit. Two-edit interactions also
  cancel in a pair's difference, so they do not bias the edits' effects.
- With up to one edit fewer than the pairs, 12, 16, 20, 24 or 32 pairs give orthogonal rows: 8 edits fit in 12 pairs
  (24 sessions).
- `screen status` fits each edit's effect on success and on log cost, under two views: a few edits matter (sparse),
  or all matter a little (dense), weighed by how well each explains the pairs. It ranks the edits by their effect on
  the objective (cost per accepted result by default), with 80% ranges and P(helps).
- The best subset is every edit with a positive expected effect and P(helps) of 0.6 or more. Its prediction is for
  that subset against the base, every edit off, which no arm runs.

## The flow

```
$ loopmath screen start --base harness --edits edits --tasks tasks.jsonl --rule tests
$ loopmath screen next
$ loopmath run record --screen SCR --arm A --workflow WORKFLOW --source designed --since START --verified tests=pass
$ loopmath screen status
$ loopmath screen apply --best --out best
$ loopmath check-change start --base harness --cand best --tasks other-tasks.jsonl --rule tests
```

- `screen start` plans the pairs and builds both arms' trees in the store.
- `screen next` prints the next pair: its task, each arm's edits and tree, and the exact lines that set up a
  worktree for each arm and record its run. Run both arms, record each with `run record --screen SCR --arm A` or
  `--arm B`, and repeat. With several pairs in flight, `screen next --pair P03` shows that pair.
- `screen status` can be read at any time. It says when to stop: every pair recorded, every edit settled at its
  confidence, or the next pair worth less than its price.
- `screen apply --best` writes the base with the best subset to a new folder. `--on EDIT,EDIT` writes another subset.
- Then check that folder against the base on tasks the screen did not use. The screen's ranges guide the choice;
  the check decides it.

## Reading `screen status`

From the example below (12 pairs, every session passed):

```
edits, best first (effect of turning each on; 80% ranges):
   1. a5-no-refactors; cost x0.87 (80% 0.80 to 0.99); success +0.00 (-0.00 to +0.00); P(helps) 0.94
   2. b1-short-tracebacks; cost x0.98 (80% 0.92 to 1.01); success -0.00 (-0.00 to +0.00); P(helps) 0.70
   ...
   8. a4-folder-once; cost x1.01 (80% 0.99 to 1.06); success +0.00 (-0.00 to +0.00); P(helps) 0.36
best: a3-smallest-change, a5-no-refactors, b1-short-tracebacks, b2-read-ranges, b3-batch-reads; predicted cut in
cost per accepted result +18% (80% +2% to +30%), P(better than base) 0.95, never run as an arm
  by view: sparse (weight 0.84) +17% (80% +1% to +28%); dense (weight 0.16) +24% (80% +10% to +36%)
  note: the prediction is against the base (every edit off), which no arm ran: it assumes the edits add up from
  the base, so an effect of having the edited files at all is not in it
  checked: in 1,000 simulated screens of 8 edits on 12 pairs with success tied, 964 picked a best subset, and its
  80% range held the true cut in 812 of them (84%); the other 36 picked no edit, which predicts no change and is
  right by construction; among the picks, the predicted cut ran on average 3.6 points high where the true cut was
  under 10%, and 4.5 points low where it was over 30%
```

- Each edit's line gives the effect of turning it on: its cost ratio and success change, with 80% ranges, and
  P(helps), the chance that it improves the objective. An edit at P(helps) 0.9 or more is named.
- `best` is the subset the screen suggests, often one that no arm ran, and its predicted improvement over the base.
- `by view` gives the same subset's prediction under each view alone, with the view's weight. The dense view runs
  higher. In simulations at the example's size it predicted more than the true cut in every setting tried, and the
  weighed prediction on the `best` line was the better calibrated. In the example the measured cut was nearer the
  dense view; that is one screen.
- `note` says what the prediction leaves out, and `checked` how its range did in simulated screens (see Limits).
- `--json` carries the same numbers: `edits`, `best` (with `by_view`, `range_note` and `range_check`), `next_pair`,
  `stop`.
- `screen status --on EDIT,EDIT` predicts any subset.

## Example

One real screen, 48 sessions in all.

- **The edits.** 8 edits, each one line of a CLAUDE.md: five lines from an existing hand-written file and three new
  rules. The base had no CLAUDE.md.
- **The screen.** 12 pairs on 12 bug-fix tasks from one Python repository (make given failing tests pass at a past
  commit), 24 Claude Code sessions, 3 pairs at a time, about 13 minutes. Every session passed its tests, so success
  could not rank the edits and the ranking was on cost alone.
- **What it said.** One edit was named: a5-no-refactors at P(helps) 0.94, cost x0.87 (80% 0.80 to 0.99). No other
  edit reached 0.9. The best subset was five edits, the named one and four at P(helps) 0.63 to 0.70, predicted to
  cut cost per accepted result by 18% (80% 2% to 30%).
- **The check.** `screen apply --best`, then `check-change` of that subset against no CLAUDE.md on 12 other tasks
  from the same repository: 24 more sessions, about 13 minutes. It measured a 33% cut (34% on total cost, $7.23 to
  $4.80), with 11 of 12 tasks passing in each arm, and adopted at P(better) 0.993.
- **Cost.** $23.09 list-equivalent for the 48 sessions: what their tokens cost at list prices. They ran on a
  subscription.
- **The miss.** The measured 33% is above the predicted 80% range of 2% to 30%, and still above it after widening
  the range for the check's own noise (to 1% to 31%). The screen's pick was right; its range was low.

**What the miss means.** The screen's analysis was then run on simulated screens of the example's size (8 edits,
12 pairs, cost noise as measured, every session passing), with true effects drawn from settings no tuning used. Of
1,000 of them, 964 picked a best subset. Its 80% range held the true cut in 812 of those (84%), and a 12-pair check
of the pick, like the one above, landed above its widened range in 9% of them. The other 36 picked no edit, which
predicts no change and is right by construction. So either the example drew a check that comes about 1 time in 12,
or the check measured something the screen cannot see: an effect of having a CLAUDE.md at all (see Limits). If
having the file at all cut cost by 14%, a check would land above the range in 40% of the screens with a pick. A
check of a neutral one-line CLAUDE.md against none tells the two apart.

## Limits

- **The range.** In the simulated screens above, the 80% range was about right: of the 964 of 1,000 that picked a
  subset, it held the true cut in 812 (84%), and the predicted cut was 0.7 points low on average; the other 36
  picked no edit, which predicts no change and is right by construction. Among the picks it leans toward the
  middle: 3.6 points high on average where the true cut was under 10%, and 4.5 points low where it was over 30%
  (there the range held the truth in 76%). In the settings the model was tuned on, when no edit mattered at all the
  screen still picked a best subset 82% of the time, predicting a cut of about 8 points, and its range left out 0
  in 13% of those: the best of eight noisy edits looks good. These hold for tied success and cost noise like the
  example's; with each session passing at random 85% of the time, the range held the truth in 92% of the screens
  with a pick. `screen status` prints the first of these as `checked`. Seven changes to the model (one view alone,
  other priors, fixed view weights) were scored on the same simulated screens against a rule fixed in advance (a
  better interval score overall and in no setting worse; coverage, false namings and the pick's loss against the
  best subset no worse beyond small margins). None passed, so the model is unchanged.
- **Having the file at all.** No arm runs the bare base: every arm has some edits on. So the screen cannot see an
  effect of having the edited files at all (for a CLAUDE.md: any CLAUDE.md, whatever it says). It cancels in every
  pair. A check against the bare base measures it together with the edits, so a check can beat the screen's
  prediction for that reason alone. To measure it, check a neutral file (one plain line, no instruction) against
  none.
- **Ties on success.** When every pair ties on success (all pass, or all fail), success cannot rank the edits and the
  ranking is on cost alone. `screen status` says so. On tasks that always pass, rank on cost; on tasks with a mix,
  success counts.
- **12 pairs.** With 8 edits in 12 pairs, each edit's effect is measured with an error of about 0.29 times the pair
  noise (in the example, about 0.07 on log cost), and only 4 pairs are left to measure the noise itself. A real
  effect of a few percent of cost rarely reaches P(helps) 0.9 there; it enters the best subset on the 0.6 rule, or
  not at all. 24 pairs bring each effect's error down to 0.2 times the pair noise.
- **Adding up.** A subset's prediction adds its edits' effects. The pairs keep each edit's effect clear of two-edit
  interactions, but a subset no arm ran is a prediction, which is why the check comes after.
