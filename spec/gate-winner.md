# `loopmath gate winner`: the selection-adjusted gain of a search's winner

A search loop scores N candidates and keeps the best. The best of N noisy scores is high partly because its
noise was high, so the winner's lead over its parent overstates its gain: the winner's curse. `gate winner`
reads every score the search made, shrinks the winner's lead by what the scores say about the candidates'
spread and the noise, and reports the winner's gain with 80% and 95% ranges and P(beats parent). Then it asks
`gate` what to run before adopting the winner, on confirmatory runs only: the search's own scores never adopt.

```
loopmath gate winner --scores search.csv --spread 0.02
loopmath gate winner --scores search.json --confirm confirm.jsonl --json
loopmath gate winner --runs runs.jsonl --parent arm:base
```

## 1. The estimand and its units

- The gain is D = theta_w - theta_p: the winner's true mean score minus the parent's, on the evaluation the
  search used. A true mean is the score's average over reruns of that evaluation: the same tasks, the same
  scorer.
- One evaluation is one complete pass of one arm over the evaluation's tasks, and its score is the task mean.
- The spread is the SD of one evaluation across reruns, in the score's units. A score that averages `reps`
  evaluations has standard error spread / sqrt(reps).
- The task set is fixed. Task-sampling variance (how a score would move on other tasks from the same
  benchmark) is not in the estimand. A published test-set lead is a different quantity: a fresh-task
  comparison, not a rerun of the search's evaluation.

## 2. Inputs

Exactly one of `--scores FILE` or `--runs FILE`.

### 2.1 Scores, CSV

A header row, then one row per scored arm, or per group of unscored candidates.

| Column | Meaning |
|---|---|
| `id` (or `candidate`) | The arm's id, unique in the file. |
| `score` | Its score. Empty on a censored row. |
| `se` | Optional: the score's standard error. Rows without one take spread / sqrt(reps). |
| `reps` | Optional: the evaluations averaged into the score (default 1). |
| `order` | Optional: the search's order, kept for the reader. |
| `role` | `parent` (also `seed` or `base`) on exactly one row; `candidate` or empty on the rest. |
| `at_most`, `at_least` | A censored row: `count` candidates whose scores are unknown, each no better than this bound (`at_most` when higher is better, `at_least` when lower is). |
| `count` | A censored row's number of candidates (default 1). |

### 2.2 Scores, JSON (`loopmath.gate-winner-scores/0.1`)

```json
{"schema": "loopmath.gate-winner-scores/0.1",
 "better": "higher",
 "scale": "fraction",
 "spread": [0.015, 0.035],
 "parent": {"id": "seed", "score": 0.7247},
 "candidates": [{"id": "gen1", "score": 0.7753, "order": 1}, {"id": "gen3", "score": 0.7865, "order": 3}],
 "censored": [{"count": 1, "at_most": 0.7753}, {"count": 12, "at_most": 0.7865}],
 "source": {"cite": "arXiv 2609.14858v2", "where": "Sec. 4.4, Fig. 5 (a)"}}
```

- `better`: `higher` (default) or `lower`.
- `scale`: `linear` (default), `log` or `fraction`. On `fraction`, scores lie in [0, 1] and the projection
  draws bounded scores (section 5).
- `spread`: optional, one number or `[lo, hi]`. `--spread` overrides it.
- `parent` and each candidate: `id`, `score`, optional `se`, `reps` and `order`.
- `censored`: groups with `count` and a bound, optional `se`, `reps` and `order`.
- `source`: optional, any object; it is copied to the output.

A best-so-far curve is a search printed this way: each new best is a scored candidate, and the candidates
between two bests are a censored group bounded by the best so far. A table that prints only the best of N is
one scored candidate and one censored group of N - 1.

### 2.3 Runs (`loopmath.harness-runs/0.1`)

`--runs FILE` reads harness-runs rows (see [harness-runs.md](harness-runs.md)). Each arm is a candidate,
grouped by harness digest, else `arm`, else configuration. `--parent SEL` names the parent arm, in `gate`'s
selector forms (default `arm:base`). The rule is `--rule score:NAME`, or `score:NAME:lower`.

- Every arm must have scored runs on the same tasks; otherwise the scores would differ by task difficulty,
  and the command names the arm and the missing tasks.
- An arm's score is the mean of its task means. Its standard error is sqrt(sum_t sigma_t^2 / r_t) / T, where
  T is the number of tasks, r_t the arm's runs on task t, and sigma_t^2 the replicate variance on task t,
  pooled over the arms with two or more runs there. A task with no repeat takes the pool over all tasks.
- One evaluation's SD is sqrt(sum_t sigma_t^2) / T.
- With no task run twice by any arm, the spread is unknown: give `--spread`, one run's SD on one task.

## 3. The model

Each score is y_i = theta_i + e_i, with e_i ~ N(0, s_i^2) and s_i its standard error; the parent's is
y_p = theta_p + e_p. The candidates' true scores are theta_i = theta_p + mu + eta_i: a mean child effect
mu ~ N(0, s_mu^2) shared by the search, plus each candidate's own deviation eta_i. The deviations are
independent under one of two views, as `loopmath screen` has them:

- `sparse` (the default, `--prior`): eta_i ~ (1 - pi) N(0, (0.05 tau)^2) + pi N(0, tau^2), with pi in
  {0.1, 0.25, 0.5, 0.75, 0.9} under a uniform prior: most candidates near the centre and a share spread out.
  At pi = 0.9 it is close to the ridge view;
- `ridge`: eta_i ~ N(0, tau^2);
- `auto`: both views, weighed by their marginal likelihoods.

The default was chosen on simulated searches by a rule fixed in advance: sparse and auto both averaged
nominal coverage, and sparse was within the coverage bands in one more kind of search, with a slightly
lower interval score.

tau has a half-Cauchy(0, tau_prior) prior on a log grid; tau_prior and s_mu are 0.1 on fractions and 3 times
the median standard error on other scales. Neither tau nor pi is plugged in: both are integrated over, so the
ranges carry their uncertainty. Censored candidates enter through the normal CDF, count x log P(score below
the bound). Given nu = theta_p + mu, the candidates are independent and theta_p integrates out in closed
form. nu is integrated on a grid that zooms until it resolves the posterior, and tau's grid zooms until its
step in log tau is within the posterior SD of log tau. The gain's posterior is a finite mixture of normals, with
exact means and tail probabilities.

Under an exchangeable prior, the posterior of a candidate picked by any rule from these scores is the same as
for one named in advance (Dawid 1994, "Selection paradoxes of Bayesian inference"). So the shrinkage is the
winner's-curse correction, provided that every score the selection read is in the input. A search that
scored 64 candidates and printed one is described honestly only with the other 63 as a censored group.

`--winner ID` assesses a named candidate, for a loop that picked by some other rule.

P(beats parent) is the model's probability that D > 0, given its prior on the shared offset mu. In
simulated searches drawn from a view's own prior (the default sparse view, and auto), it was calibrated: in
every bin, the share that beat the parent was within about two binomial SEs of the probability given. That
checks the computation under its prior, not how often real searches beat their parents. In simulated
searches whose true offset was fixed at 0, it was
too sure on both sides: winners given about 0.85 beat the parent about 65% of the time, and winners given
about 0.06 about 24%. When the true gain is near 0, the parent score's luck alone moves P(beats parent) across
that whole span. The ranges do not lean on the prior for mu in the same way, so read it with them.

Two reasons name regimes where the ranges ran narrow in simulated searches, each with the coverage seen there:
a parent whose se is at most half the median candidate se (the range then rests on the model's shape for the
candidates), and, with 32 or more scored candidates and none censored, scores with heavier tails than a normal
spread (excess kurtosis of the scores, each centred and scaled by its moment-estimated spread plus its own se,
above 2 sqrt(24 / N)). They are text: no number depends on them. Neither is a test, and their absence is no
all-clear: with a few candidates far better than the rest, the searches where the tails reason did not show
covered worse (80% ranges held 52% to 62% of the time at 64 candidates) than those where it did.

## 4. The gate

`--confirm FILE` gives confirmatory runs of the parent and the winner as harness-runs rows, selected by
`--confirm-base` (default `arm:base`) and `--confirm-cand` (default `arm:candidate`). The gate block is
`gate.decide(base, cand, policy)` on those runs, exactly as `loopmath gate` gives it, with `--rule`,
`--floor`, `--r-min`, `--r-max` and `--budget-usd`. With no confirmatory runs it asks for its first pair.

The next pair's value stop is on, as `loopmath gate` has it by default. `default_policy()` states it in the
policy rather than leaving it to the gate's default, so the action does not depend on that default; a gate
with no such setting always stops on value. A `GatePolicy` passed to `assess` is used as given.

- With scores input, each confirmatory run is one complete evaluation of its arm.
- `gain_confirmed` is the gain's posterior after the held confirmatory pairs, each normal part updated by the
  pairs' mean difference taken as normal with the spread's variance (an approximation on bounded scores).
  The search-side posterior `gain` is unchanged.

## 5. The projection

The projection says what the gate would do from its current state if the posterior is right: P(adopt), the
expected number of further pairs and its 80th percentile, and the dollars when the runs are priced. It draws
D from the posterior (`gain_confirmed` when pairs are held) and runs the real `gate.decide` on fresh
evaluations, `--draws` times (default 200; 0 for none). It assumes:

- each projected pair is one fresh evaluation per arm, independent, with the spread as its SD;
- the base arm's mean is held at the parent's posterior mean; its uncertainty and its dependence with the
  gain are left out;
- held confirmatory pairs update the gain as a normal mean with the spread's variance;
- a projected run costs its arm's mean over the held confirmatory runs (else over its search runs), when
  those are priced. The gate prices each pair and stops on value from these costs, budget or not;
- with a spread range (section 6), each projected evaluation's SD is the range's spreads averaged by their
  weights, not a spread drawn together with the gain.

On `fraction` scores, the projection draws Beta evaluations with the asked mean and SD, never clipped
normals. A drawn gain that puts a mean at or past 0 or 1, or that needs a spread wider than a fraction allows
there, is infeasible. Such draws are left out, the output says `conditioned_on: "feasible draws"` and gives
`infeasible_share`, and above 2% the projection is withheld with its reason. It is also withheld for an
acceptance rule, for task-level confirmatory runs with `--runs` input, and for a budget when an arm's cost is
unknown. The gate's own action always stands.

Like P(beats parent), the projection is a forecast under the model's prior, not a frequency for one search.
In simulated searches whose true scores were fixed, it was too sure before any confirmatory pair. With
normally spread gains, P(adopt) ran 0.08 to 0.10 above the share the real gate adopted, and the expected
pairs 0.6 to 0.9 short; with no candidate better than the parent, it gave 0.22 where the gate adopted 0.5% of
the time. Held pairs shrank the gaps: after two, P(adopt) was within 0.03 of the gate's share with normally
spread gains, but 0.09 high with a floor of half the spread, and 0.07 against 0.002 with no better candidate.
Near the bound of a fraction (true means about 0.97), 37% to 81% of projections were withheld as infeasible.
The 80th percentile of further pairs was almost always the runs left under `--r-max`; in the few searches
where it was lower, the gate ran past it more often than not.

`buy` sums it up: `now` (the gate's next request), `at_least` (pairs still needed before any adoption),
`expected` (the projection's mean), and `at_most` (runs left under `--r-max`). All four are 0 once the gate
has stopped.

## 6. Spread ranges

`--spread LO:HI` (or `"spread": [lo, hi]`) states a range when the spread was never measured. The gain
integrates over 7 spreads, log-uniform across the range, each weighed by how well it fits the scores, and
`by_spread` shows the gain and P(beats parent) at 5 of them. The output's reasons say the range is assumed.

## 7. Output

Text: one headline (the winner, its lead, the number of candidates, the gain and its ranges), then P(beats
parent), the gate line, the buy line, the `by_spread` rows and the reasons. `--json` prints one
`loopmath.gate-winner/1` object:

| Field | Meaning |
|---|---|
| `winner`, `parent`, `better`, `scale` | As read. |
| `n_candidates` | `scored`, `censored` and `total`. |
| `observed_lead` | The winner's score minus the parent's, in the better direction. |
| `gain` | `mean`, `median`, `lo` and `hi` (the 80% range), `bands` with `80` and `95`. |
| `p_better`, `p_above_floor`, `floor` | P(D > 0), and P(D > floor) when a floor is set. |
| `shrinkage` | 1 - gain mean / observed lead, when the lead is positive. |
| `gain_confirmed` | As `gain`, after the held pairs, with `pairs` and `p_better`; null with none. |
| `spread` | `value` and `source` (`given`, `runs` or `se`), or `range` and `weights`. |
| `by_spread` | One row per shown spread, with a range. |
| `prior` | The view, the views' weights, the posterior means of `tau`, `pi` (sparse view), `mu` and `theta_p` (the parent's true score), the hyperprior's `tau_prior` and `s_mu`, and the grid sizes. |
| `gate` | `gate.decide(...).to_json()` on the confirmatory runs. |
| `projection` | `p_adopt`, `p_adopt_if_real`, `p_adopt_if_not`, `mean_more`, `p80_more`, `usd_more`, `infeasible_share`, `conditioned_on`, `assumes`; or `withheld` with a `reason`. |
| `buy` | `now`, `at_least`, `expected`, `at_most`. |
| `reasons`, `inputs`, `source` | Plain-language reasons, the parent's and winner's scores and errors, and the file's `source`. |

Exit codes: 0 on an answer; 1 when an input is refused (the message names the file, line or field); 2 for a
missing file, a usage error, or scores with no standard error and no spread.

## 8. Python

```python
from loopmath.gate import winner

search = winner.read_scores("search.csv")          # or winner.search_from_runs(rows, parent="arm:base")
v = winner.assess(search, spread=0.02, policy=winner.default_policy(floor=0.01), confirm=(base_runs, cand_runs))
v.gain["bands"]["80"], v.p_better, v.gate["action"], v.buy
```

`assess` writes nothing. `v.to_json()` is the `--json` object.

## 9. Worked examples

[`examples/gate-winner/`](examples/gate-winner/) holds searches from published tables, each with its
`source`: the arXiv id and version, the table or figure, the basis of its spread, and a published held-out
result where there is one. The spreads there are assumed ranges, since none of these papers prints a
replicate spread on the selection side. A published test-set lead is a fresh-task comparison, set beside
the gain for reference only (section 1).
