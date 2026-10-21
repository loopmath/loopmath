# plugin eval fixtures: written from the docs, never from a run

`pefixture.py` writes synthetic `claude plugin eval` result documents into a test's temp folder. It was written by reading Anthropic's docs on 2026-10-20, 10:0x to 10:16 PDT, read only. Nothing here runs Claude Code, starts a process or reads a real result, and every string inside is loopmath's own.

The docs give no JSON example of the result. Their field table lists "the fields a gating script usually reads" (https://code.claude.com/docs/en/plugin-evals#json-result), so it is not a full schema. Each field below is either **documented**, at the URL given, or an **assumed** name for content the docs describe without naming. The reader checks its reading of the run grades against each case's documented `aggregates.score` and `aggregates.delta`, and refuses a file where they disagree. So an assumed name that a real file spells differently fails loudly, never silently.

PE is https://code.claude.com/docs/en/plugin-evals.

| field | status | source |
|---|---|---|
| `aggregate-result.json` | documented: in `<eval dir>/results/<timestamp>/`, or the `--json [path]` document | https://code.claude.com/docs/en/plugin-evals#read-the-results |
| `schemaVersion` | documented: `1`; new fields are added without renaming, so readers ignore unknown ones | https://code.claude.com/docs/en/plugin-evals#json-result |
| `claudeVersion` | documented: the Claude Code version that ran the suite | https://code.claude.com/docs/en/plugin-evals#json-result |
| `partial` | documented: true when the suite stopped early | https://code.claude.com/docs/en/plugin-evals#json-result |
| `partialReason` | documented: `cost_ceiling`, `interrupted` or `auth_failed` | https://code.claude.com/docs/en/plugin-evals#json-result |
| `costUsd` | documented: the suite's list-price estimate, judge calls included | https://code.claude.com/docs/en/plugin-evals#json-result |
| `durationSeconds` | documented: the suite's wall-clock seconds | https://code.claude.com/docs/en/plugin-evals#json-result |
| `aggregates.casesPassed` | documented: cases at or above `--threshold` | https://code.claude.com/docs/en/plugin-evals#json-result |
| `aggregates.casesTotal` | documented: the number of cases | https://code.claude.com/docs/en/plugin-evals#json-result |
| `cases[].name` | documented: the case name | https://code.claude.com/docs/en/plugin-evals#json-result |
| `cases[].aggregates.score` | documented: the mean with-arm run score | https://code.claude.com/docs/en/plugin-evals#json-result |
| `cases[].aggregates.delta` | documented: with minus without; omitted for one arm or arms not comparable | https://code.claude.com/docs/en/plugin-evals#json-result |
| `cases[].arms.with[]` | documented: the with-plugin runs | https://code.claude.com/docs/en/plugin-evals#json-result |
| `cases[].arms.without[]` | documented: the no-plugin baseline runs | https://code.claude.com/docs/en/plugin-evals#compare-against-a-no-plugin-baseline |
| `run.error` | documented: null or a reason such as "timed out after 300s"; a max_turns stop is a run error | https://code.claude.com/docs/en/plugin-evals#json-result |
| `run.aborted` | documented: a mock stopped the run (server, tool, reason); score 0, error null | https://code.claude.com/docs/en/plugin-evals#json-result |
| `run.skippedPaidGraders` | documented: judge graders skipped at the cost ceiling | https://code.claude.com/docs/en/plugin-evals#json-result |
| `grader.passed` | documented: a grader's result on a run | https://code.claude.com/docs/en/plugin-evals#troubleshooting |
| `grader.scored` | documented: false when a grader is excluded from scoring in two-arm mode | https://code.claude.com/docs/en/plugin-evals#compare-against-a-no-plugin-baseline |
| `grader.weight` | documented: a grader's weight, default 1 | https://code.claude.com/docs/en/plugin-evals#how-a-case-is-scored |
| `run.score` | assumed name; the run score itself is "the fraction of graders passed, weighted by weight" | https://code.claude.com/docs/en/plugin-evals#how-a-case-is-scored |
| `run.graders` | assumed name; the reader finds the run's one list of items with a boolean `passed`, by shape, not by this key | https://code.claude.com/docs/en/plugin-evals#json-result |
| `plugin.json` | documented: the plugin manifest at the plugin root or in `.claude-plugin/` | https://code.claude.com/docs/en/plugin-evals#requirements |

Also from the docs, and used by the reader's rules:

- With `--ablation none` only one arm runs (https://code.claude.com/docs/en/plugin-evals#command-options). The docs do not say how `arms.without` looks then; the fixture leaves it out.
- The default `--threshold` is 1.0 (https://code.claude.com/docs/en/plugin-evals#command-options). A case passes when its score reaches it (https://code.claude.com/docs/en/plugin-evals#how-a-case-is-scored).
- Since 2.1.299, a run that ends on an API error scores 0 without grading (https://code.claude.com/docs/en/changelog#2-1-299): it "scores 0 with no grader results", and its case's W/OUT and delta print as a dash (https://code.claude.com/docs/en/plugin-evals#runs-ended-on-an-api-error). No member marks such a run; its `error` is free text. So the reader never infers a 0: a run with neither a score nor a grader list is refused unless it is `aborted`, or its case has no delta and is withheld.
- The docs name pass and fail, `scored` and weights, not their JSON types. The fixture writes JSON booleans for `passed` and `scored` and numbers for `weight` and `score`, and the reader requires those types: 1, 0 or "yes" for `passed` is refused, never read as truthy. These types are part of the reader's provisional layout, unverified against a real result.
- The docs omit `cases[].aggregates.delta` when the case ran one arm or the arms are not comparable (https://code.claude.com/docs/en/plugin-evals#json-result). The fixture omits it only where a test says so.
- There are no custom-code graders (https://code.claude.com/docs/en/plugin-evals#grader-types). The twin's `tests-pass` grader is a `file_exists` stand-in for the shipped rows' pytest verdict.
- Not in the result at all: per run cost, tokens, model, duration, times, session id or transcript path.

The twin (`twin_suite`, `TWIN_COSTS`) restates `spec/examples/harness-runs/plugin-evals.jsonl`: the same 3 tasks, 1 run per arm, the same verdicts, and that file's dollars and tokens as a `--cost-from` file.
