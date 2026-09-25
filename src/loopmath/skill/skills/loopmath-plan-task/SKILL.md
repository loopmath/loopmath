---
name: loopmath-plan-task
description: "Choose the workflow, models and effort for one coding task with loopmath, then start it: recommend, show the ready choices with chance and cost, ask one question, open the run. Use when the user asks to plan a task or an implementation with loopmath, asks which workflow, model or effort to use, or starts a coding task of more than a few minutes of agent work in a repo where they use loopmath."
allowed-tools: Bash(loopmath:*), Bash(git rev-parse:*), Bash(git remote get-url:*), Bash(open:*), Bash(xdg-open:*)
---

# Plan a task with loopmath

One pass: classify the task, ask loopmath, show its choices, ask one question, start the chosen run, then do the work. The user makes one choice; you type everything else.

Use exactly the commands below. Flags and fields are in `reference.md` next to this file. Never run `--help`.

## 1. Classify the task (no question)

From the user's words and the repo, decide:
- `TYPE`: one of `bug_fix`, `feature`, `refactor`, `tests`, `docs`, `research`, `infra`, `data`;
- `TITLE`: the task in at most 60 characters;
- features you can tell: `size`, `lang`, `has_tests`, `spec_clarity`, `needs_design`, `touches` (values in `reference.md`). Leave out the rest;
- a target, only when the user states a score or runtime goal: `--target 'heldout_perf>=2400'`.

Then, in the repo:

```sh
git rev-parse HEAD; git remote get-url origin || git rev-parse --show-toplevel
```

`SHA` is the first line. `REPO` is `owner/name` from the origin URL, lowercased; with no origin, the last folder name of the top level.

## 2. Ask loopmath

```sh
loopmath recommend --type TYPE --repo REPO --title "TITLE" --feature size=m --feature lang=python \
  --base-commit SHA --json --brief --html
```

Read `rec`, `page`, `message`, `reference` (`kind`, `label`, `text`), `rescue`, `choices[]` (`key`, `title`, `label`, `recommended`, `chance`, `run_cost_usd`, `cost_per_accepted_usd`, `expected_rescue_usd`, `strategy`, `price_now_usd`, `gain_per_future_run_usd`). If it exits 5 (no usable fit), run `loopmath fit --json`, then the command above once more.

## 3. Show the choices and ask one question

Show, in this order:
1. `reference.text` in one line. It names the baseline honestly by `reference.kind`: `usual` (the user's habit), `best_recorded` (their best recorded workflow, when they have no habit) or `default`. Never call it "your usual" unless `reference.kind` is `usual`;
2. the choices as a numbered list, `recommended` marked, one line each: `label`, the chance with its range (`chance.mean`, `chance.lo` to `chance.hi`, of `chance.of`), run cost (`run_cost_usd.mean`, median `run_cost_usd.median`), and cost per accepted result (`cost_per_accepted_usd.mean`). For the pair, add `price_now_usd` (the extra run now), `gain_per_future_run_usd` (the expected saving on each future similar task), the payback in tasks (`price_now_usd` divided by `gain_per_future_run_usd`, rounded up; leave it out when the gain is 0), and that the two runs are judged blind;
3. one line on the rescue: cost per accepted result counts `expected_rescue_usd`, the expected cost of redoing a failed run with the reference workflow (`rescue`);
4. the planning page path (`page`);
5. the strategy, when the `goal` choice's `strategy` is not null: quote `strategy.text` word for word, as its own line just before the question. It says to try the pick first and let the reference rescue a miss. Never compute, round or reword it, and never write one when `strategy` is null.

There is a `pair` choice only when a second workflow is worth trying beside the recommended one. When `choices[]` has none, number the choices there are, with `goal` still recommended, say in one plain sentence that no workflow is worth trying beside it right now, and never offer or start a pair.

Then ask exactly one question: "Which one should I start? (1 to N, or describe another workflow)". Example, for a user with a habit (`reference.kind` is `usual`):

> Usual: implement_review: opus-5-5/high, gpt-6-astra/xhigh, 71% chance of an accepted result.
> 1. Recommended: plan_implement: opus-5-5/high, sonnet-5/high: 78% (61 to 90%), $1.20 a run (median $0.95), $1.73 per accepted result
> 2. Pair: 1 plus plan_implement_review: sonnet-5/high x3: extra $0.80 now, judged blind; pays back after 4 similar tasks
> 3. Your usual (reference): 71% (52 to 86%), $1.45 a run (median $1.10), $2.36 per accepted result
> 4. Cheapest run: solo: sonnet-5/medium: 52% (30 to 73%), $0.35 a run, $2.40 per accepted result
> Cost per accepted result includes the expected cost of redoing a failed run with the reference workflow.
> Planning page: /Users/me/.loopmath/views/recommend-20260924-181502.html
> Which one should I start? (1 to 4, or describe another workflow)

When the recommended pick is less likely than the reference but cheaper per accepted result, its `strategy.text` is the line before the question, for example:

> Try plan_implement: sonnet-5/high, sonnet-5/medium first; if it misses (about 4 in 10 tasks), your usual workflow rescues it; expected $1.50 per accepted result, against $2.36 with your usual workflow alone.

## 4. Start the run

```sh
loopmath run start --rec REC --choice KEY --base-commit SHA --json
```

`KEY` is the chosen choice's `key`, always one of the keys in `choices[]`. The choice sets the task, configuration and source, and the run is judged by the rule the recommendation was made for, so never pass `--source` or `--rule` with it. Read `slate`, `runs[]` (`run`, `label`, `piece_settings[]` (`piece`, `role`, `width`, `harness`, `model`, `effort`)). The settings are in workflow order. Tell the user the run id(s) in one line.

For a workflow the user describes instead:

```sh
loopmath run start --rec REC --type TYPE --repo REPO --title "TITLE" --workflow SHAPE \
  --set PIECE=HARNESS:MODEL:EFFORT --source user_edit --base-commit SHA --json
```

with one `--set` per piece. That prints one run, flat, with the fields of one `runs[]` entry.

## 5. Do the work

Run the pieces in order, each with its setting from `piece_settings`. Note every session id; the record step needs them.
- **A piece you do yourself** (its harness and model are yours): do it in this session. In Claude Code its session is `self`; in Codex there is no `self`, so note the folder you work in, for `--cwd PIECE=PATH`.
- **Claude Code piece:** `claude -p --session-id UUID --model MODEL --effort EFFORT "PROMPT"`, with a fresh UUID (`uuidgen`). The UUID is its session id.
- **Codex piece:** `codex exec --json -m MODEL -c model_reasoning_effort="EFFORT" -C DIR "PROMPT"`. The first JSON event's `thread_id` is its session id.
- **Width above 1:** launch that many agents for the piece, each with its own session.
- **Pair:** both runs start from `SHA` in separate git worktrees (`git worktree add ../TASK-a SHA`, `../TASK-b SHA`), and both finish before any judging.

When the work is done, follow `loopmath-record-run`.

## Never

- Do not start a run the user did not choose.
- Do not edit files under `~/.loopmath` or `$LOOPMATH_HOME` by hand.
