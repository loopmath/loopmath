---
name: loopmath
description: "Plan agent workflows with loopmath and record how they went. Use before starting a coding task that will take more than a few minutes of agent work, when the user asks which workflow, model or effort to use, or when the user asks how past agent runs went or what loopmath believes about a model or workflow. loopmath predicts, records and refits; it never starts agents itself."
---

# loopmath

loopmath is a CLI that helps you, the orchestrator, choose a workflow for a coding task and learn from how it went. It predicts the chance of success and the cost of the user's usual workflow and of alternatives, suggests exploration picks with a price and a payback count, records each run you make, fills token costs from the Claude Code and Codex logs, and refits.

loopmath never starts or coordinates agents. You do that, as you always do. loopmath only informs and records.

Every command takes `--json` and then prints exactly one JSON object on stdout. Use `--json` whenever you read the output yourself.

## When to use it

- Before starting a coding task that will take more than a few minutes of agent work.
- When the user asks which workflow, model or effort to use.
- When the user asks how past runs went, or what loopmath believes about a model, effort or workflow.

If `loopmath` is not on the PATH, tell the user and continue without it. If a command exits 5 (no fit yet), run `loopmath fit` once and retry. Run `loopmath doctor` when something looks wrong.

## First run: onboarding

`loopmath onboard` turns the user's past Claude Code and Codex sessions into recorded runs, labels their task types, and finds the usual workflow per task type and repo. Labelling sends a short summary of each session group to a model, so the user chooses that model. Never choose it yourself.

Before the first `loopmath onboard`, ask the user which model should label their sessions, and name the options:
- a Claude model through their Claude Code CLI: `--labeler claude:<model>`, for example `claude:claude-haiku-4-5`;
- a GPT model through their Codex CLI: `--labeler codex:<model>`, for example `codex:gpt-6-luna`;
- a local model through a command they name, so nothing leaves the machine: `--labeler command:<cmd>`, for example `command:ollama run <model>`;
- no labelling: `--labeler none` (task types then stay unclassified).

Then run `loopmath onboard --labeler CHOICE --dry-run` and show the user what it found and the expected labelling cost. Run it for real (`--yes`) only after the user agrees. loopmath saves the confirmed choice as config `onboard.labeler`. If `loopmath onboard` exits 2 asking for a labeller, ask the user again; do not retry with a model of your choosing.

## Plan

1. **Classify the task.** Pick one type from `loopmath task-types --json`, the repo (`owner/name`, or a local name), a subtype if the user has them, and the features you can tell: `size`, `lang`, `has_tests`, `spec_clarity`, `needs_design`, `touches`. Unknown is fine; leave out what you cannot tell.

2. **Ask for a recommendation.**

   ```sh
   loopmath recommend --type TYPE --repo REPO [--subtype S] [--title "TEXT"] \
     --feature size=m --feature lang=python ... [--base-commit SHA] --json > rec.json
   ```

   Keep `rec.json` (outside the repo, or in a folder git ignores): `run start` reads the task from it.

   When the user states a score, runtime or quality goal, add a target: `--target 'heldout_perf>=2400'` or `--target 'runtime_s<=200'`. A named rule from config is `--rule NAME`.

3. **Show the user the result, then ask.** Show the `message` field exactly as it is. Then show the `curve` as a short table (level, workflow, chance of success, cost in dollars and tokens). Then ask which of these to run:
   - the reference workflow (`reference`): the user's usual workflow when they have one, else their best recorded workflow or the default (`reference.kind` says which; `usual` is null without a habit);
   - the goal pick, if it differs from the usual (`default_pick`, or the curve row the user names);
   - the pair with the best-value exploration: the goal plus `exploration.best_value`;
   - the pair with the biggest-gain exploration: the goal plus `exploration.max_gain`, when it is not `{same_as: "best_value"}`;
   - one of the `alternatives`.

   Standing rule: if `loopmath config get explore.auto_payback_runs --json` has a value, and the pick named by `explore.default_pick` (default `best_value`) has `auto_ok: true`, run that pair (`pair` in the output) without asking, and tell the user you did and why. If a pick is `{paused: ...}`, say that the budget cap is reached for it. If it is `{none: ...}`, say that no exploration has a positive gain.

4. **Offer the plans page** when the user wants to see the options: `loopmath recommend ... --html` writes a page and prints its path.

Keep the `rec` id from the output. You pass it to `run start`.

## Record

5. **Start each run.** One run per workflow you run.

   ```sh
   loopmath run start --task-file rec.json --base-commit SHA \
     --config CFG --source usual|alternative|exploration|user_edit --rec REC --json
   ```

   `CFG` is the chosen configuration's id (`cfg_...`): for a pair, `pair.members` lists both; otherwise take `goal.config`, `default_pick.config` or a curve row's `config`, or `config.id` inside `reference` (or `usual`), an alternative, or `exploration.best_value.candidate` (or `max_gain`). Use `--source usual` only when the run is the user's usual (`reference.kind` is `usual`); otherwise `alternative`.

   `--task-file rec.json` takes the task from the recommendation, with its id. For a pair, start the first member with `--new-slate` and the second with `--slate SLT` (the slate id printed by the first), both from the same `rec.json` and the same base commit: a slate holds one task, and task flags would make a new one. Keep each printed run id. Without a recommendation, give the task as flags instead: `--type TYPE --repo REPO [--feature K=V ...]`.

6. **Record each agent before it starts.** For each piece of the workflow (`plan`, `implement`, `review`, ...):

   ```sh
   loopmath run attempt --run RUN --piece PIECE --harness claude-code|codex|command \
     --model MODEL --effort EFFORT --cwd PATH [--session ID] --json
   ```

   How to get the session id, so the cost match is exact:
   - **Claude Code.** Generate a UUID and launch with `claude --session-id <uuid>` (headless: `claude -p --session-id <uuid> ...`). Pass the same UUID as `--session`.
   - **Codex headless.** Launch with `codex exec --json ...`; the first event carries the thread id. Pass it as `--session`.
   - **Codex interactive.** Omit `--session`. loopmath matches by working folder, time window and model, at the heuristic tier.
   - **You do the piece yourself**, in your own session: under Claude Code, use `--session self` (loopmath reads your own session id). Under Codex, pass your own session id explicitly; `--session self` is an error there.

   When the agent stops, settle the attempt:

   ```sh
   loopmath run attempt --run RUN --end ATT --status done|failed|rejected|canceled|lost
   ```

   For a repair round, record a new attempt for the same piece with `--round K --cause sent_back` (a reviewer sent it back) or `--cause gate_failed` (a check failed).

7. **Record commits.** When an attempt produces a commit, record it, so later issues and reverts can be traced to the run:

   ```sh
   loopmath run artifact --run RUN --kind commit --path SHA --by ATT
   ```

8. **Record verdicts after each gate.**

   ```sh
   loopmath outcome --run RUN --signal tests=pass --kind verdict --at-attempt ATT \
     --source orchestrator|ci|user|judge --tier verified|reported
   ```

   Use tier `verified` only for results you observed directly: a test command's exit code, CI status read from the API, or a merged pull request. Use `reported` for what an agent told you.

9. **Record scores when you have them** (benchmark score, runtime, quality):

   ```sh
   loopmath outcome --run RUN --signal runtime_s=182 --kind score --unit s --better lower [--target 200] [--scale linear|log|fraction]
   ```

   If the user's rule names a score you could not measure, record it with an empty value (`--signal quality=`) so the slot exists.

10. **Finish the run.**

    ```sh
    loopmath run finish --run RUN --json
    ```

    Show the user the one-line receipt: predicted against actual cost, and the outcome. If attempts are unmatched, report them as they are. Never guess them.

## Pair

11. Both runs start from the same base commit, in separate worktrees, and both finish before you judge them.

12. Launch a referee: a fresh agent of the family in config `referee.model` (by default a family different from both runs' implementers). Give it the task and the two diffs in random order as A and B, without run ids, workflow names or models. Use this prompt:

    "Two changes attempt the same task. Judge which one better completes the task as stated, considering correctness, tests, scope and code quality. Answer A, B or tie, then give three sentences of reasons."

13. Record the preference, mapping A or B back to its run:

    ```sh
    loopmath outcome --slate SLT --prefer RUN|tie --judge referee --blinded
    ```

    Record each run's own verdicts as in step 8. Keep the preferred change. The other worktree is the user's to delete.

## Later

14. When the user mentions an issue, incident, revert or hotfix tied to a change made under loopmath, record it:

    ```sh
    loopmath outcome --commit SHA --signal incident=INC-42 --kind event --source user|tracker
    ```

15. To answer "how have past runs gone", run `loopmath runs --html`. To answer "what does loopmath believe about X", run `loopmath posterior --html` (add `--level model|effort|role|topology|type|repo|feature` or `--workflow CFG`). Both print the path of the page they wrote. `loopmath status` shows the fit's age, open runs and spend against the cap.

## Never

- Do not start agents because loopmath said so, without the user's choice or a standing rule in config.
- Do not edit files under `~/.loopmath` (or `$LOOPMATH_HOME`) by hand. Use the commands.
- Do not record another person's sessions.
