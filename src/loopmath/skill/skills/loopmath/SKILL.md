---
name: loopmath
description: Use whenever the user mentions loopmath.
allowed-tools: Bash(loopmath:*), Bash(git rev-parse:*), Bash(git remote get-url:*), Bash(open:*), Bash(xdg-open:*)
---

# loopmath

## What loopmath is

loopmath is a command line tool that helps choose how to run a coding agent task: which workflow, which models and how much effort. It does three things:

- **Predicts.** For a task, it estimates each workflow's chance of meeting the task's tests or stated target, and what it costs.
- **Records.** When a task is done, it stores what ran, what it cost and whether the result was accepted.
- **Refits.** From every recorded run it updates its estimates, so the next prediction is better.

Three words come up everywhere:

- A **workflow** is how a task is done: a shape such as `solo`, `plan_implement` or `implement_review`, plus the agent, model and effort for each piece.
- A **run** is one task done with one workflow: the agent sessions that did the work, their cost and the outcome.
- A **fit** is loopmath's current estimate of each workflow's chance and cost, with ranges. It is built from the user's recorded runs plus a starting estimate that ships inside loopmath.

loopmath never starts agents itself. It recommends; you start the agents, and only for the workflow the user chose.

## How to learn each job

The step by step instructions for each job come from the CLI: `loopmath guide <topic>`. A guide call only prints instructions; it never runs the job or changes anything. The guides ship inside the installed loopmath, so they always match its commands and JSON output, even after an upgrade. Read a job's guide in full before running the commands it names, then follow it. `loopmath guide` alone lists every topic.

Do not run loopmath commands with `--help`: the help text is written for people at a terminal and lists options the guides leave out on purpose. If a guide seems to miss something, run `loopmath doctor --json` and tell the user.

## What to do first

1. Run `loopmath status --json`. If `loopmath` is not found, tell the user to install it with `uv tool install --python 3.12 loopmath`, and stop. If status shows update.available, tell the user in one line and offer the update.command; run it only if they agree.
2. Pick the guide from what the user asked for:

| When | Run and follow |
|---|---|
| set up, start or try loopmath | `loopmath guide onboard` |
| OCP files or a folder of finished runs to bring in | `loopmath guide import` |
| a task to do, or which workflow, model or effort to use | `loopmath guide plan` |
| a task is done: record its run, or judge two runs blind | `loopmath guide record` |
| update the fit, or what loopmath learned | `loopmath guide fit` |
| a command, flag, JSON field or exit code you need | `loopmath guide reference` |

   For example, if you need to onboard the user, run `loopmath guide onboard` and follow its output.

3. If the user named no job, decide from the status, in this order:
   - no store (`exists` false) or no runs (`counts.runs` 0): onboard;
   - `open_runs` not empty: record, but only runs whose work is done;
   - `fit.usable` false: fit, and say `fit.problem` in one line;
   - otherwise say in three lines what loopmath knows (how many runs, which fit) and offer to plan the next task, bring in existing runs, or open the results page (`loopmath posterior --html`).

## Rules that always hold

- Never start agents or a run without the user's choice. A recommendation is advice, not a choice.
- Never edit the store (`~/.loopmath`, or `$LOOPMATH_HOME`) by hand. Only loopmath commands write there.
- Never record another person's sessions.
