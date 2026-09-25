# Changelog

## 0.1.1 (2026-09-24)

### recommend

- The workflows you recorded for the task's type and repo (else its type), designed runs included, are candidates with their own shape and width: up to 60, most runs first, and one-step edits of the first 30. A catalog or edited workflow that runs the same as a recorded one is listed once.
- With no usual workflow, the baseline is a reference: your best recorded workflow by expected cost per accepted result, labelled as such ("no usual workflow; reference: your best recorded workflow (...)"). Text and JSON never call a workflow you did not run "your usual".
- Codex offers `max` effort for the gpt-5.6 and gpt-6 models, and candidates use every effort you have run.
- The rescue is named once at the top. Each workflow shows its run cost with the median beside the mean, the expected rescue, and the cost per accepted result; a score target's chance to reach carries its 80% range.
- `recommend --json` is `loopmath.recommend/2`: `reference`, `numbers`, and `choices` (at most four options an agent can show before it asks one question); `usual` is null when you have none.
- `recommend --fit ID` and `posterior --fit ID` read a kept fit.

### fit, status and sources

- Runs in your store are yours, whatever their `task.source` says. A shipped source name never matches them: `fit --without rq1` leaves out only the shipped RQ1 runs, while `--without user` and `--without shared` still leave out your runs and your shared imports. When a stored run has the same id as a shipped one, your copy is used, and `run import`, `fit` and `status` say so once.
- `fit` is about four times faster (a block factorization, Newton steps, and lazy imports).
- `status` names the current fit's options and counts, for example `fit: fit_..., 0 min old, --without rq1, runs 1469 prior + 44 yours`.
- `posterior --subtype S --feature K=V` predicts for that task.

### run import, runs, ocp validate

- `run import` takes several files or a directory, with one refit at the end; `--no-fit` skips it.
- `ocp validate` ends with `N passed, M failed`.
- `runs RUN` works as `runs --run RUN`; the table shows the model and effort of each piece, and `--wide` adds the configuration id.
- Plurals and units read correctly (`1 run`, `heldout_perf 3140`).
- `plan` is hidden from `--help` and prints a note on stderr that it is experimental; its output is unchanged.

### onboard

- Reading the history uses up to 8 worker processes (one per CPU). Set `LOOPMATH_WORKERS=1` to keep everything in one process.

### OCP

- `cost.output_tokens` includes reasoning tokens; `reasoning_tokens` is a breakdown a reader never adds. The OpenCode adapter now adds reasoning into `output_tokens`.
- `model.raw` keeps a label the source wrote with the effort in it; a producer never builds one, and effort known separately goes in `effort`.
- The full-fields example records an accepted run without a rescue.

### Views

- The posterior page lists every configuration recorded for the task's type and repository, grouped by workflow graph, with no cap.
- A piece that runs several workers is drawn as one box per worker (up to 6), and configuration labels name the width (`best_of_n: 3 x gpt-5.6-sol/xhigh`).
- The plans page names the rescue once, adds an expected rescue column, shows the chance to reach a score target with its range and the median run cost beside the mean, and names the reference when you have no usual workflow.
