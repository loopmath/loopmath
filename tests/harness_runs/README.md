# harness2 fixtures: written from the code, never from a run

`h2fixture.py` writes synthetic harness2 runs folders into a test's temp folder. It was written by reading harness2 (clone at `1cfe254`, Apache 2.0, `src/harness2/`). Nothing here imports, runs or copies harness2. The text inside the trees, answers and streams is loopmath's own.

The real demo output that cross-checks this reading stays outside the repo. Neither harness2-authored text nor private data ships.

Paths below are relative to `src/harness2/` in the clone.

| what the builder writes | harness2 source |
|---|---|
| runs root `<RUNS>`; adapt folder `<RUNS>/<bench>/adapt` | `harness2/config.py:91`; `harness2/cli/_common.py:70-72` |
| namespace `<bench>-<sub>-<mode><k>[-s<seed>]`; modes `agg`, `seq` | `harness2/core/schedule.py:41-42, 111-127` |
| summary in the `agg` namespace for both modes | `harness2/cli/_common.py:91-94` |
| task folder `<ns>/<task id with / as __>`; run id `<ns>/<task>/<label>` | `harness2/core/schedule.py:98-102` |
| `config.json` keys (`method_version`, `executor`, `judge`, ...) | `harness2/cli/run_sweep.py:209-235`; `METHOD_VERSION` `harness2/config.py:26` |
| `summary.json` top level | `harness2/core/schedule.py:251-262, 289-328, 609` |
| `base` entries `{label, run_id, run_dir}`, run once in `agg`, shared | `harness2/core/schedule.py:606, 619-628` |
| bare-label `base` entries (legacy) | `harness2/cli/judge.py:115-118` |
| `modes[m]`: finals, ok, stages, ledger | `harness2/core/recursion.py:79-105` |
| ledger fields; `rollout_usd` always 0 | `harness2/core/ledger.py:58-59, 115-132`; `harness2/core/recursion.py:299, 338, 360-361` |
| stage labels and parents (parallel edits base; round t edits base or `c<t-1>`) | `harness2/core/recursion.py:360-425, 476-562` |
| trees `harnesses/base`, `harnesses/<label>` | `harness2/core/recursion.py:274-276, 366, 479-482` |
| tree files (OpenCode surface; Codex `AGENTS.md`, `.agents/skills/`) | `harness2/core/prompts.py:424-431`; `harness2/substrate/codex.py:41, 47`; README stubs `harness2/core/tree.py:141` |
| `selections[m]` | `harness2/core/selection.py:46-68, 237-252` |
| a crashed mode `{ok: false, error, traceback}` | `harness2/core/schedule.py:655-659` |
| legacy `arms` and `arm` keys | `harness2/core/schedule.py:74-81`; `harness2/cli/judge.py:57-59, 81-86` |
| `judged.json` rows (`Scored` plus `frac`, `graded`) | `harness2/core/judge.py:32-53, 80-85, 146-155`; `harness2/cli/judge.py:186-191` |
| `base_judged.json` rows `{task, label, run_id, score}`, indent 1 | `harness2/cli/judge.py:108-147` |
| judge score dict, `judge_model` | `harness2/benchmarks/jobbench.py:600-618` |
| JB rollout folders `dataset/<task>/model_output/<label>`, `model_traj/<label>/*.jsonl` | `harness2/benchmarks/jobbench.py:642, 676-681, 721-723, 869-880` |
| LAB rollout folder `lab/results/<oc\|cx>/<run_id>`, session file names | `harness2/benchmarks/lab.py:524-529, 585-586, 762` |
| LAB runner `config.json` (`model`, `reasoning_effort`, `started_at`) | `third_party/lab_runner/opencode/run_opencode.py:230-251`; `third_party/lab_runner/adapt_runs_cx/run_codex_lab.py:264-281` |
| `.adapt_complete` `{version, run_id, at, harness}`; base digests to `"base"` | `harness2/core/runmark.py:30-81` |
| harness2 fingerprint (SHA-1, 16 hex) | `harness2/core/runmark.py:41-61` |
| OpenCode stream lines, `step_finish` tokens and cost | `harness2/benchmarks/jb_opencode_runner.py:32, 203-206`; `third_party/lab_runner/opencode/session.py:38-49`; `third_party/wb_runner/agents/oc_agent.py:224, 353-370`; `harness2/core/proposer.py:486-504` |
| Codex `turn.completed` usage | `harness2/substrate/codex.py:432-444` |
| JB retries are separate attempt files | `harness2/benchmarks/jobbench.py:801, 872-875` |
| WB bench `wb-<subset>` at `<RUNS>/wb/<subset>`; subsets `code`, `web`, `office` | `harness2/benchmarks/workbuddy.py:84-110, 653-658` |
| WB trial folder `rollouts/<label>/<run>/<task, at most 32 characters>__<suffix>`; label is the run id with `/` as `__`, sanitized | `harness2/benchmarks/workbuddy.py:113-114, 121-145, 451, 565-579, 716-728` |
| WB bases run as one batch: label `<first base run id>__x<k>`, every fresh trial marked with the first base's run id | `harness2/benchmarks/workbuddy.py:826-858`; `harness2/core/schedule.py:430-470` |
| WB stage: one trial per run id, marked with that run id and its tree | `harness2/benchmarks/workbuddy.py:805-824`; `harness2/core/recursion.py:271-299` |
| WB marker extra keys `solver`, `oc_ver`, `connection` | `harness2/benchmarks/workbuddy.py:738-746`; `harness2/config.py:99-100, 168, 172` |
| WB run dirs recorded absolute (base `run_dir`, `chosen_run_dir`) | `harness2/config.py:91`; `harness2/core/schedule.py:540-541, 628` |
| WB trial contents: `verifier/score.json`, `agent/oc-output.txt` or `agent/codex-output.jsonl` | `harness2/benchmarks/workbuddy.py:408-420, 863-864` |
| WB judge score `{n_passed of 100, n_criteria 100, all_pass, raw_score}` in `judged.json` raw and `base_judged.json` | `harness2/benchmarks/workbuddy.py:986-1004`; `harness2/core/judge.py:72-90`; `harness2/cli/judge.py:108-147` |

The variants below change one thing each:

| variant | what it writes | harness2 source |
|---|---|---|
| `stale-judge` | the `agg` selection now adopts `agg2`; `judged.json` still has the row for the earlier `agg1` run | `harness2/cli/judge.py:65-80` (rows merge per `(task, mode, k)`, and a row carries its own `run_id`) |
| `failed-rollout` | `dom1`'s folder and stream exist, the stream stops after a `step_start` (OpenCode) or `turn.started` (Codex), and no `.adapt_complete` is written | `harness2/core/runmark.py:64-67` (marked only after a normal return); `third_party/lab_runner/opencode/session.py:42`; `harness2/substrate/codex.py:432, 446` |
| `mixed-cost` | one `step_finish` reports `cost`, the next has tokens and no `cost` key | `third_party/lab_runner/opencode/session.py:44-49` (a missing cost reads as 0 there); `third_party/wb_runner/agents/oc_agent.py:353-370` |
| `moved` (WB) | recorded run dirs name `/srv/harness2/runs/...`, where harness2 ran, not the folder being read | `harness2/config.py:91` (an absolute runs root); `harness2/core/schedule.py:540-541, 628` |
| `rerun-trial` (WB) | an earlier trial of `inv1` in another run folder, marked with the same run id and another tree's fingerprint | `harness2/benchmarks/workbuddy.py:753-758, 805-824` (a trial is reused only when its marker names the current tree; otherwise a fresh one is marked) |
| `dead-rollout` | a rollout listed in the ledger with no output or trajectory folder | `harness2/core/recursion.py:289-299, 338` (a runner that raised gets `run_dir` `missing`, and the ledger row is still written) |
