# OpenEvolve and ShinkaEvolve fixtures: written from the code, never from a run

`oefixture.py` and `shinkafixture.py` write synthetic program databases into a test's temp folder. `simdb.py`
uses them to simulate whole runs with known per-model rates. They were written by reading each tool's source on
GitHub at a pinned commit. Nothing here imports, installs, runs or copies either tool. The code, prompts, diffs and
feedback inside the databases are loopmath's own marker strings, and tests check that none of them reaches a row or
a run file.

- OpenEvolve, Apache 2.0, `https://github.com/algorithmicsuperintelligence/openevolve` at
  `34427bae6239063d46be0d38e0fabe11279d05aa`. Paths are relative to `openevolve/`.
- ShinkaEvolve, Apache 2.0, `https://github.com/SakanaAI/ShinkaEvolve` at
  `8adc053a2ce4511ad2ac310e004c530a73fb974a`. Paths are relative to `shinka/`.

## OpenEvolve

| What the builder writes | Source |
|---|---|
| `Program` fields, one line each | `database.py:65-100`; `to_dict` is `asdict` (102-104) |
| `<db>/programs/<id>.json`, with `prompts` when `log_prompts` | `database.py:942-977` (prompts 971-973) |
| `<db>/metadata.json` keys | `database.py:760-774` |
| `feature_stats` keyed by each feature dimension; defaults `complexity`, `diversity` | `database.py:771, 979-1026, 2429-2445`; `config.py:343-350` |
| fitness: `combined_score`, else the mean of numeric metrics that are not feature dimensions | `utils/metrics_utils.py:72-118` |
| checkpoint folder `checkpoints/checkpoint_<iteration>` | `controller.py:446-501` |
| output folder `<initial program dir>/openevolve_output` | `controller.py:59-60` |
| child `parent_id`, `generation`, `iteration_found`, metadata `changes`, `parent_metrics`, `island`, `token_usage` | `process_parallel.py:373-389`, usage from 217 |
| `token_usage` keys and `model` | `llm/openai.py:54, 235-253`; `llm/ensemble.py:98-105` |
| root has no parent and no metadata | `controller.py:275-300` |
| migrant copies | `database.py:2114-2140` |
| eviction past `population_size`, protecting only the best and the new program (the root can go) | `database.py:1916-1990`; `config.py:329` |
| failed iterations are not stored | `process_parallel.py:744-756` |
| trace line fields; prompts in by default | `evolution_trace.py:24-49`; `config.py:417-427` |
| trace only for a child whose parent is present | `process_parallel.py:776-802` |

## ShinkaEvolve

| What the builder writes | Source |
|---|---|
| `programs.sqlite` in the results folder, WAL journal; default `results_<%Y%m%d_%H%M%S>` | `core/async_runner.py:282-286, 1192`; `database/dbase.py:411` |
| per-query record `gen_<g>/attempts/novelty_<n>/resample_<r>/patch_<p>/metadata.json`: attempt numbers, `success`, `num_applied`, naive local `timestamp`, and `llm_cost`, `llm_model` when there was a reply | `core/async_runner.py:3050-3125`; `core/config.py:16`; written at 3466, 3524, 3569, 3691, 3761, 3805 |
| one proposal per generation: its folder is created exclusively | `core/async_runner.py:2604-2634` |
| model weights once per proposal; each query draws its own model | `core/async_runner.py:2733-2737`; `llm/llm.py:619-645` |
| `api_costs` sums the proposal's queries | `core/async_runner.py:2805-2817, 2969` |
| `programs` table, column for column | `database/dbase.py:423-449` (later columns 545-558) |
| `attempt_log` table and `record_attempt_event` | `database/dbase.py:507-523, 713-735` |
| program insert; JSON columns | `database/dbase.py:866-899` |
| `timestamp` | `database/dbase.py:167`; `core/async_runner.py:4172` |
| child metadata: `meta_patch_data`, costs, logs, pipeline times | `core/async_runner.py:3779-3797, 4179-4208`; `core/pipeline_timing.py:30-54` |
| `llm_kwargs`: `model_name`, `temperature`, `reasoning.effort` (the proposal's draw) | `llm/kwargs.py:103-107, 117-175` |
| `llm_result.kwargs` are the reply's own provider kwargs | `llm/providers/openai.py:207-209`; `llm/kwargs.py:143-159` |
| `llm_result` = `QueryResult.to_dict()` | `llm/providers/result.py:41-59` |
| the writer is the query's `llm_model` (= its reply's `model_name`); the stored `model_name` is a separate draw | `core/async_runner.py:3670, 3678-3684, 3116-3118`; `llm/llm.py:619-645` |
| `correct`; missing results give false and 0.0 | `core/async_runner.py:4125-4147` |
| failed proposal `details` | `core/async_runner.py:3307-3398` |
| Shinka's own failed-generation count leaves out generations with a program | `core/async_runner.py:6048-6069` |
| island copies and spawned copies | `database/islands.py:603-660, 788-836` |
| a copy of a root keeps a null parent: the initial program's island copies, a spawned island's root | `database/dbase.py:976-980`; `database/islands.py:603-660, 974-1000` |
| bandit allocation by default | `core/config.py:30`; `core/async_runner.py:4484-4498` |
