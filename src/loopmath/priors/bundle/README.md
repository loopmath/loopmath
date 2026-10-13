loopmath's shipped prior: OCP v0.3 runs, gzipped, with provenance in `manifest.json`.

Every file here and `manifest.json` come from one full `loopmath prior build` (0.2.3): `sweep`, `e0`, `rq1` and `lanes`, each with its inputs' digest, converter, notes and counts in the manifest. A full build needs every source's input: `--sweep-dir` (repeatable, one results folder per sweep batch), `--e0-corpus`, `--rq1-dir` and `--lanes-dir`, or their environment variables (`LOOPMATH_SWEEP_DIR` names one folder).

- Sweep batches: `sweep0830` and `sweep0925` (gpt-6-sol and gpt-6-luna developers). The batch is the run id prefix (`sweep0925/<run>`) and the source ref. Both batches use the task set's ids (`sweep0830/<task>`): they ran the same tasks and hidden gates, so a task is one node in the fit.
- No real date or clock time: every run starts at `1970-01-01T00:00:00Z` and keeps its real elapsed seconds, in UTC (`run.ext["dev.loopmath.prior"].clock`). RQ1 run ids carry no start stamp; a store that keeps the stamped ids still matches the shipped copies (`loopmath.priors.run_id_of`). The price table's date (`cost.tariff.date`) is the only date in a run, and `built_at` is the build's UTC date.
- `lanes`: rounds are capped at the catalog budget of 3 (see the manifest notes).
