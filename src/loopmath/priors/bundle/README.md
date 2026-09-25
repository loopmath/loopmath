OCP v0.3 runs, gzipped, with provenance (lane 11).

`lanes.jsonl.gz` (0.2.2) was built alone (`LOOPMATH_PRIOR_SOURCES=lanes loopmath prior build --lanes-dir PATH`) and merged in with `loopmath.priors.lanes.merge_into_bundle`, which leaves the other files' bytes alone: the other files were not rebuilt. A full `prior build` needs every source's input, `--lanes-dir` included.
