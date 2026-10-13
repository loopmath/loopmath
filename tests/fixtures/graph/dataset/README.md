# Dataset fixture: one hand-written synthetic workspace with its label files

Minimal JSONL transcripts for `tests/test_graph_dataset.py` (D1a). Nothing here is copied
from a real session; paths such as `/ws/delta/notes.md` are strings inside the transcripts
and are never touched on disk. The test module builds the run records that point at these
files; timestamps are on 2026-08-31 between 12:00 and 12:16 UTC.

```
projects/ws-delta/lead.jsonl                       top session, the lead (12:00 to 12:10): one Task call, a heredoc Bash write of notes.md, a git commit call
projects/ws-delta/lead/subagents/agent-impl.*      subagent whose .meta.json matches Task id toolu_impl; writes src/parser.py
projects/ws-delta/reviewer.jsonl                   top session (12:15) with no cwd on any line; its first user line is a tool_result, the prompt is the second
labels/swarms.json                                 labels in the README format for the three nodes, one artifact, and one node the graph lacks (delta-ghost)
labels-defects/swarms.json                         every malformed case the loader must count: wrong enum, wrong value type, unknown tier, missing tier, a bare string for a label, duplicated keys (a conflict), an artifacts list, a nodes string
labels-defects/e2-arms.json, contract-v3.json      wrong-type arms container; an attempt with a bad cause type and a string send_back, and a non-object attempt
labels-full/swarms.json                            (D1b) the labels/ file plus a parent label for delta-ghost (an edge the graph lacks because the node is absent), a heuristic consumers tier on notes.md (the positive artifact edge carries the weaker tier) and an artifact flow the graph lacks (parser.py, read by nobody)
labels-full/e2-arms.json                           (D1b) three E2 runs in the README format: T1/S whose dev session is delta-impl (verdict rejected; roles for delta-impl, delta-reviewer and one session the graph lacks), T2/V whose dev session is absent, T3/W under workspace ws-other naming delta-lead (present, but not under that workspace)
labels-full/contract-v3.json                       (D1b) six attempts: W0 n1 and n2 on delta-lead agreeing (merged; the verified session match wins), P1 n1 and n2 on delta-impl disagreeing on both labels (no gold value, both attempts listed), R1 on a session the graph lacks, R2 with no session label
```

`labels/` holds only `swarms.json`, so the loader reports `e2-arms.json` and
`contract-v3.json` missing there; `labels-defects/` and `labels-full/` hold all three.
