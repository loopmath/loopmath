# Skeleton fixture: one hand-written synthetic swarm

Minimal JSONL transcripts for `tests/test_graph_extract.py` and `tests/test_graph_render.py`.
Nothing here is copied from a real session; paths such as `/ws/alpha/plan.md` are strings
inside the transcripts and are never touched on disk. `tests/test_graph_extract.py` builds
the run records that point at these files and documents the scenario in its module docstring.

```
projects/ws-alpha/lead.jsonl                      top session, the lead (two Task calls, three Bash calls; reads plan.md before it exists)
projects/ws-alpha/lead/subagents/agent-plan.*     subagent with a .meta.json matching Task id toolu_plan
projects/ws-alpha/lead/subagents/agent-dev.*      subagent matching Task id toolu_dev
projects/ws-alpha/lead/subagents/agent-orphan.*   meta names a Task id the lead never issued
projects/ws-alpha/ghost/subagents/agent-lost.jsonl  no meta, enclosing session not in the records
projects/ws-alpha/reader.jsonl                    top session that reads plan.md after the lead ended; one Read has an unparseable timestamp
projects/ws-other/analyst.jsonl                   another workspace; launches codex into /ws/alpha
codex/rollout-review.jsonl                        codex session launched by the lead's Bash call
codex/rollout-audit.jsonl                         codex session launched by the analyst
codex/rollout-stray.jsonl                         codex session no record launched
```

Timestamps are all on 2026-08-31 between 10:00 and 10:31 UTC; the lead runs 10:00 to 10:10.
