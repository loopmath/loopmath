# Launch fixtures (task L1: script-mediated and backgrounded launches)

A synthetic workspace and the transcripts of one build night in it, for
`tests/test_graph_launch.py`. Paths inside the transcripts are placeholders the test
substitutes before use: `__WS__` is the workspace directory (`launch-ws/` here, or a path
that no longer exists when the test exercises the workspace-name fallback), `__OTHER__`
the foreign workspace (`other-ws/`).

- `launch-ws/tools/review.sh`: the review harness; its text runs `codex exec`.
- `launch-ws/tools/noop.sh`: a script that launches nothing.
- `other-ws/tools/review.sh`: the same harness in another workspace.
- `parent.jsonl`: the lead session in `launch-ws`. Six Bash calls, all at 2026-08-30:
  1. 10:00:00 `nohup tools/review.sh T1 ... &`, tool result 0.5 s later (backgrounded).
  2. 10:05:00 `for t in T2 T3; do tools/review.sh $t ...; done`, result at 10:11:00.
  3. 10:15:00 `tools/review.sh T5 ...`, result 1 s later, no session within 120 s.
  4. 10:20:00 `tools/noop.sh`: not a launcher.
  5. 10:21:00 `grep -n codex tools/review.sh`: names the script without running it.
  6. 10:22:00 `bash ../other-ws/tools/review.sh T6 ...; tools/gone.sh; tools/$t.sh`: three
     script candidates, all rejected and counted (outside the root, missing, unresolved).
- `foreign.jsonl`: a session in `other-ws` running `nohup tools/review.sh T9 ... &` at
  10:00:01, closer to codex session c1 than the parent's call; it must not be joined.
- `rollout-c1.jsonl` (10:00:03), `rollout-c3.jsonl` (10:05:03), `rollout-c4.jsonl`
  (10:08:03): the three codex sessions the parent launched (c1 through the widened
  window, c3 and c4 inside the loop's interval, shared launcher).
- `rollout-c2.jsonl` (10:00:20): a codex session in the window of call 1 after it claimed
  c1; stays unlaunched. `rollout-c5.jsonl` (10:17:30): 150 s after call 3, outside the
  window of a call that was not backgrounded; stays unlaunched.

The test gives each record a `worktree` (the same directory as the transcripts' cwd):
scripts are resolved from the Bash entry's cwd but must lie, after symlinks, under one
script root, the record's worktree when it exists on disk, else the workspace name under
`WORKSPACE_ROOT` (the test's second parameter points the worktrees at directories that
no longer exist). The `nohup` call is backgrounded, so it never joins by interval
containment: c1 is claimed through the widened window, one session per launch the
command text names.
