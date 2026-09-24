# Cache and origin fixtures (task L2: the link cache and codex origin corroboration)

A synthetic workspace and one build morning in it, for `tests/test_graph_cache.py`.
`__WS__` inside the transcripts is a placeholder the test replaces with the workspace
directory (`cache-ws/` here). The set is small on purpose: every `session_meta` variant
the corroboration distinguishes appears exactly once, and the cold and warm cache runs
are compared over the whole set.

- `cache-ws/`: the workspace directory (empty; the records point their worktree at it).
- `parent.jsonl`: the lead session in `cache-ws`. Four direct `codex exec ...` Bash
  calls at 2026-08-31 10:00, 10:05, 10:10 and 10:15, each returning a minute later.
  Every one launches exactly one rollout (it starts 2 s after the call, inside the
  interval).

Launched rollouts (one per call, in order):

- `rollout-exec.jsonl` (10:00:02): `session_meta` says `codex_exec` / `exec`, cwd in
  the workspace. Corroborates the edge.
- `rollout-cli.jsonl` (10:05:02): `codex-tui` / `cli`, an interactive start by its own
  report. Contradicts the edge (the edge stays; the contradiction is on it and counted).
- `rollout-nometa.jsonl` (10:10:02): no `session_meta` line at all. Origin missing;
  the absence is established by reading the whole rollout, so it carries tier
  `verified`.
- `rollout-late.jsonl` (10:15:02): `codex_exec` / `exec`, but the `session_meta` line
  is record 41 of 52, after 40 other records. The scan must read the whole rollout, not
  a prefix, to find it. Corroborates the edge.

Unlaunched rollouts (no Bash call anywhere near them):

- `rollout-nocwd.jsonl` (10:30:00): `codex_exec` / `exec` with no `cwd`. Counted as
  "exec, cwd unknown", never as "outside the workspace".
- `rollout-unsupported.jsonl` (10:31:00): originator `bb`, source `vscode`, a pair the
  corroboration does not recognise (seen in real rollouts, meaning unknown). Counted as
  unsupported, with the pair in `Graph.meta`, never guessed to be interactive.
- `rollout-elsewhere.jsonl` (10:32:00): `codex_exec` / `exec` from a cwd outside the
  workspace.
- `rollout-unscanned.jsonl` (10:33:00): `codex_exec` / `exec` in the workspace: some
  session started it and none in scope did, "launched by an unscanned session".
- `rollout-vscode.jsonl` (10:34:00): `codex_vscode` / `vscode`, interactive: "not
  launched".
- `rollout-subagent.jsonl` (10:35:00): `codex_exec` with `source` a dict
  (`{"subagent": {"thread_spawn": ...}}`, the CLI's own subagent spawn), also seen in
  real rollouts. Unsupported pair, counted, source kept as written.
