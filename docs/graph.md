# The workflow graph

`loopmath.graph` turns the session logs loopmath already ingests into a graph of who did what for
whom: which session started which, which file one session wrote and another read, and what
each session was for. It is built from structure in the logs alone. No model is asked
anything; every relation is a fact found in a transcript, or a rule applied to one, and the
graph says which.

## What the graph is

Input is the list of priced run records that `ingest.parse_all`, `grade.grade_all` and
`price.price_all` produce (one plain dict per session: `run_id`, `session_path`, `harness`,
`workspace`, `model`, `effort`, `ts`, `wall_s`, `tokens`, `usd`). Output is a `Graph`
(`src/loopmath/graph/schema.py`) with three kinds of object:

- **Node**: one session. Nodes are never invented; every node is a session file on disk.
  `source` says what kind of file: `top` (a Claude Code session started by a person or a
  CLI), `subagent` (a transcript under `<session>/subagents/`), `codex` (a codex rollout),
  or `external` (a session outside the requested workspaces that launched sessions inside
  them; added so the launch has a visible origin, it carries no pricing). Each node also
  has `parent`, `role`, `phase`, `launched_by` and `spawn` when the extractor could
  establish them, and `None` when it could not.
- **Edge**: a directed relation between two nodes, of one of five kinds. The extractor
  discovers three of them.
  `spawn`: a Claude Code session started a subagent through the Task tool.
  `launch`: a session ran another harness's CLI (`codex exec`, `claude -p`) and a session
  of that harness started while the call was running.
  `artifact`: a node wrote a file that another node read later.
  The other two are the OCP core scheduling kinds, which only the reader supplies and the
  extractor never infers from a session.
  `dep`: the target cannot start before the source settles.
  `fan_in`: the source is a member of the target gate's acceptance input set.
- **Artifact**: a file path with its `producer` (first writer), `writers`, `consumers`
  (readers other than the writer, after a write), write and read counts, and a `kind`
  guessed from the path (`plan`, `review`, `test`, `config`, `doc`, `code`).

Artifact recording adds nullable `bytes`, `lines_added`, `lines_removed`, `language`, and `tests_touched` values with sibling evidence tiers, a `fate` of `kept`, `edited`, `reverted`, `deleted`, or the default `unknown` with `fate_tier`, and a free-form `meta` object that core graph logic ignores and consumers may ignore. `bytes_tier` is `verified` only when a successful transcript write records the complete UTF-8 content. `lines_added_tier` and `lines_removed_tier` are `verified` for diff lines recorded by the source and `heuristic` when the adapter computes a diff from recorded replacement text. `tests_touched_tier` is `heuristic`: its case-sensitive predicate counts distinct paths below a `test/` or `tests/` directory or with a basename matching `test_*.py`, `*_test.<lowercase letters>`, or `conftest.py`, and only `/` separates components. A write without an observed transaction fact does not fall back to testing its artifact path. Across the three 08-31 swarms, 54 of 603 artifacts have a non-null path-list observation for `tests_touched`, while 549 remain null: 110 because multiple writes collapse into one artifact and 439 because no observed transaction fact establishes the touched paths. `language` makes no language claim: it holds the basename text after the final dot, ASCII-lowercased, when that identifier matches `[a-z0-9][a-z0-9_+-]*`, and is null for an absent or invalid extension; `language_tier` is `heuristic`. A known `fate` tier follows the explicit source evidence and cannot be stronger than the represented write; absence of a later write leaves `fate` as `unknown` with a null tier. Because the graph still collapses repeated writes of one path into one artifact, any per-version measurement that cannot be assigned honestly remains null and its machine-readable reason is counted in `Graph.meta`. Artifact version lineage, `meta` export privacy, valuation, and UI are deliberately absent.

`Graph.meta` holds the counts: nodes by source, edges by kind and tier, roles, phases,
artifacts and consumed artifacts, total cost, and every exclusion the extractor made
(`unlinked_subagents`, `unmatched_launches`, `unlaunched_codex`, `external_launchers`,
`pending_writes_attributed`, `pending_writes_unresolved`, `usd_unpriced_nodes`,
`records_skipped_no_id_or_path`, `reads_without_writes`, `reads_before_first_write`,
`events_invalid_ts`, `nodes_missing_wall_s`, `nodes_missing_tokens`). Nothing is dropped
quietly: if a subagent cannot be linked to a parent it stays in the graph with `parent:
None` and the counter goes up; a record with no `run_id` or no `session_path` cannot be a
node and is counted; a Read of a path no node wrote cannot be an artifact edge and is
counted in `reads_without_writes`; a Read of a written path that happens before the path's
first observed write has no producer to join to and is counted in
`reads_before_first_write` (it is not a consumption, so it adds no consumer); a write or
read whose transcript timestamp does not parse never reaches the artifact join and is
counted in `events_invalid_ts` (it is not in `n_reads` or `n_writes` either); a node whose
record has no `wall_s` or `tokens` keeps `None` there (never `0.0` or `{}`) and is
counted. `reads_without_writes` and `reads_before_first_write` are disjoint (a path is
either written by some node or not); `events_invalid_ts` is a count of bad timestamps and
is independent of the other two, so a read of an unwritten path under a bad timestamp
appears in both `reads_without_writes` and `events_invalid_ts`.

`Graph.to_dict()` gives the JSON form (`{"dagr_graph": 1, "meta", "nodes", "edges",
"artifacts"}`), and `render.to_dot` gives Graphviz DOT.

## Evidence tiers

Every edge, every role and every artifact kind carries a tier, so a reader knows what to
trust before trusting it:

| Tier | Meaning | Examples in the skeleton |
|---|---|---|
| `verified` | both ends of the relation are mechanically present in the logs | a subagent's `.meta.json` names the Task tool_use id found in the parent; a Write of a path and a later Read of the same path; a model name taken from a Claude transcript |
| `heuristic` | a rule inferred it from timing, text or path layout | a codex session that started 3 s into a `codex exec` Bash call in the same workspace; a subagent under a session's directory whose meta names no known Task id; the lead role (most spawn and launch out-edges); a reviewer role from the word "review" in the launch command; an artifact kind from the file name |
| `reported` | a session or a harness asserted it and it is taken at face value | a subagent role from a declared `subagent_type` such as `Plan`; a codex session's model field |

Upgrading a tier without new mechanical evidence is a bug. Anything the extractor cannot
establish is `None` and counted, never filled with a default.

An artifact edge joins two scan events, a write and a read, and each of those carries its
own tier (a Write or Read tool call is `verified`; a write inferred from a Bash command
such as `codex exec -o review.md` is `heuristic`; a scan event with no tier at all is
treated as `heuristic`). The edge takes the weaker of the two, with `verified` strongest,
then `reported`, then `heuristic`: a heuristic write read by a verified Read is a
`heuristic` edge, and only verified plus verified gives a `verified` edge. The edge's
`detail` records both ends as `write_tier` and `read_tier`, next to `path` and `lag_s`,
so a reader can see which end holds the edge back.

Roles today are `lead` (the top-level session with the most spawn and launch out-edges in
its workspace; `solo` when it has none), `planner`, `dev` and `reviewer` for subagents and
codex sessions, `cli` for a launched codex session with no role words anywhere, and
`external` for launchers from other workspaces. A top-level session that is not the lead
has no role; its `role_evidence` says why. Phases are `build` (during the lead's run),
`post` (started more than 60 s after the lead ended, which is where the post-build
reviews of the 08-31 swarms sit) and `external` (launched from outside the workspace).
A phase is an inference, so every node with a phase carries `phase_tier: "heuristic"`;
a node whose start time or lead is unknown has `phase: None` and `phase_tier: None`.

## The verb

The CLI verb `loopmath graph [--workspace W] [--ocp FILE] --format ocp|json|dot|run|html
[--out PATH] [--since DAYS|--all]` accepts repeatable native workspace and OCP v0.2
sources. At least one `--workspace` or `--ocp` is required. An OCP-only invocation does
not scan local harness logs. `loopmath analyze --ocp FILE` likewise reads only the supplied
documents; combine it with an explicit `--logs PATH` to include native log records too.
The graph is also reached from Python:

```python
import json

from loopmath import ingest, grade, price
from loopmath.graph import extract, to_dot

records, _ = ingest.parse_all(None, limit=None, use_cache=True, progress=None, since_days=3)
records, _ = grade.grade_all(records)
records, _ = price.price_all(records, price.load_prices(None))
g = extract(records, workspaces=["dagr-swarm-a"])
open("swarm-a.json", "w").write(json.dumps(g.to_dict(), indent=1))
open("swarm-a.dot", "w").write(to_dot(g))          # then: dot -Tsvg swarm-a.dot > swarm-a.svg
```

## HTML visualizer

`--format html` writes one self-contained page. Its scripts, styles, and graph data are
inline, so the page works without a network connection. The attempt table comes before
the pictures and sorts on each of its ten columns. Shared role and model filters update
the table and all three views.

The foldable views appear as swimlanes, force, and cost curve, in that order. Every edge
kind has one line style, named in each view's legend: dotted for `dep`, dash-dot for
`fan_in`, solid for `spawn`, dashed for `launch`, and a gray arc for a handoff. Each view
has its own color and handoff controls. Swimlanes and cost curve keep artifact display
off until their artifact checkbox is selected. Force displays consumed artifacts as
clickable nodes on first load. Selecting a table row, attempt node, or artifact opens a
detail card, and pointing at a node, edge, or artifact shows its tooltip.
Attempt cards retain all fields from the source `GraphNode`. The `spawn` and
`launched_by` mappings list every recorded key and value rather than a selected
summary, and the viewer's declared attempt-field omission list is empty.

For example:

```console
loopmath graph --ocp run.ocp.json --format html --out out/run.html --quiet
```

`loopmath view` remains a separate, unavailable verb. It is not an alias for this format.

## OCP v0.2 reader round trip

`loopmath.ingest.ocp.from_ocp` preserves every attempt as a Graph node, including model,
effort, the four billable token streams and cache-retention buckets, USD, start and end
time (as start plus wall-clock duration), session correlate, origin, role, phase, edges
with their kinds, evidence tiers and evidence, and artifacts. One-attempt logical nodes
keep their OCP node ids. The reader also carries the native emitter's node provenance in
`ext["dev.dagr.graph"]` so metadata-only title withholding remains reproducible without
inventing or restoring the withheld transcript text.

### Measured A/B/C native-emitter contract

The native-emitter contract exercised by A/B/C is canonically lossless. The acceptance
test loads the checked-in, unedited `Graph.to_dict()` outputs from the actual A, B and C
swarm evaluation, calls
`to_ocp`, reads that document with `from_ocp`, and calls `to_ocp` again. Canonical JSON
uses sorted object keys and compact separators; the two strings and all entity ids are
identical. The test pins these entity counts in `(groups, nodes, edges, attempts,
artifacts, events)` order:

| Swarm | Native Graph `(nodes, edges, artifacts)` | OCP entity counts | Canonical comparisons |
|---|---:|---:|---:|
| A | `(111, 914, 434)` | `(2, 111, 914, 111, 434, 220)` | `1/1` equal |
| B | `(42, 319, 120)` | `(2, 42, 319, 42, 120, 82)` | `1/1` equal |
| C | `(4, 27, 49)` | `(2, 4, 27, 4, 49, 6)` | `1/1` equal |

These are fixture measurements, pinned by
`test_native_eval_swarm_ocp_round_trip_is_canonically_lossless`; they are not a claim
that an arbitrary producer's OCP document fits the smaller Graph model.

The committed native fixtures are historical writer artifacts, so reader enrichment
creates one deliberate metadata disagreement in them. Swarm A has 87 source metadata
keys and 96 loaded keys: 86 ordinary `Graph.meta` keys plus 10 preserved source states.
B and C each have 83 source keys and 92 loaded keys: 82 ordinary keys plus the same 10
preserved states. Nine states are source absences for the closed set of `ocp_*` token
diagnostics; the tenth is the source and recomputed values of `nodes_missing_tokens`.
Unambiguously, swarm A has source `1` and reader `111` of 111 nodes; swarm B has
source `1` and reader `42` of 42 nodes; swarm C has source `1` and reader `4` of 4
nodes. The reader values are the measurements we believe because every node lacks at
least one of the six required token streams; the historical writer counted only the
single node whose entire cost record was absent. Source fidelity nevertheless restores
that known-wrong source value when an otherwise unchanged imported graph is re-emitted.
The historical snapshots remain unchanged deliberately: they are the only committed
inputs exercising this ten-state compatibility path, and regenerating them would delete
that evidence.

Fresh A/B/C documents generated by the fixed writer retain those respective source
and loaded key counts. Swarm A has source `111` and reader `111` of 111 nodes; swarm B has source
`42` and reader `42` of 42 nodes; swarm C has source `4` and reader `4` of 4 nodes.
The fresh round trip re-derives 87 of 96 loaded metadata keys for A and carries nine
source-absence states; it re-derives 83 of 92 for both B and C and carries the same
nine states. The complete observed delta is exactly the nine named token-diagnostic
absences; the historical value state activates only when a source counter actually
disagrees.

This restoration is private, non-contract emission state. `_ocp_source_meta` contains
the closed nine absence states for these current-writer documents and adds the one
closed value state only for a historical disagreement. `_ocp_loaded_meta` is a full
equality guard but is never emitted. Both fields are excluded from `Graph.to_dict()`
and Graph equality. If current `Graph.meta` differs from the guard, preservation
disengages and the current mapping is emitted exactly: a deleted preserved key stays
dropped, an explicit `None` is emitted as JSON null, a changed value stays changed,
and no metadata value is recomputed. Restoring `Graph.meta` exactly to the loaded guard
re-enables source preservation. All modeled OCP fields outside this metadata object
are regenerated from graph content; the pre-existing `_ocp_node_ext` remains the
separate private path for native-emitter node-extension provenance.

### Generic OCP fields that do not survive Graph re-emission

The table below is exhaustive for the field families in `ocp-v0.2.schema.json`. "Does
not survive" means the value or shape is not guaranteed after `to_ocp(from_ocp(doc))`;
some values can coincide with the emitter's derived value. The Graph retains node ids
for the one-attempt native shape, harness, source, model raw name and tier, effort,
workspace, start plus duration, represented token streams and USD, role value, tier and
evidence, phase value and tier, graph-representable edges, and the core artifact
relationship. Everything else that is lossy is named here.

| OCP entity | Fields that do not survive generically | Reason |
|---|---|---|
| `producer`, `privacy` | Every member, including both `ext` objects | They are `to_ocp` call options, not Graph fields. Defaults are regenerated unless the caller supplies them again. |
| `run` | `id`, `title`, explicit `started_at` and `ended_at`, `labels`, `ext`; exact `workspace` text | The emitter derives identity, title, bounds and workspace from the represented nodes. Run annotations have no Graph slot. |
| `groups[]` | Exact membership, `title`, `parent`, `ext`; an unreferenced group and a group id distinct from workspace | Graph has a flat workspace string per node and no group entity or hierarchy. Groups are regenerated from those workspace strings. |
| `nodes[]` | `kind`, `title`, `state`; `labels` except `harness` and `source`; general `ext` | Graph stores role rather than node kind and has no task title, state or general node annotation slots. The native `dev.dagr.graph` title-provenance extension is the deliberate exception. |
| Logical node and attempt identity | Arbitrary `attempt.id`, retry-preserving `node`, `n`, and node cardinality for zero or multiple attempts | Graph has one node per attempt. A logical node with retries expands to attempt-id Graph nodes; re-emission then makes one logical node and one `.a1` attempt per Graph node. A node-level edge without attempt endpoints is attached to the latest input ordinal on read. |
| `edges[]` | Producer-supplied `from_attempt` and `to_attempt` identity after retry expansion; general `ext` | Attempt endpoints are regenerated from Graph node ids. Graph keeps only detail it understands. Native launch-command and later-writer routing extensions are understood and regenerated. |
| `attempts[]` | `actor`, `cause`, `status`, `outcome`, `labels`, general `ext` | Graph has no acceptance, retry-cause, actor or general attempt-annotation model. The emitter reports one initial, settled-unverified attempt for each Graph node. Native spawn and launch provenance in `dev.dagr.graph` is understood. |
| `attempts[].model` | `id`, `family`, `provider`, `ext`; the distinction between an `id`-only model and `raw` | Graph has one model string plus a tier. The reader prefers `raw`, falls back to `id`, and the emitter writes that string as `raw`. |
| `attempts[].origin` | Producer-authored `evidence`, `ext`, and origin details not expressible as Graph launch provenance | Graph represents workspace and a limited `launched_by` record. The emitter derives its origin claim and evidence from graph edges. |
| `attempts[].role`, `phase` | Both `ext` objects; producer-authored phase `evidence`; role evidence changed by the metadata-only privacy reducer | Graph has no extension slots for these labels. Phase evidence is derived from the phase value. Metadata-only output retains a rule and a character count, not transcript text. |
| `attempts[].cost` | `reasoning_tokens`, `requests`, `basis`, `ext`; any other future stream | Graph has the six mapped token and cache-bucket fields plus USD. Re-emitted Graph costs use `basis: measured`; an allocated basis cannot be represented. |
| `attempts[]` time and session | Timestamp offset and precision finer than milliseconds; an `ended_at` without a valid start; a `session` correlate containing `/` | Graph stores start plus nonnegative duration. The emitter normalizes UTC timestamps to milliseconds and emits the basename of `session_path`. |
| `artifacts[]` | An `id` distinct from the full path, `labels`, `kind.evidence`, general `ext` outside `dev.dagr.artifact`, exact timestamp offset or sub-millisecond precision | Graph identity is the full path and kind has only value plus tier. The emitter re-derives artifact id, presentation path and kind evidence. Native long-path `dev.dagr.graph` provenance and the Q7 `dev.dagr.artifact` measurements and free-form `meta` are understood. |
| `events[]` | Every producer-supplied event and all event `actor`, `detail`, `ext` data | Graph has no event log. The emitter synthesizes only attempt-started and attempt-settled events from node times. |
| Top-level `ext` | Every namespace except the native `dev.dagr.graph` structure (`meta`, recomputed emitter counters, and held-edge records) | Graph has `meta`, not a generic OCP extension store. Unknown namespaces are intentionally ignored. |

The shipped generic `swarm-v02.ocp.json` exercises these limits. Its 6 nodes, 7 edges,
6 attempts and 3 artifacts remain represented, while its 21 input events become 12
derived attempt events. Its run label, 6 request counts, 6 each of model id, family and
provider, 1 reasoning-token count, artifact-specific extension, and outcome receipts are
not Graph fields. This measurement is pinned beside the reader tests; the native A/B/C
equality above is the stronger contract used by the A/B/C evaluation and swarm gate.

`workspaces` restricts the nodes to those workspaces; launchers from other workspaces
still appear, as `external` nodes, when they started a session inside the scope. Without
the argument every record becomes a node and no session is external.

In the DOT output, node fill says the source (blue top, green subagent, amber codex, grey
external), a dashed border says phase `post`, a grey border says the role is heuristic.
Edges: solid for spawn, dashed for launch (labelled with the lag from the command to the
session start), dotted through a note-shaped file node for artifacts. Dark edges are
`verified`, grey ones are not, and that holds for the dotted artifact connectors too: the
file-to-consumer connector takes the colour of the artifact edge it stands for, and the
producer-to-file connector is dark when any edge out of that file is verified. A file
nobody consumed (drawn only with `artifacts="all"`) has no edge to take a tier from, so
its connector is grey. By default only artifacts with at least one consumer are drawn
(`to_dot(g, artifacts="all")` draws every written file, `"none"` skips them).

## Known limits of the skeleton

These are what the build plan's remaining tasks address; until they land, the counts in
`meta` show the gap rather than hiding it.

- **Files written through Bash are invisible.** Only the Write, Edit, MultiEdit,
  NotebookEdit and Read tools produce writes and reads. A plan written with `cat > f
  <<'EOF'` or a review produced by `codex exec -o f` is not an artifact, which is why the
  swarms' plan and review documents show as 0 artifacts today (task A1 adds Bash
  patterns, A2 the codex side, A3 kinds and a git fallback).
- **Codex sessions have no writes or reads yet** (A2), so an artifact edge never ends at a
  codex node and a read a codex session makes of a Claude-written file is not seen.
- **Launches through scripts and backgrounded commands are missed.** The join needs the
  CLI name in the Bash command text and a session starting inside the call's interval (or
  within 5 s before to 120 s after its start, same workspace only). `nohup tools/review.sh
  &` launches codex from inside the script and returns at once, so it does not match;
  those sessions are counted in `unlaunched_codex` (L1, L2).
- **Cross-workspace launches must name the workspace** in the command (`cd
  ~/work/swarm-a && codex exec ...`); this restriction exists because without it
  reviewers attached to a review launched into another swarm 37 s earlier.
- **The external launcher node is thinner than the others.** It is built from the
  launcher's record only, so it carries no `wall_s`, `tokens` or `usd` (they are `None`
  and counted in `nodes_missing_wall_s`, `nodes_missing_tokens` and
  `usd_unpriced_nodes`); its phase `external` has tier `heuristic` like every phase.
- **Roles are coarse.** Lead, planner, dev, reviewer, solo, cli, external. Repair versus
  implement, send-backs and approvals, and boundaries inside one long session are what
  the labeling eval (section 8 of the spec) exists to measure. The lead rule is a
  heuristic with ties broken by output tokens then wall clock; a workspace with two
  independent top-level sessions gets one lead and one unlabeled session.
- **The 60 s post-build gap** is a fixed rule (`labels.POST_BUILD_GAP_S`); a session that
  started during the lead's run but did review work is phase `build`.
- **Role evidence text repeats a declared type** when the Task's `subagent_type` and the
  meta's `agentType` agree (`declared type 'Plan Plan'`). Cosmetic; the tier and role are
  right.

## Reading the swarm eval

`tools/eval_swarms.py [DAYS] [--graph-source ocp|native]` runs the extractor over the
three 08-31 swarms (`dagr-swarm-a`, `-b`, `-c`), writes graph, OCP and DOT documents
under `tools/out/`, and scores the result against `tools/ground-truth-swarms.json`.
The default `ocp` source scores the Graph returned by the OCP reader after native
ingest and `to_ocp`; `native` retains the direct extractor path. For each swarm it
prints:

- a header with the node count and `meta` (everything except the role and phase tables),
  so exclusions are visible before any check: `unlinked_subagents`,
  `unmatched_launches`, `unlaunched_codex`, `external_launchers`;
- a `phases` line counting `source/phase` pairs and the external launchers found, with
  one line per external launch saying how it was matched and the command that did it;
- `role tiers (build)` and `usd by role (build)`, so the cost by role is read together
  with how much of it rests on heuristic labels;
- the codex launch join as `linked/total`, plus the number of launch commands that
  matched no session;
- artifacts by kind and consumed by kind, and the plan and review documents found;
- `[PASS]` or `[FAIL]` lines for each check, ending with a summary of checks passed per
  swarm.

The checks that matter most: session counts equal the raw counts (nothing dropped); spawn
edges are `verified` for every subagent; the six post-build evaluation sessions are
separated from the build (phase `post` or `external`); role tables match the published
ones. The done bars in the spec (sections 3 and 4) are read off the launch-join and
artifact lines: launch edges must reach every build codex run, and the swarms' plan and
review documents must appear as artifacts with a producer and at least one consumer.

## Leaderboard: model labelers on swarm B

Measured 2026-09-01 and 2026-09-02 by `tools/grid.sh --out-dir <dir> <model> <effort>` on
the **FD-B** `out/dataset-b.jsonl` (195 node items, 1260 edge items, 10 batches of up to 20 nodes). Every
measurement in this section is labeled **FD-B**, meaning fixture-developed on the swarm B
development fixture: the prompts were developed against this fixture. The original rows are
in `grid/a2`; prompt v2 is in `grid/v2`; prompt v3 and its strict control are in `grid/v3`
and `grid/v3-strict`; prompt v4 is in `grid/v4`. There are 29 development and replication
score receipts under `grid/**` and 5 held-out receipts under
`tests/fixtures/graph/heldout/results/`, as recomputed by the receipt utility. Each row links
its exact score object.

Every outcome cell below is `coverage; correctness; majority baseline`. Coverage is the
number of gold-labeled items with a non-null prediction over all gold-labeled items.
Correctness is the number right among those covered items. The baseline is the accuracy at
full coverage from always predicting the most frequent gold class in that row's score
object. Thus an abstaining run cannot look identical to a full-coverage run. Role has 194
gold labels, fine role 149, edge 151 gold parents, send-back and approved 39 gold labels,
and exact boundary-count classification has 195 gold-labeled nodes. Timed boundary recall
discussed below uses 10 gold boundary events at 300 s tolerance. `root` in the edge baseline
is the fixture root id `cc_cc6d29ea-d99d-4789-9453-5c4601645356`. Cost and wall time are
loopmath's pricing of the ten labeler sessions; every listed run priced all ten.

The headline is that role barely moves when the attempt block disappears entirely. On this
single-run fixture, v3 and `v3-strict` both score 192/194 on terra high, and score 179/194
and 178/194 on luna low (**FD-B**). The v3 role gain therefore does not depend on either the fields
removed from v2 or the remaining id-and-time block. The outcome gains do not survive the
same honesty test. Relative to v2, v3 terra send-back recall falls from 25/25 to 1/25 and
approval recall from 29/31 to 0/31; luna falls from 14/25 to 9/25 and from 29/31 to 12/31.
The flat role result beside those large outcome drops is the central finding: the role gain
is real on this fixture, while the v2 send-back and approval gains came from supplied label
fields and their join (**FD-B**).

| configuration and exact score object | role | fine role | edge | send-back | approved | boundary count | usd | wall s |
|---|---|---|---|---|---|---|---:|---:|
| **FD-B** `claude-haiku-4-5-default`<br>[`grid/a2/claude-haiku-4-5-default.scores.json`](../grid/a2/claude-haiku-4-5-default.scores.json) | cov 194/194; correct 150/194; base dev 104/194 | cov 149/149; correct 141/149; base review 85/149 | cov 151/151; correct 151/151; base root 107/151 | cov 8/39; correct 3/8; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | cov 195/195; correct 184/195; base count 0 194/195 | 1.162 | 1158 |
| **FD-B** `gpt-5.6-luna-high`<br>[`grid/a2/gpt-5.6-luna-high.scores.json`](../grid/a2/gpt-5.6-luna-high.scores.json) | cov 194/194; correct 150/194; base dev 104/194 | cov 149/149; correct 129/149; base review 85/149 | cov 151/151; correct 151/151; base root 107/151 | cov 0/39; correct 0/0; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | cov 195/195; correct 181/195; base count 0 194/195 | 0.144 | 1313 |
| **FD-B** `gpt-5.6-luna-low`<br>[`grid/a2/gpt-5.6-luna-low.scores.json`](../grid/a2/gpt-5.6-luna-low.scores.json) | cov 192/194; correct 151/192; base dev 104/194 | cov 147/149; correct 138/147; base review 85/149 | cov 149/151; correct 149/149; base root 107/151 | cov 1/39; correct 0/1; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | cov 193/195; correct 155/193; base count 0 194/195 | 0.088 | 461 |
| **FD-B** `gpt-5.6-luna-medium`<br>[`grid/a2/gpt-5.6-luna-medium.scores.json`](../grid/a2/gpt-5.6-luna-medium.scores.json) | cov 193/194; correct 150/193; base dev 104/194 | cov 148/149; correct 137/148; base review 85/149 | cov 150/151; correct 150/150; base root 107/151 | cov 0/39; correct 0/0; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | cov 194/195; correct 158/194; base count 0 194/195 | 0.102 | 682 |
| **FD-B** `gpt-5.6-luna-xhigh`<br>[`grid/a2/gpt-5.6-luna-xhigh.scores.json`](../grid/a2/gpt-5.6-luna-xhigh.scores.json) | cov 193/194; correct 148/193; base dev 104/194 | cov 148/149; correct 123/148; base review 85/149 | cov 150/151; correct 150/150; base root 107/151 | cov 8/39; correct 7/8; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | cov 194/195; correct 181/194; base count 0 194/195 | 0.229 | 2587 |
| **FD-B** `gpt-5.6-terra-high`<br>[`grid/a2/gpt-5.6-terra-high.scores.json`](../grid/a2/gpt-5.6-terra-high.scores.json) | cov 194/194; correct 170/194; base dev 104/194 | cov 149/149; correct 123/149; base review 85/149 | cov 151/151; correct 151/151; base root 107/151 | cov 1/39; correct 1/1; base true 25/39 | cov 1/39; correct 0/1; base true 31/39 | cov 195/195; correct 194/195; base count 0 194/195 | 0.945 | 566 |
| **FD-B** `gpt-5.6-terra-low`<br>[`grid/a2/gpt-5.6-terra-low.scores.json`](../grid/a2/gpt-5.6-terra-low.scores.json) | cov 194/194; correct 151/194; base dev 104/194 | cov 149/149; correct 130/149; base review 85/149 | cov 151/151; correct 151/151; base root 107/151 | cov 8/39; correct 7/8; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | cov 195/195; correct 193/195; base count 0 194/195 | 0.859 | 470 |
| **FD-B** `gpt-5.6-terra-medium`<br>[`grid/a2/gpt-5.6-terra-medium.scores.json`](../grid/a2/gpt-5.6-terra-medium.scores.json) | cov 194/194; correct 153/194; base dev 104/194 | cov 149/149; correct 122/149; base review 85/149 | cov 151/151; correct 151/151; base root 107/151 | cov 0/39; correct 0/0; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | cov 195/195; correct 193/195; base count 0 194/195 | 0.895 | 508 |
| **FD-B** `gpt-5.6-terra-xhigh`<br>[`grid/a2/gpt-5.6-terra-xhigh.scores.json`](../grid/a2/gpt-5.6-terra-xhigh.scores.json) | cov 194/194; correct 153/194; base dev 104/194 | cov 149/149; correct 118/149; base review 85/149 | cov 151/151; correct 151/151; base root 107/151 | cov 12/39; correct 1/12; base true 25/39 | cov 12/39; correct 11/12; base true 31/39 | cov 195/195; correct 189/195; base count 0 194/195 | 1.274 | 1089 |
| **FD-B** `gpt-5.6-terra-high` (`prompt v3`)<br>[`grid/v3/gpt-5.6-terra-high.scores.json`](../grid/v3/gpt-5.6-terra-high.scores.json) | cov 194/194; correct 192/194; base dev 104/194 | cov 149/149; correct 135/149; base review 85/149 | cov 151/151; correct 151/151; base root 107/151 | cov 8/39; correct 1/8; base true 25/39 | cov 1/39; correct 0/1; base true 31/39 | cov 195/195; correct 195/195; base count 0 194/195 | 0.964 | 603 |
| **FD-B** `gpt-5.6-luna-low` (`prompt v3`)<br>[`grid/v3/gpt-5.6-luna-low.scores.json`](../grid/v3/gpt-5.6-luna-low.scores.json) | cov 187/194; correct 179/187; base dev 104/194 | cov 143/149; correct 142/143; base review 85/149 | cov 144/151; correct 144/144; base root 107/151 | cov 21/39; correct 12/21; base true 25/39 | cov 17/39; correct 12/17; base true 31/39 | cov 188/195; correct 187/188; base count 0 194/195 | 0.087 | 441 |
| **FD-B** `gpt-5.6-terra-high` (`prompt v3-strict`)<br>[`grid/v3-strict/gpt-5.6-terra-high.scores.json`](../grid/v3-strict/gpt-5.6-terra-high.scores.json) | cov 194/194; correct 192/194; base dev 104/194 | cov 149/149; correct 133/149; base review 85/149 | cov 150/151; correct 150/150; base root 107/151 | cov 5/39; correct 1/5; base true 25/39 | cov 5/39; correct 4/5; base true 31/39 | cov 195/195; correct 191/195; base count 0 194/195 | 0.977 | 615 |
| **FD-B** `gpt-5.6-luna-low` (`prompt v3-strict`)<br>[`grid/v3-strict/gpt-5.6-luna-low.scores.json`](../grid/v3-strict/gpt-5.6-luna-low.scores.json) | cov 183/194; correct 178/183; base dev 104/194 | cov 139/149; correct 137/139; base review 85/149 | cov 140/151; correct 140/140; base root 107/151 | cov 0/39; correct 0/0; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | cov 184/195; correct 170/184; base count 0 194/195 | 0.086 | 451 |
| **FD-B** `gpt-5.6-terra-high` (`prompt v2`)<br>[`grid/v2/gpt-5.6-terra-high.scores.json`](../grid/v2/gpt-5.6-terra-high.scores.json) | cov 193/194; correct 190/193; base dev 104/194 | cov 148/149; correct 129/148; base review 85/149 | cov 150/151; correct 150/150; base root 107/151 | cov 38/39; correct 38/38; base true 25/39 | cov 39/39; correct 36/39; base true 31/39 | cov 194/195; correct 193/194; base count 0 194/195 | 1.085 | 568 |
| **FD-B** `gpt-5.6-luna-low` (`prompt v2`)<br>[`grid/v2/gpt-5.6-luna-low.scores.json`](../grid/v2/gpt-5.6-luna-low.scores.json) | cov 193/194; correct 182/193; base dev 104/194 | cov 148/149; correct 146/148; base review 85/149 | cov 150/151; correct 150/150; base root 107/151 | cov 36/39; correct 26/36; base true 25/39 | cov 36/39; correct 29/36; base true 31/39 | cov 194/195; correct 193/194; base count 0 194/195 | 0.101 | 455 |
| **FD-B** `gpt-5.6-terra-high` (`prompt v4`, afternoon)<br>[`grid/v4/gpt-5.6-terra-high.scores.json`](../grid/v4/gpt-5.6-terra-high.scores.json) | cov 194/194; correct 186/194; base dev 104/194 | cov 149/149; correct 70/149; base review 85/149 | cov 151/151; correct 151/151; base root 107/151 | cov 25/39; correct 17/25; base true 25/39 | cov 19/39; correct 7/19; base true 31/39 | cov 195/195; correct 195/195; base count 0 194/195 | 2.599 | 697 |
| **FD-B** `gpt-5.6-luna-low` (`prompt v4`, afternoon)<br>[`grid/v4/gpt-5.6-luna-low.scores.json`](../grid/v4/gpt-5.6-luna-low.scores.json) | cov 186/194; correct 175/186; base dev 104/194 | cov 141/149; correct 83/141; base review 85/149 | cov 143/151; correct 143/143; base root 107/151 | cov 17/39; correct 15/17; base true 25/39 | cov 11/39; correct 1/11; base true 31/39 | cov 187/195; correct 187/187; base count 0 194/195 | 0.272 | 458 |

**FD-B** The v3 timed boundary cells are not detection achievements. Attempt starts supply the gold
times directly, and their 10/10 disappears to 0/10 for both models when `v3-strict`
withholds the block. Treat 0/10 as the honest timed result. Without times, luna retains
6/10 count-based boundary recall (6/15 precision), while terra retains 0/10 (0/0). Timed
event recall has no finite negative-event class, so a majority-class baseline is not defined
for it. The table therefore baselines exact boundary-count classification per node and keeps
timed event recall separate.

**FD-B** Prompt v3 was measured with `tools/grid.sh --prompt-version v3 --out-dir grid/v3` at code
commit `a890d97`. It keeps the prompt v2 role guard, but its per-node attempt rows contain
only `session_id`, `started_at` and `ended_at`. Task ids, attempt numbers, derived attempt
ids, results, every cause field and the dataset-wide ledger are absent. For every attempt
and each gold field, the audit constructs two contract-legal ledgers that hold all three
retained values byte-identical while varying the gold label. The two role
measurements pass the predeclared default thresholds: terra high is 0.990 (192/194), above
0.906, and luna low is 0.923 (179/194), above 0.814. Prompt v3 is therefore the default;
v1, v2 and `v3-strict` remain selectable. These were the first and only measurements of
the two fixed variants, and no prompt bytes changed after a score. They are
untuned-within-this-v3-comparison, not globally held out: v3 inherits the v2 role guard,
which was developed during the previous night's work on the same swarm B fixture.

**FD-B** The `v3-strict` control was measured with `--out-dir grid/v3-strict`. It retains the same
role guard but removes the attempt block entirely. Its role results, 0.990 (192/194) for
terra and 0.918 (178/194) for luna, show that the role gain survives with no structured
attempt evidence. Under v3, leak-supplied timed boundary recall is 10/10 for both models
because the prompt supplies the attempt starts that define the gold boundaries; all 10
boundaries are on one 11-attempt session. Under `v3-strict`, honest timed boundary recall
is 0/10 for both models; luna's count-based boundary recall is 6/10 and terra's is 0/10.

**FD-B** The v3 columns still need two finite-fixture caveats. Its 49 projected attempt rows are all
distinct, which permits fixture lookup but is not itself leakage, and the presence of an
attempt block identifies the 39 sessions scored for send-back and approval. Also, a fitted
start-time rank rule happens to reproduce approval 49/49 on this fixture. These are
selection and recency shortcuts on 49 rows, not proof that the gold label is a function of
the projection. The same audit reports 48/49 for the removed task-plus-number approval
shortcut and 31/49 for the removed `n > 1` send-back shortcut. Read v3 send-back and
approval as detection from session spans with these small-fixture shortcuts disclosed.
**FD-B** Terra send-back precision is 1/8 from 8 gold-scored predicted positives, and approval
recall is 0/31 with no predicted
true on a gold-labeled node. Luna send-back precision is 9/13 from 13 such positives, and
approval precision is 12/17 from 17. Luna's higher outcome recall, 9/25 send-back and 12/31
approval against terra's 1/25 and 0/31, is an inversion of their role ranking. The positive
counts show that terra is more conservative here, not that this one run establishes luna
as more capable. The cost-bound ratio remains not measured because 1
of the 195 labeled sessions is unpriced; all 40 v3 and `v3-strict` labeler sessions were
priced.

**FD-B** Prompt v2 was measured with `tools/grid.sh --prompt-version v2 --out-dir grid/v2` at code
commit `3b132ad`. It remains selectable as the historical circular control. V2 adds an
allowlisted view of contract attempt task ids, numbers, causes, results and times, both on
each node and in a deterministic dataset-wide ledger. Actor, model, gold labels and label
sources never enter the prompt. Prompt and prediction metadata account for all input
attempt records and now normalize and count every missing or invalid projected field and
cause subfield by reason. On the final v2 rows, send-back precision is 25/25 for terra and
14/15 for luna; approval precision is 29/30 for terra and 29/36 for luna.

**FD-B** The first v2 measurement, preserved under `grid/v2/initial/`, predated the explicit role
guard. Its terra-high role accuracy was 0.794 (154/194), below the 0.846 floor, although
send-back, approval and boundary recall were 25/25, 21/31 and 10/10. Luna-low in that
measurement scored role 0.742 (144/194), send-back 21/25, approval 12/31 and boundary
10/10 ([terra receipt](../grid/v2/initial/gpt-5.6-terra-high.scores.json),
[luna receipt](../grid/v2/initial/gpt-5.6-luna-low.scores.json)). This failure led to a narrow
role-preservation change. That pre-review measurement, preserved under `grid/v2/pre-review/`,
recovered terra role to 191/194
([exact receipt](../grid/v2/pre-review/gpt-5.6-terra-high.scores.json)), but the reviewer
correctly rejected it because actor and model fields from contract attempts leaked the
answers into the role task. The one review fix removed both fields, added the complete
attempt-evidence accounting above, and produced the two final rows.

What the numbers say, and what they do not:

- **FD-B** Orchestrator finding at the lane 2 merge: the prompt v2 send-back and approved rows are
  not detection performance, and must not be read as such. The gold labels for both fields
  are recoverable in full from the same contract-v3 attempt fields that prompt v2 places in
  the model prompt. Two rules reproduce them exactly on all 49 attempt records: send-back is
  true when an attempt's own cause type is `sent_back` or when its attempt id is cited as
  another attempt's cause reference (49/49); approved is true when the attempt result is
  `done` and no later attempt of the same task was opened with cause `sent_back` (49/49).
  Prompt v2 projects `result`, `cause.type` and `cause.ref` per attempt and states those two
  derivations in its own instructions, so a labeler that follows the instruction is applying
  a supplied rule to supplied fields. Receipt:

  ```text
  .venv/bin/python tools/check_label_circularity.py
  attempt records: 49
  send_back reproduced by cause.type and cause.ref alone: 49/49
  approved  reproduced by result and later cause.type alone: 49/49
  gold positives: send_back 25, approved 41
  ```

  Read those columns as rule following on structured input, and read the spread across
  models that way too: terra high at 25/25 applies the stated rule, luna low at 14/25 often
  does not. Detection of a send-back from session evidence alone remains unmeasured. Both
  fields are also scored over only the 39 nodes that carry attempt evidence, not over the
  194 role-labeled nodes, which is why their denominators differ from the role column.
- **FD-B** The prompt v2 role accuracy is not affected by that circularity, and this was checked
  rather than assumed. Splitting the 194 role-labeled nodes by whether they carry attempt
  evidence, terra high scores 39/39 on the 39 nodes with attempt evidence under both v1 and
  v2, and 131/155 = 0.845 under v1 against 151/155 = 0.974 under v2 on the 155 nodes with no
  attempt evidence at all. All 22 newly correct nodes and both newly wrong nodes fall in the
  no-evidence stratum, so the role gain comes from the v2 instruction rewrite and not from
  the attempt allowlist.

- **FD-B** The cost bound (labeler under 0.5 percent of the labeled work) is reported as not measured
  in every configuration: 1 of the 195 labeled sessions is unpriced, so the labeled total
  (883.13 usd) is a lower bound. Against that lower bound every run is under 0.15 percent.
- **FD-B** Under prompt v1, send-back, approved and boundary detection are near zero for every model: the labelers
  abstain on almost every send-back and approval field (a null prediction on a gold-labeled
  node counts as wrong, and the abstentions are counted per field in the scores file) and
  place boundaries on nodes that have no attempts. These are honest low numbers, not gaps in
  the scorer; the prompt (`v1`) gives the models little to detect them from.
- **FD-B** Edge recall is near 1.0 for every model because each batch prompt lists the extractor's
  parent candidate (195 candidates per batch); the labelers mostly confirm it.
- **FD-B** Per-dollar figures divide by the sum of ten small session prices, so they favour cheap
  models heavily; per-second figures favour low effort. Neither is a quality score.
- **FD-B** Default labeler (Analyst decision at G-V01, 2026-09-01 19:12): gpt-5.6-terra at effort
  high. Every configuration satisfies the cost bound against the lower-bound labeled total
  (all under 0.15 percent), so cost is not binding and the default goes to the best role
  accuracy (170/194). gpt-5.6-luna at low is the named cost alternative for bulk runs
  (151/194 at 0.09 usd). Caveat: one run per configuration, and the terra effort curve is
  not monotone (0.778, 0.789, 0.876, 0.789), so gaps under about 3 points are within noise;
  re-measure before ranking within that band. Neither the labeler CLI nor tools/grid.sh has
  a default model or effort (both take them as required arguments), so nothing was set in code.

### Replication (night of 09-01)

**FD-B** Runs 2 and 3 used `tools/grid.sh --out-dir grid/rep2` and `grid/rep3` on the same
1,455-item `out/dataset-b.jsonl`; run 1 is the `grid/a2` measurement above. Every score
file priced all 10 labeler sessions, 120 of 120 across the twelve runs. Counts are shown
for every outcome. These are also **FD-B** measurements. Cells use the same `coverage;
correctness; majority baseline` convention as the main leaderboard.

| configuration, run, and exact score object | role | fine role | send-back | approved | usd | wall s |
|---|---|---|---|---|---:|---:|
| **FD-B** `gpt-5.6-terra-high`, 1<br>[`grid/a2/gpt-5.6-terra-high.scores.json`](../grid/a2/gpt-5.6-terra-high.scores.json) | cov 194/194; correct 170/194; base dev 104/194 | cov 149/149; correct 123/149; base review 85/149 | cov 1/39; correct 1/1; base true 25/39 | cov 1/39; correct 0/1; base true 31/39 | 0.945 | 566 |
| **FD-B** `gpt-5.6-terra-high`, 2<br>[`grid/rep2/gpt-5.6-terra-high.scores.json`](../grid/rep2/gpt-5.6-terra-high.scores.json) | cov 194/194; correct 154/194; base dev 104/194 | cov 149/149; correct 143/149; base review 85/149 | cov 0/39; correct 0/0; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | 0.961 | 619 |
| **FD-B** `gpt-5.6-terra-high`, 3<br>[`grid/rep3/gpt-5.6-terra-high.scores.json`](../grid/rep3/gpt-5.6-terra-high.scores.json) | cov 194/194; correct 162/194; base dev 104/194 | cov 149/149; correct 132/149; base review 85/149 | cov 3/39; correct 0/3; base true 25/39 | cov 7/39; correct 7/7; base true 31/39 | 0.997 | 661 |
| **FD-B** `gpt-5.6-terra-medium`, 1<br>[`grid/a2/gpt-5.6-terra-medium.scores.json`](../grid/a2/gpt-5.6-terra-medium.scores.json) | cov 194/194; correct 153/194; base dev 104/194 | cov 149/149; correct 122/149; base review 85/149 | cov 0/39; correct 0/0; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | 0.895 | 508 |
| **FD-B** `gpt-5.6-terra-medium`, 2<br>[`grid/rep2/gpt-5.6-terra-medium.scores.json`](../grid/rep2/gpt-5.6-terra-medium.scores.json) | cov 194/194; correct 155/194; base dev 104/194 | cov 149/149; correct 143/149; base review 85/149 | cov 19/39; correct 19/19; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | 0.853 | 468 |
| **FD-B** `gpt-5.6-terra-medium`, 3<br>[`grid/rep3/gpt-5.6-terra-medium.scores.json`](../grid/rep3/gpt-5.6-terra-medium.scores.json) | cov 193/194; correct 153/193; base dev 104/194 | cov 148/149; correct 134/148; base review 85/149 | cov 0/39; correct 0/0; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | 0.874 | 472 |
| **FD-B** `gpt-5.6-luna-low`, 1<br>[`grid/a2/gpt-5.6-luna-low.scores.json`](../grid/a2/gpt-5.6-luna-low.scores.json) | cov 192/194; correct 151/192; base dev 104/194 | cov 147/149; correct 138/147; base review 85/149 | cov 1/39; correct 0/1; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | 0.088 | 461 |
| **FD-B** `gpt-5.6-luna-low`, 2<br>[`grid/rep2/gpt-5.6-luna-low.scores.json`](../grid/rep2/gpt-5.6-luna-low.scores.json) | cov 192/194; correct 149/192; base dev 104/194 | cov 147/149; correct 144/147; base review 85/149 | cov 6/39; correct 5/6; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | 0.089 | 483 |
| **FD-B** `gpt-5.6-luna-low`, 3<br>[`grid/rep3/gpt-5.6-luna-low.scores.json`](../grid/rep3/gpt-5.6-luna-low.scores.json) | cov 194/194; correct 152/194; base dev 104/194 | cov 149/149; correct 148/149; base review 85/149 | cov 1/39; correct 0/1; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | 0.089 | 498 |
| **FD-B** `gpt-5.6-luna-medium`, 1<br>[`grid/a2/gpt-5.6-luna-medium.scores.json`](../grid/a2/gpt-5.6-luna-medium.scores.json) | cov 193/194; correct 150/193; base dev 104/194 | cov 148/149; correct 137/148; base review 85/149 | cov 0/39; correct 0/0; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | 0.102 | 682 |
| **FD-B** `gpt-5.6-luna-medium`, 2<br>[`grid/rep2/gpt-5.6-luna-medium.scores.json`](../grid/rep2/gpt-5.6-luna-medium.scores.json) | cov 193/194; correct 150/193; base dev 104/194 | cov 148/149; correct 146/148; base review 85/149 | cov 0/39; correct 0/0; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | 0.100 | 665 |
| **FD-B** `gpt-5.6-luna-medium`, 3<br>[`grid/rep3/gpt-5.6-luna-medium.scores.json`](../grid/rep3/gpt-5.6-luna-medium.scores.json) | cov 194/194; correct 160/194; base dev 104/194 | cov 149/149; correct 147/149; base review 85/149 | cov 0/39; correct 0/0; base true 25/39 | cov 0/39; correct 0/0; base true 31/39 | 0.101 | 678 |

The summary rows pool the three receipt counts; cost and wall time remain arithmetic mean
(minimum to maximum). Each row links all three exact receipts.

| configuration and exact score objects | role | fine role | send-back | approved | usd | wall s |
|---|---|---|---|---|---:|---:|
| **FD-B** `gpt-5.6-terra-high`<br>[`grid/a2/gpt-5.6-terra-high.scores.json`](../grid/a2/gpt-5.6-terra-high.scores.json), [`grid/rep2/gpt-5.6-terra-high.scores.json`](../grid/rep2/gpt-5.6-terra-high.scores.json), [`grid/rep3/gpt-5.6-terra-high.scores.json`](../grid/rep3/gpt-5.6-terra-high.scores.json) | cov 582/582; correct 486/582; base dev 312/582 | cov 447/447; correct 398/447; base review 255/447 | cov 4/117; correct 1/4; base true 75/117 | cov 8/117; correct 7/8; base true 93/117 | 0.968 (0.945-0.997) | 615.3 (566-661) |
| **FD-B** `gpt-5.6-terra-medium`<br>[`grid/a2/gpt-5.6-terra-medium.scores.json`](../grid/a2/gpt-5.6-terra-medium.scores.json), [`grid/rep2/gpt-5.6-terra-medium.scores.json`](../grid/rep2/gpt-5.6-terra-medium.scores.json), [`grid/rep3/gpt-5.6-terra-medium.scores.json`](../grid/rep3/gpt-5.6-terra-medium.scores.json) | cov 581/582; correct 461/581; base dev 312/582 | cov 446/447; correct 399/446; base review 255/447 | cov 19/117; correct 19/19; base true 75/117 | cov 0/117; correct 0/0; base true 93/117 | 0.874 (0.853-0.895) | 482.7 (468-508) |
| **FD-B** `gpt-5.6-luna-low`<br>[`grid/a2/gpt-5.6-luna-low.scores.json`](../grid/a2/gpt-5.6-luna-low.scores.json), [`grid/rep2/gpt-5.6-luna-low.scores.json`](../grid/rep2/gpt-5.6-luna-low.scores.json), [`grid/rep3/gpt-5.6-luna-low.scores.json`](../grid/rep3/gpt-5.6-luna-low.scores.json) | cov 578/582; correct 452/578; base dev 312/582 | cov 443/447; correct 430/443; base review 255/447 | cov 8/117; correct 5/8; base true 75/117 | cov 0/117; correct 0/0; base true 93/117 | 0.089 (0.088-0.089) | 480.7 (461-498) |
| **FD-B** `gpt-5.6-luna-medium`<br>[`grid/a2/gpt-5.6-luna-medium.scores.json`](../grid/a2/gpt-5.6-luna-medium.scores.json), [`grid/rep2/gpt-5.6-luna-medium.scores.json`](../grid/rep2/gpt-5.6-luna-medium.scores.json), [`grid/rep3/gpt-5.6-luna-medium.scores.json`](../grid/rep3/gpt-5.6-luna-medium.scores.json) | cov 580/582; correct 460/580; base dev 312/582 | cov 445/447; correct 430/445; base review 255/447 | cov 0/117; correct 0/0; base true 75/117 | cov 0/117; correct 0/0; base true 93/117 | 0.101 (0.100-0.102) | 675.0 (665-682) |

**FD-B** Role correctness is materially noisier for terra high (0.794-0.876, an 8.2-point range) and
luna medium (0.777-0.825, a 4.8-point range) than for terra medium or luna low (**FD-B**).
Terra high still has the best pooled role correctness at 486/582 = 0.835 with full coverage,
but its second run fell below terra medium, while luna medium's third run rose to 160/194 =
0.825 (**FD-B**). The special-label recalls remain unstable: the
only nonzero terra-medium send-back result was 9/25 in run 2, and the only nonzero approval
result was 4/31 for terra high in run 3. The decision rule keeps `gpt-5.6-terra high` as the
default. The nearest configuration costing at least 5 times less is luna medium: its mean
cost is 0.101 usd versus 0.968 usd, 9.6 times less, but its pooled role correctness is
460/580 = 0.793, 4.2 points lower, outside the 3-point threshold. Luna low costs 10.9 times
less and has pooled role correctness 452/578 = 0.782, 5.3 points lower.
No default-change recommendation goes to the Analyst (**FD-B**).

### Held-out fixture (swarm A, n=110; HO-A v4 measured 09-02 21:48)

Every figure above this line was measured on the swarm B development fixture. This section
uses the fresh swarm A fixture: 110 sessions with gold annotated by gpt-5.6-sol high inside a
read-only sandbox from a sanitized evidence pack, never from run JSON or contract fields
(`tests/fixtures/graph/heldout/README.md`, `annotation-audit.json`). The fixture is held out at
the gold and measurement boundary, not at the raw-corpus boundary, because its sessions also
appear in `out/dataset-b.jsonl`. The measured input is
`tests/fixtures/graph/heldout/fresh-dataset.jsonl`. The 20-node `dataset.jsonl` in the same
directory is the quarantined, contaminated artifact from the afternoon run and was never
measured tonight. Predictions and score receipts are under
`tests/fixtures/graph/heldout/results/`.

The v4 commands, after sourcing `tools/env.sh`, were:

```
tools/grid.sh --prompt-version v4 --dataset tests/fixtures/graph/heldout/fresh-dataset.jsonl --out-dir grid/heldout/v4 gpt-5.6-terra high
tools/grid.sh --prompt-version v4 --dataset tests/fixtures/graph/heldout/fresh-dataset.jsonl --out-dir grid/heldout/v4 gpt-5.6-luna low
```

Each is one first-pass grid. Omitted predictions are scored as incorrect against the full 110
gold denominator; no omission was rerun or filled. Every row is labeled **HO-A**, meaning a
held-out measurement on the fresh swarm A fixture. As above, each outcome cell is `coverage;
correctness; majority baseline`, and a zero correctness denominator is undefined rather than
zero. Boundary correctness is exact boundary-count correctness per covered node. `root` in the
edge baseline is the fixture root id
`cc_cc6d29ea-d99d-4789-9453-5c4601645356`.

| prompt, model, effort, and exact score object | role | fine role | edge | boundary count | send-back | approved | labeler usd | cost ratio |
|---|---|---|---|---|---|---|---:|---:|
| **HO-A** v1, gpt-5.6-terra high<br>[`tests/fixtures/graph/heldout/results/v1-gpt-5.6-terra-high.scores.json`](../tests/fixtures/graph/heldout/results/v1-gpt-5.6-terra-high.scores.json) | cov 110/110; correct 110/110; base reviewer 68/110 | cov 110/110; correct 54/110; base send_back 59/110 | cov 107/107; correct 107/107; base root 107/107 | cov 110/110; correct 100/110; base count 0 98/110 | cov 36/98; correct 31/36; base true 92/98 | cov 32/90; correct 27/32; base false 82/90 | 0.546 | 0.000887 |
| **HO-A** v3, gpt-5.6-terra high<br>[`tests/fixtures/graph/heldout/results/v3-gpt-5.6-terra-high.scores.json`](../tests/fixtures/graph/heldout/results/v3-gpt-5.6-terra-high.scores.json) | cov 110/110; correct 109/110; base reviewer 68/110 | cov 105/110; correct 47/105; base send_back 59/110 | cov 107/107; correct 107/107; base root 107/107 | cov 110/110; correct 110/110; base count 0 98/110 | cov 18/98; correct 15/18; base true 92/98 | cov 15/90; correct 12/15; base false 82/90 | 0.568 | 0.000922 |
| **HO-A** v3, gpt-5.6-luna low<br>[`tests/fixtures/graph/heldout/results/v3-gpt-5.6-luna-low.scores.json`](../tests/fixtures/graph/heldout/results/v3-gpt-5.6-luna-low.scores.json) | cov 110/110; correct 110/110; base reviewer 68/110 | cov 110/110; correct 42/110; base send_back 59/110 | cov 107/107; correct 107/107; base root 107/107 | cov 110/110; correct 109/110; base count 0 98/110 | cov 10/98; correct 9/10; base true 92/98 | cov 0/90; correct 0/0; base false 82/90 | 0.050 | 0.000082 |
| **HO-A** v4, gpt-5.6-terra high<br>[`tests/fixtures/graph/heldout/results/v4-gpt-5.6-terra-high.scores.json`](../tests/fixtures/graph/heldout/results/v4-gpt-5.6-terra-high.scores.json) | cov 110/110; correct 110/110; base reviewer 68/110 | cov 110/110; correct 104/110; base send_back 59/110 | cov 107/107; correct 107/107; base root 107/107 | cov 110/110; correct 110/110; base count 0 98/110 | cov 96/98; correct 95/96; base true 92/98 | cov 88/90; correct 86/88; base false 82/90 | 1.502 | 0.002438 |
| **HO-A** v4, gpt-5.6-luna low<br>[`tests/fixtures/graph/heldout/results/v4-gpt-5.6-luna-low.scores.json`](../tests/fixtures/graph/heldout/results/v4-gpt-5.6-luna-low.scores.json) | cov 99/110; correct 99/99; base reviewer 68/110 | cov 99/110; correct 85/99; base send_back 59/110 | cov 96/107; correct 96/96; base root 107/107 | cov 99/110; correct 98/99; base count 0 98/110 | cov 80/98; correct 78/80; base true 92/98 | cov 56/90; correct 55/56; base false 82/90 | 0.143 | 0.000232 |

The **HO-A** v4 completeness figures, separated to avoid conflating output coverage with correctness,
are:

- **HO-A** Terra high predictions returned: 110 of 110; omissions: 0. Role coverage is
  110/110 and correctness is 110/110; fine-role coverage is 110/110 and correctness is
  104/110. Their majority baselines are reviewer 68/110 and send_back 59/110.
- **HO-A** Luna low predictions returned: 99 of 110; omissions: 11. Role coverage is
  99/110 and correctness is 99/99; fine-role coverage is 99/110 and correctness is 85/99.
  Their majority baselines are reviewer 68/110 and send_back 59/110. All 11 omissions are absent `labels` entries in otherwise
  nonempty, valid JSON answers: 3 in batch 0 and 8 in batch 4. Empty outputs: 0. Unparseable
  outputs or entries: 0.

Role is saturated and carries no useful signal on this fixture. **HO-A** v1 terra is 110/110,
v3 terra 109/110, v3 luna 110/110, and v4 terra 110/110; v4 luna's 99/110 is exactly its 11 missing
predictions. The held-out fixture therefore cannot evaluate v4's role guard and cannot show
that the **afternoon v4 claim, FD-B**, repaired the swarm B role regression from 192/194 to
186/194.

#### Outcome baselines and the results that survive them

These majority-class baselines are deliberately trivial. They make the class imbalance
visible next to every outcome claim.

| fixture, outcome, and exact score object | coverage | correctness | majority baseline | recall | precision |
|---|---:|---:|---|---:|---:|
| **HO-A** swarm A held-out send-back<br>[`tests/fixtures/graph/heldout/results/v4-gpt-5.6-terra-high.scores.json`](../tests/fixtures/graph/heldout/results/v4-gpt-5.6-terra-high.scores.json) | 98/98 | 92/98 | always true, 92/98 (0.939) | 92/92 (1.000) | 92/98 (0.939) |
| **HO-A** swarm A held-out approved<br>[`tests/fixtures/graph/heldout/results/v4-gpt-5.6-terra-high.scores.json`](../tests/fixtures/graph/heldout/results/v4-gpt-5.6-terra-high.scores.json) | 90/90 | 82/90 | always false, 82/90 (0.911) | 0/8 (0.000) | undefined (0/0) |
| **FD-B** swarm B development send-back<br>[`grid/v4/gpt-5.6-terra-high.scores.json`](../grid/v4/gpt-5.6-terra-high.scores.json) | 39/39 | 25/39 | always true, 25/39 (0.641) | 25/25 (1.000) | 25/39 (0.641) |
| **FD-B** swarm B development approved<br>[`grid/v4/gpt-5.6-terra-high.scores.json`](../grid/v4/gpt-5.6-terra-high.scores.json) | 39/39 | 31/39 | always true, 31/39 (0.795) | 31/31 (1.000) | 31/39 (0.795) |

Approval is the strongest **HO-A** v4 result. Terra high has coverage 88/90, correctness
86/88, and full-denominator accuracy 86/90 (0.956); it finds all 8 approved positives with
2 false positives, recall 8/8, and precision 8/10. The always-false majority baseline has
full coverage, correctness 82/90 (0.911), and finds no positives. Thus v4 detects every positive at a small
false-positive cost and still beats the majority baseline on fresh gold and a measurement
target that were absent during prompt development. The raw sessions were not held out, as
noted above. This is genuine detection under the fresh fixture's outcome convention; it is
not evidence that the incompatible development convention measures the same target.

Send-back is a weak result, not the headline. **HO-A** v4 terra has coverage 96/98,
correctness 95/96, and full-denominator accuracy 95/98 (0.969),
recall 89/92 (0.967), and perfect precision 89/89. It beats the unconditional baseline by
only 3 nodes of full-denominator accuracy; the majority baseline has coverage 98/98 and
correctness 92/98. The gain lives entirely in precision: its recall is below the baseline's
92/92, so 89/92 read alone is worse than always guessing send-back.

**Correction to the afternoon v4 claim, FD-B:** `PM-REPORT.md` and `PM-STATUS.md` reported
v4's swarm B terra send-back change from 1/25 to 17/25 recall as a real win on the swarm B
development fixture. They did not measure the majority-class baseline. **FD-B** development
v4 terra actually has coverage 25/39, correctness 17/25, full-denominator accuracy 17/39
(0.436), recall 17/25 (0.680), and precision 17/25 (0.680), versus the always-true baseline's
full coverage 39/39, correctness 25/39 (0.641), recall 25/25, and precision 25/39 (0.641).
V4 is worse than the baseline by 8 nodes
of accuracy and 8 positives of recall, and better only by 0.039 on precision. The published
send-back win therefore does not survive its own baseline. Development approval is also worse:
**FD-B** v4 coverage is 19/39, correctness is 7/19, and full-denominator accuracy is 7/39
against the majority baseline's full coverage 39/39 and correctness 31/39, with 0/31 recall
and undefined precision versus the baseline's 31/31 recall and 31/39 precision.

Every **FD-B** recall and precision figure in the development leaderboard was originally reported
without its majority-class baseline, so precision in particular can describe a classifier
that is worse than guessing on accuracy and recall. This applies beyond v4. Rechecking the
committed score receipts, only the final v2 terra-high row beats both development outcome
baselines; final v2 luna-low beats send-back accuracy by one node (26/39 versus 25/39) but is
below the approval baseline (29/39 versus 31/39). Every final v1, v3, v3-strict, and v4 outcome
row and every three-run replication outcome row is below its corresponding majority accuracy
baseline (**FD-B**). On held-out swarm A, only **HO-A** v4 terra-high beats both outcome
accuracy baselines.

#### Incompatible fine-role and outcome conventions

A zero-spend rerun of the current scorer over every committed v1/v3 prediction file reproduced
each saved score object exactly. The disagreement is not scorer drift. It is incompatible gold:
swarm B uses assignment-oriented fine roles, with 85 `review` and no `approve` or `send_back`
fine roles; swarm A uses outcome-oriented fine roles, with 59 `send_back`, 4 `approve`, and 7
`review`. Fine-role scores are not comparable across the two fixtures at all.

V4 behaves the same way on both fixtures by making reviewer fine role outcome-sensitive. On
**FD-B** swarm B it predicts `send_back` for 75 of the gold `review` nodes and `approve` for
4, scoring 70/149; on **HO-A** swarm A the compatible outcome-oriented gold lets the same
behavior score 104/110.
Those numbers are one behavior scored against two rulebooks, not a capability that changed.
Accordingly, neither the **HO-A** held-out jump from v3 47/110 to v4 104/110 nor the
**afternoon v4 claim, FD-B**, development fall from v3 135/149 to v4 70/149 is a
cross-fixture improvement or regression. The swarm B fine-role collapse is retired as
evidence of a v4 defect.

The same convention conflict answers why **HO-A** swarm A has 92 send-back positives among 98 labeled
nodes. Its 92 positives comprise 59 rejecting reviewer sessions and 33 developer sessions
whose work was rejected. **FD-B** swarm B instead marks the 14 repair attempts opened by a sent-back
cause and the 11 reviewer attempts cited by those causes; it does not mark the original
implementation attempt whose work was rejected. Swarm A also uses null where affirmative raw
evidence does not establish false, while swarm B has structured cause evidence. The 92/98 rate
is therefore principally a convention and evidence-conditioned-denominator difference, not a
real base rate comparable with swarm B's 25/39. There is a rejection-heavy process underneath
the labels, but the fixtures do not estimate the same event.

#### Exact v1 versus v3 fine-role delta

Exactly 15 of 110 **HO-A** held-out fine-role predictions change between v1 terra high and v3 terra
high. V3 gains 3 formerly wrong nodes, loses 5 correct nodes to another class, loses 5 correct
nodes to null, and changes 2 nodes that remain wrong. The paired accounting is
`54 + 3 - 5 - 5 = 47`; the other 95 predictions are identical, with 44 correct and 51 wrong.

| row | node id | gold | v1 | v3 | effect |
|---:|---|---|---|---|---|
| 5 | `cc_cc6d29ea-d99d-4789-9453-5c4601645356_agent-a1ff17a1b4ede88f5_f40f7676ab` | implement | repair | implement | v3 correct |
| 10 | `cc_cc6d29ea-d99d-4789-9453-5c4601645356_agent-a4c7a875811c5c9bf_4a8a506ec8` | implement | repair | implement | v3 correct |
| 20 | `cc_cc6d29ea-d99d-4789-9453-5c4601645356_agent-a770927514d895926_7d0d72fc0c` | implement | repair | implement | v3 correct |
| 24 | `cc_cc6d29ea-d99d-4789-9453-5c4601645356_agent-a953a7efebc09ae31_6482598943` | repair | repair | implement | v1 correct |
| 29 | `cc_cc6d29ea-d99d-4789-9453-5c4601645356_agent-aab402fbf7e35a1f4_40f7193c2e` | repair | repair | null | v1 correct |
| 30 | `cc_cc6d29ea-d99d-4789-9453-5c4601645356_agent-aaca38e6beedacfae_a06609bcbf` | repair | repair | null | v1 correct |
| 33 | `cc_cc6d29ea-d99d-4789-9453-5c4601645356_agent-ac2d99d6dd5cbbf63_8e06f21714` | repair | repair | implement | v1 correct |
| 37 | `cc_cc6d29ea-d99d-4789-9453-5c4601645356_agent-adb0a959fd8762cae_7176d644b0` | repair | repair | null | v1 correct |
| 39 | `cc_cc6d29ea-d99d-4789-9453-5c4601645356_agent-aec8919df496a940e_22787547cc` | repair | repair | null | v1 correct |
| 40 | `cc_cc6d29ea-d99d-4789-9453-5c4601645356_agent-af14677c62cbaad65_3d75629f80` | repair | repair | null | v1 correct |
| 46 | `cx_01a05a09-0b11-7f52-aab5-f9a2da7d1d6d` | review | review | send_back | v1 correct |
| 47 | `cx_01a05a0e-94fb-7a11-aef7-1fb93b4ea28d` | review | review | approve | v1 correct |
| 48 | `cx_01a05a0e-feee-7b00-8498-808d72bc4dbf` | review | review | approve | v1 correct |
| 73 | `cx_01a05a75-816f-7353-81f8-8268a52cda24` | send_back | approve | review | both wrong |
| 74 | `cx_01a05a76-b9f4-78e0-b92c-3ecee03ed94c` | send_back | approve | review | both wrong |

V3 contains the complete v1 prompt and then adds timestamp-only `ATTEMPT_EVIDENCE`. It says
that attempt evidence changes only boundaries and that role, fine role, and parent must apply
the v1 definitions as if it were absent. It adds no legitimate fine-role evidence. The 15
changes are therefore context sensitivity across independent stochastic calls, not the
application of a new fine-role rule; one run per prompt cannot establish deterministic cause.

#### V5 proposal, not an implementation

The exposed defect is not prompt wording. The two golds disagree about what a review node's
fine role is called. Before building v5, choose and document one cross-fixture convention. The
recommended assignment convention records what a session was asked to do as fine role and
leaves later disposition to `send_back` and `approved`; the alternative outcome convention
requires updating both golds consistently. Separately define which endpoint owns send-back
(original work, rejecting review, repair, or whole session) and apply one false-versus-null
rule.

Only after harmonizing or reannotating both fixtures should a candidate v5 be considered. If
repeated measurements still show instability, isolate assignment-based fine-role
classification from disposition classification, add paired examples for `implement` versus
`repair` and `review` versus `approve`/`send_back`, and retain v3's timestamp projection for
boundary comparability. This is a proposal only; no v5 prompt was built. These measurements do
not by themselves authorize a default change.

## Tests

`tests/test_graph_schema.py` pins the constants and the `to_dict` round trip.
`tests/test_graph_extract.py` runs the extractor on the hand-written swarm under
`tests/fixtures/graph/skeleton/` (a lead with two verified subagents, one path-contained
subagent and one unlinkable one; a Bash-launched codex review; a plan written by one
session and read by two others, plus a read of it before it was written; a read under a
timestamp that does not parse; an external launcher in another workspace; a codex
session nobody launched) and asserts every node, edge, tier and count, including the
exclusion counters.
`tests/test_graph_render.py` pins the DOT output. Run them with:

```
.venv/bin/python -m pytest -q tests/test_graph_schema.py tests/test_graph_extract.py tests/test_graph_render.py
```
