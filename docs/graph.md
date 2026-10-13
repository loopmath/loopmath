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

Artifact recording adds nullable `bytes`, `lines_added`, `lines_removed`, `language`, and `tests_touched` values with sibling evidence tiers, a `fate` of `kept`, `edited`, `reverted`, `deleted`, or the default `unknown` with `fate_tier`, and a free-form `meta` object that core graph logic ignores and consumers may ignore. `bytes_tier` is `verified` only when a successful transcript write records the complete UTF-8 content. `lines_added_tier` and `lines_removed_tier` are `verified` for diff lines recorded by the source and `heuristic` when the adapter computes a diff from recorded replacement text. `tests_touched_tier` is `heuristic`: its case-sensitive predicate counts distinct paths below a `test/` or `tests/` directory or with a basename matching `test_*.py`, `*_test.<lowercase letters>`, or `conftest.py`, and only `/` separates components. A write without an observed transaction fact does not fall back to testing its artifact path. Across three of our own swarm runs, 54 of 603 artifacts have a non-null path-list observation for `tests_touched`, while 549 remain null: 110 because multiple writes collapse into one artifact and 439 because no observed transaction fact establishes the touched paths. `language` makes no language claim: it holds the basename text after the final dot, ASCII-lowercased, when that identifier matches `[a-z0-9][a-z0-9_+-]*`, and is null for an absent or invalid extension; `language_tier` is `heuristic`. A known `fate` tier follows the explicit source evidence and cannot be stronger than the represented write; absence of a later write leaves `fate` as `unknown` with a null tier. Because the graph still collapses repeated writes of one path into one artifact, any per-version measurement that cannot be assigned honestly remains null and its machine-readable reason is counted in `Graph.meta`. Artifact version lineage, `meta` export privacy, valuation, and UI are deliberately absent.

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
`post` (started more than 60 s after the lead ended, where post-build reviews sit) and `external` (launched from outside the workspace).
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
g = extract(records, workspaces=["shop-swarm-a"])
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
test loads unedited `Graph.to_dict()` outputs of three real swarm runs, A, B and C. They
are private inputs, so the test skips in the public tree. It calls
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

The native fixtures are historical writer artifacts, so reader enrichment
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
The historical snapshots remain unchanged deliberately: they are the only
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

Until these are addressed, the counts in `meta` show the gap rather than hiding it.

- **Files written through Bash are invisible.** Only the Write, Edit, MultiEdit,
  NotebookEdit and Read tools produce writes and reads. A plan written with `cat > f
  <<'EOF'` or a review produced by `codex exec -o f` is not an artifact, which is why the
  swarms' plan and review documents show as 0 artifacts.
- **Codex sessions have no writes or reads yet**, so an artifact edge never ends at a
  codex node and a read a codex session makes of a Claude-written file is not seen.
- **Launches through scripts and backgrounded commands are missed.** The join needs the
  CLI name in the Bash command text and a session starting inside the call's interval (or
  within 5 s before to 120 s after its start, same workspace only). `nohup tools/review.sh
  &` launches codex from inside the script and returns at once, so it does not match;
  those sessions are counted in `unlaunched_codex`.
- **Cross-workspace launches must name the workspace** in the command (`cd
  ~/work/swarm-a && codex exec ...`); without it, a session that starts in one workspace
  could be attached to a launch made seconds earlier from another.
- **The external launcher node is thinner than the others.** It is built from the
  launcher's record only, so it carries no `wall_s`, `tokens` or `usd` (they are `None`
  and counted in `nodes_missing_wall_s`, `nodes_missing_tokens` and
  `usd_unpriced_nodes`); its phase `external` has tier `heuristic` like every phase.
- **Roles are coarse.** Lead, planner, dev, reviewer, solo, cli, external. Repair versus
  implement, send-backs and approvals, and boundaries inside one long session are what
  the labelling evaluation exists to measure. The lead rule is a
  heuristic with ties broken by output tokens then wall clock; a workspace with two
  independent top-level sessions gets one lead and one unlabeled session.
- **The 60 s post-build gap** is a fixed rule (`labels.POST_BUILD_GAP_S`); a session that
  started during the lead's run but did review work is phase `build`.
- **Role evidence text repeats a declared type** when the Task's `subagent_type` and the
  meta's `agentType` agree (`declared type 'Plan Plan'`). Cosmetic; the tier and role are
  right.

## Evaluation

The extractor and the labeller were measured on our own swarm runs, which stay private.

The extractor is scored against hand-labelled ground truth:

- session counts equal the raw counts, so nothing is dropped;
- every subagent's spawn edge is `verified`;
- post-build evaluation sessions are separated from the build (phase `post` or `external`);
- launch edges reach every `codex` run of the build;
- plan and review documents appear as artifacts with a producer and at least one consumer.

The labeller (`--prompt-version`, default `v3`) was compared across models, efforts and prompt versions on a development fixture of 195 nodes, then on a held-out swarm.

- `gpt-5.6-terra` at effort `high` gave the best role accuracy.
- `gpt-5.6-luna` at `low` is the cheaper alternative for bulk runs.
- Every configuration cost under 0.15 percent of a lower bound on the labelled work's cost, so cost did not decide.
- The labeller command takes the model and effort as required arguments; it has no built-in default.

What the numbers do not show:

- Prompt `v2` places attempt fields in the prompt from which the send-back and approval labels can be recovered. Its outcome rows are therefore not detection performance; it stays selectable as a circular control.
- Prompt `v3` keeps only session ids and times per attempt. `v3-strict` drops the attempt block entirely, as a control; its role accuracy shows the role result does not depend on the attempt evidence.
- Timed boundary recall under `v3` is supplied by the attempt start times, so the honest timed result is the `v3-strict` one.
- Three repeated runs of one configuration varied by up to 8 points in role accuracy, so small gaps between configurations are within noise.

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
