# OCP Draft (08-30)

The JSON schema (`spec/ocp-v0.3.schema.json`; `spec/ocp-v0.2.schema.json` for
v0.2 documents) defines the shape of a document, and the meaning of each field
in its description strings is authoritative for per-field meaning. This document is authoritative only for rules that JSON
Schema cannot express, including cross-field rules, evidence and process rules,
and privacy profiles. Each such rule will be numbered and backed by one checker
rule and one golden pair; reducing this document to that numbered list is a
later task, not this one; the v0.3 rules in section 8 are numbered that way.
The checker (`loopmath.ocp.conformance`, run as `loopmath ocp validate` or the
`spec/ocp_conformance.py` shim) tests both sources and is never the authority.

Status: v0.1 draft, written 2026-08-30, revised same day after the six framework survey notes arrived. The revision rewrote section 5 and the adoption path against the notes, added an optional `cost.basis` field to the schema, and added a referential conformance checker at `spec/ocp_conformance.py`. Spec and examples live in the loopmath repo at `spec/ocp-v0.schema.json` and `spec/examples/`. Author: Analyst.

## Abstract

OCP, the Orchestration Context Protocol, is a standardized file format for orchestration telemetry. The pitch in one line: what MCP did for tool access, OCP does for orchestration run data. Any agent framework plus a small plugin should emit files that loopmath can learn from. Version 0.1 is a single JSON document per run carrying the DAG topology as nodes and typed edges, a flat list of attempts with model and reasoning-effort labels, per-attempt token cost records split into input, cached input, and output, acceptance outcomes graded by evidence tier, timestamps, producer identity, and a declared privacy profile where metadata_only is first class. The seed is herdr-dagr contract v3: OCP keeps its attempt vocabulary, evidence tiers, and cause types verbatim, flattens its nesting, splits its fused model-and-effort labels, and adds the two things v3 lacks, cost records and producer identity. A working stdlib converter turns real contract v3 runs into valid OCP today, and a JSON Schema (draft 2020-12) validates them. The main design bet is that a minimal required core (run, nodes, attempts) with everything else optional and typed is small enough that a Claude Code, BB, Orca, Pi, LangGraph, or CrewAI plugin fits in a few hundred lines. The main honest risk is that the E0 pipeline's hardest problem, attributing token cost to attempts, does not disappear under OCP; it moves into the producer plugins, and the spec must say whose job it is.

## 1. Motivation

loopmath learns a posterior over agent-workflow cost from orchestration logs. Today its corpus comes from exactly one producer lineage: herdr-dagr contract v3 run files plus a hand-built harvest of Claude Code session logs. That is one operator, one toolchain, ten runs. The posterior cannot generalize past what the corpus contains, and the corpus cannot grow past what one contract emits.

The E0 analysis made the cost of this concrete in three places. First, contract v3 attempts carry no token counts at all, so tokens per task had to be reconstructed by joining session logs to attempts by time-window containment, and 83 percent of those join rows sat in the loosest match tier. Second, model labels arrived as an inconsistent fused vocabulary (`fable·xhigh`, `claude-fable-5·xhigh`, `opus5.xhigh` with a typo separator, 34 nulls), so an ad hoc canonicalization table had to ship inside the analysis script. Third, acceptance grades had to be split between a gate grade that only the DAG slice has and a proxy grade parsed out of session structure. All three are format problems before they are statistics problems.

The MCP analogy is the framing, used honestly. Before MCP, every assistant grew bespoke connectors to every tool, and the fix was a small protocol that either side could implement once. Orchestration telemetry is in the bespoke-connector stage now: every framework logs runs in its own shape, every analysis pipeline writes one-off parsers, and cost-to-acceptance data that could be pooled across users and frameworks stays siloed. OCP is the small protocol for that seam: frameworks implement one emitter, loopmath implements one reader, and a corpus can come from anywhere. The analogy has a known weakness, recorded in the open questions: MCP had a major vendor pushing both sides of the seam, and OCP starts with one consumer.

## 2. Design principles

**File based and append friendly.** An OCP run is a JSON document at a path, complete at rest, like contract v3. No daemon, no endpoint, no database. Watching is an mtime poll. The internal shape favors appending: attempts, edges, and events are flat top-level arrays rather than structures nested inside nodes, so a producer extends a run by appending records and rewriting one file, and a streaming producer can buffer records in the same shape. Whether v0 also needs a formal JSONL stream profile is an open question.

**Minimal required core with typed extensions.** Only five things are required: the version marker, producer identity, the privacy declaration, a run id, and the node list. Everything else, including edges, attempts, costs, and events, is optional but typed when present. A framework that knows only "these steps ran with these models and these tokens" can emit valid OCP on day one and add topology later. Producer-specific data goes in an `ext` object present on every entity, with dotted-namespace keys such as `dev.dagr.policy`, and consumers must ignore namespaces they do not understand. Unknown fields are ignored for forward compatibility, the same stance contract v3 takes.

**Metadata-only privacy as a first-class profile.** Every document declares `privacy.profile`, either `metadata_only` or `full`. Under metadata_only, no field anywhere may contain prompt, completion, or transcript text; free-text fields are short producer-authored labels capped at 500 characters. This is the profile loopmath's own corpus rules already require, and making it declared rather than assumed means a third-party corpus states its own admissibility. The cap is enforceable by schema; the no-transcript rule is a conformance promise, and the trust model for that is an open question.

**Producer agnostic.** The protocol never assumes herdr, panes, Claude, or any runtime. Runtime locators are exactly the thing contract v3 learned to keep out of identity, and OCP drops them entirely. Node ids are producer-scoped and stable, model labels preserve the producer's raw string as ground truth, and canonicalization is explicitly the consumer's job. The producer block names the emitting software, the framework, and the upstream native format, so a consumer can stratify by lineage and debug bad producers.

## 3. Core data model

All timestamps are ISO-8601 with explicit offset. All entities accept an `ext` object.

**Run.** One orchestration execution. It has a stable id, an optional short title, start and end timestamps, an opaque `workspace` locator so task-mix normalization can stratify by project without learning the path, and free-form string labels for experiment arms.

**Producer.** Who emitted the document: the plugin or converter name and version, the framework and framework version the run executed under, the upstream native format if the document was converted (for the seed lineage, `dagr/3`), and the emission time.

**Privacy.** The declared profile as above, with an optional short note.

**Group.** An optional named scope with an id, title, and optional parent, forming an acyclic forest. Groups are contract v3 projects renamed. Grouping answers where work is shown; edges answer what work blocks. The two stay orthogonal.

**Node.** A unit of work with stable identity independent of any attempt at it. It has an open-set `kind` with a recommended vocabulary (impl, review, test, gate, question, docs, ship, plan, ops), an optional title, group, and labels, and an optional `state` that is the producer's projection over the node's attempts. A consumer learning from settled runs should trust attempt outcomes over node state. A gate is simply a node whose kind is `gate`; it needs no separate entity.

**Edge.** A directed pair of node ids with a kind. `dep` means the target cannot start before the source settles. `fan_in` means the source is a member of the target gate's acceptance input set. `spawn` means the target node was created dynamically by work on the source, which is how frameworks with runtime-discovered topology record what actually happened. Dep and fan_in edges together must form a DAG.

**Attempt.** One try at a node, and the record that carries almost everything loopmath learns from. It has a globally unique id, its node, a 1-based ordinal, a logical actor, a model reference, a separate reasoning-effort label, a cause, a status, start and end timestamps, an outcome when terminal, a cost record, an optional opaque session correlate, and free-form string labels. Attempts are records, not counters: a retry opens a new attempt with a cause pointing backward at what triggered it. The cause vocabulary is contract v3's five types (initial, sent_back, gate_failed, followup, superseded) plus other, the escape value producers use when a trigger fits none of them. The status vocabulary is contract v3's attempt states plus canceled: queued, working, done, failed, rejected, canceled, settled_unverified, lost.

**Model reference and effort.** The model is an object, not a string. `raw` preserves the label exactly as the producer knew it and is always kept. `id` is the canonical provider id when known, `family` a coarse grouping label, `provider` the vendor. Reasoning effort is a separate open-set field on the attempt (low, medium, high, xhigh, max, and whatever comes next), never fused into the model label. This split exists because the fused labels in the current corpus cost E0 a hand-maintained regex table.

**Cost record.** Token and money cost of one attempt, summed over all API requests the attempt made: `input_tokens` (uncached), `cached_input_tokens` (cache reads), optional `cache_creation_tokens`, `output_tokens`, optional `reasoning_tokens`, a `requests` count, and optional `usd`. Cached input is separated because cache pricing differs by an order of magnitude and merging the two blurs every cost comparison. Consumers should prefer recomputing dollars from tokens and a dated price table over trusting `usd`. Added in the 08-30 revision: an optional `basis` enum, `measured` or `allocated`. Measured means the counts come from usage records that identify this attempt exactly, such as a per-agent transcript. Allocated means they were apportioned from a coarser aggregate by a heuristic such as a time-window join. The survey notes showed allocation is the norm, not the edge case: Orca persists its own time-window attribution heuristic, CrewAI reports only crew-level aggregates, and Claude Code hook payloads carry no token counts at all. Consumers should down-weight allocated costs.

**Outcome.** The terminal result of an attempt plus how that result is known. Results: done, failed, rejected, canceled, settled_unverified, lost. Evidence tiers, verbatim from contract v3: verified (mechanically checked), reported (typed self-report), heuristic (inferred), asserted (bare claim). Missing evidence is treated as asserted, never upgraded. An optional `via` names the gate or review node that produced the acceptance signal, plus a short receipt and reason. This is the acceptance signal loopmath's gate-grade slice runs on.

**Event.** An optional append-only provenance record: timestamp, open-set type (recommended: attempt_started, attempt_settled, promoted, directive, note), optional node, attempt, actor, and short detail. Events answer why an attempt exists and when promotions happened; they are not required for cost learning.

## 4. Contract v3 as the first OCP producer

Contract v3 is the seed and stays the richest producer. The converter at `spec/examples/ocp-from-contractv3.py` turns any v1, v2, or v3 run file into valid OCP today. What changes, exactly:

1. **Kept verbatim:** the attempt state vocabulary, the outcome result vocabulary, the four evidence tiers, and the five cause types. These were designed against real failure modes and OCP adopts them unchanged. Event types also pass through as-is.
2. **Renamed only:** projects become groups; task refs on events become node refs.
3. **Restructured:** tasks become nodes; attempts move out of tasks into one flat top-level array with a `node` backref; `deps` and gate `inputs` become explicit edges with kinds `dep` and `fan_in`. Same information, append-friendly shape.
4. **Split:** the fused model string becomes `model.raw` (always preserved) plus `model.family` and a separate `effort` field when the middle-dot convention parses cleanly. Labels like `fable∥gpt` or the `opus5.xhigh` typo stay raw for the consumer's table.
5. **Added, and this is the real gap:** `attempt.cost`. Contract v3 carries no token usage anywhere, which is why the E0 pipeline needed the fragile session join. The converter therefore emits cost-less OCP from v3 files. Closing the gap needs a small contract amendment or orchestrator change: an optional per-attempt usage object with the same field names as the OCP cost record. Until an orchestrator writes it, loopmath keeps the session join as fallback, and OCP's optional `attempt.session` correlate at least makes that join declared instead of inferred.
6. **Added:** the required `producer` and `privacy` blocks. The converter fills them in.
7. **Moved to ext or dropped:** loop policy moves to `ext["dev.dagr.policy"]` as inert typed data; chain keys and progress move to ext; pane locators, liveness hints, operator-message plumbing, and legacy actions are dropped. Those serve live rendering and control, not learning telemetry. loopmath the viewer keeps reading contract v3; OCP is the interchange for loopmath the learner.

Round-trip check on the real assemble run: 11 nodes, 12 edges, 12 attempts, 9 events, no dangling references, validates against the schema.

## 5. Framework mapping appendix

Grounding note: the six framework survey notes commissioned for this draft arrived on 2026-08-30, after the first version was written from general knowledge, and this appendix has been rewritten against them. Every mapping below is grounded in the notes unless labeled unverified. The notes themselves are tiered: BB, Orca, Claude Code, and core Pi claims were verified against local installs, live artifacts, or source; Pi workflow-package internals and Claude Code agent-teams details come from READMEs and docs only. Residual uncertainty is flagged inside each entry. Entries are ordered by priority: the tools in daily use first, generic frameworks after.

### 5a. Claude Code (dynamic workflows, subagents, hooks)

Verified locally: 37 workflow runs and 420 agents inspected structurally on this machine.

Artifacts it actually produces, per run under `~/.claude/projects/{project}/{session}/`: a run-state file `workflows/wf_{runId}.json` (runId, status, phases, per-agent entries with label, phaseIndex, resolved model id, state, queuedAt and startedAt, an attempt counter, one total token figure, durationMs), the generated orchestration script, a `journal.jsonl` with started and result lines per agent, and per-agent transcripts `agent-{agentId}.jsonl` plus `meta.json` (agentType, model, parentAgentId, spawnDepth). Every assistant transcript line carries the full usage split (input, output, cache creation at 5m and 1h tiers, cache read) and the model id per API call.

Capturable: tokens yes with full split, but only by joining the per-agent transcripts, since the run file holds a single total per agent. Model yes, resolved ids at both grains. Effort no in any artifact: it is requested in scripts and frontmatter but never persisted, and survives only in live hook payloads and the OTel token-usage event, so a purely post-hoc scrape loses it. Timestamps yes throughout. Attempts weak: the per-agent attempt field was 1 in all 420 observed agents, failures are recorded as an error state, and resume reruns agents instead of linking retries. Topology partial: phase membership, the parentAgentId spawn tree, and journal key order, but no edge list; data dependencies live only in the generated script code. Gates are conventions only (adversarial verify stages, schema-validated structured output); the teams TaskCompleted hook can block, but teams are experimental and docs-only.

Exporter extension point: hooks. SubagentStart and SubagentStop carry agent identity and the effort level; a SessionEnd pass joins token splits from the transcripts by agent id. Alternatives: a zero-config post-hoc scraper over the run files, or an OTLP receiver (the api_request event carries model, token splits, and cost_usd, and content attributes are off by default, so OTel is metadata-only out of the box).

Size: roughly 100 lines of hook script plus the transcript-join pass, 100 to 300 lines total.

Consequence for OCP core, confirmed: edges must stay optional, or Claude Code becomes a second-class producer. Also, run files are swept after about 30 days by default, so harvest promptly.

### 5b. BB (get-bb/bb)

Mapping verified against the source and schema pinned at d0e727447, not upstream HEAD, which moves fast and has no API stability promise.

Artifacts: one SQLite database, default `~/.bb/bb.db`. The threads table carries parent and source thread lineage, model_override, and reasoning_level_override. An append-only per-thread events table carries turn started and completed events with status, turn requests with the execution options (model, reasoningLevel from the enum none, low, medium, high, xhigh, ultracode, max, ultra), item lifecycle events, cumulative token-usage updates split into input, cached input, output, and reasoning output, and model-fallback events. A pending_interactions table records blocked-on-human approvals and questions with resolution and resolved_at. Read access also exists over an HTTP API (threads, paged and long-poll events, timeline, interactions) and a WebSocket hub.

Capturable: tokens yes with a full split including reasoning output, but cumulative per thread, so per-attempt attribution means diffing the breakdown at turn boundaries; no dollar cost anywhere. Model yes per turn request, corrected by fallback events. Effort yes, first class per turn, the richest effort enum surveyed. Timestamps yes, ms epoch plus per-thread sequence numbers. Topology partial: a manager and worker thread lineage tree, but no declared dependency DAG; flat per-thread task lists exist without dependencies, and upstream dropped its workflow tables, so there are no first-class gate or review objects. Fresh-agent respins appear as unlinked sibling threads the exporter must link, and there is no native run boundary, so the mapping adopts the convention that a root manager thread is the run.

Exporter extension point: a backend plugin, with no upstream changes needed. The plugin SDK provides thread lifecycle events, a loopback SDK for reading events and timelines, its own storage, cron schedules, and its own HTTP routes. Alternatives: an external poller on the HTTP API, or read-only SQLite access for backfill.

Size: a few hundred lines, spent mostly on the metadata whitelist (event payloads embed transcript text) and the turn-boundary token diffing.

### 5c. Orca ADE (stablyai/orca)

Verified against a shallow source clone on 2026-08-30.

Artifacts: an orchestration SQLite database in the app's userData directory with runs, tasks whose deps column is a JSON array of task ids (a real declared DAG, the only external framework surveyed that has one) plus parent nesting, dispatch_contexts (per-task tries with dispatch and completion times, status including circuit_broken, failure_count, free-text failure reasons, depth), decision_gates (question, options, resolution, timeout), and typed messages (worker_done, merge_ready, escalation, handoff, heartbeat). Beside it: per-provider usage caches with per-session token splits including reasoning output and cache writes plus a primary model label, a main store with worktree lineage keyed to run and task ids, and automation runs that carry estimated dollar cost and an explicit attribution field whose value is a time-window heuristic. A CLI exposes run, task, and gate listings as JSON.

Capturable: topology yes, declared. Attempts yes, one dispatch row per try, though retry causes are free text rather than typed. Gates yes, decision gates plus typed completion messages, without evidence tiers. Tokens yes with full split, but rolled up per provider session, day, and worktree, never per dispatch, so per-attempt cost needs a time-window join, the same heuristic Orca itself uses; OCP records that as `cost.basis: allocated`. Model labels live only in the usage sessions; dispatch rows carry the agent kind (claude, codex) instead, which maps to the attempt actor, not the model. Effort is never persisted.

Exporter extension point: the young plugin event bus does not expose orchestration or usage events yet, so the exporter is a poller reading the orchestration database plus the usage caches, or the CLI JSON commands, optionally scheduled by Orca's own cron automations.

Size: a few hundred lines of SQLite reading, the usage join, and metadata filtering (drop task specs, results, message bodies, gate question text, terminal archives).

Assessment: the closest external fit to the OCP model, a plausible second adopter given its pluggable usage layer, and worth watching as a herdr competitor. Caveat: the schema is young and actively migrating, so the exporter must pin a schema version.

### 5d. Pi coding agent ecosystem

Core Pi verified against the locally installed package and its docs (pi 0.84.1 on this machine); the workflow packages are README-grade and their internal schemas are unconfirmed.

Concrete names: core is earendil-works/pi on GitHub, npm `@earendil-works/pi-coding-agent`. Workflow packages: `@quintinshaw/pi-dynamic-workflows` (fan-out orchestration scripts, model routing, token budgets, journaled resume, per-project run JSON), `@tintinweb/pi-subagents` (parallel subagents with gates, per-agent records carrying tokens, dollar cost, attempt counts with retry annotations), `nicobailon/pi-subagents` and `@mjasnikovs/pi-task` (catalogued on pi.dev but not examined in depth; the latter claims deterministic pipelines with verify and enforce gates).

Artifacts: session JSONL with a documented format, the only one surveyed with public documentation. Entries form a tree; every assistant message carries provider, model, stop reason, and a usage object with input, output, cache read, cache write, and dollar cost per message. Thinking-level changes are recorded as session-scoped entries. The same events stream over `pi --mode json`, and an RPC mode exposes state and messages.

Capturable: tokens and dollars first class per message, the friendliest cost surface surveyed. Model yes per message. Effort partial: session-scoped thinking level, not stamped per subagent attempt. Attempts partial: retry annotations in pi-subagents and a positional journal in pi-dynamic-workflows, nothing in core. Topology weak: a conversation tree plus parent-session links; workflow structure is imperative JavaScript. Gates partial: shell exit-code gates, reviewer quorums, human checkpoints, no evidence tiers. Every native artifact embeds transcript text, so the exporter projects out metadata itself.

Exporter extension point: a pi package installed with `pi install`. A TypeScript extension subscribes to lifecycle events (session start, agent start, agent settled, session shutdown), has session-manager access, and can persist its own state directly into the session JSONL. Offline batch conversion of historical sessions works through the same session-manager API. The pattern is proven: herdr already installs a pi reporter extension on this machine, and Braintrust, Raindrop, and PostHog ship telemetry extensions.

Design call adopted from the notes: the OCP pi plugin depends only on core Pi (events, session JSONL, parent-session links), with optional adapters for the workflow packages, whose run formats are internal and unstable.

Size: roughly 200 to 400 lines for the core extension.

### 5e. LangGraph (drafted mapping confirmed, with corrections)

The notes read the checkpoint structures from source. Confirmed from the first draft: a thread is the run, node executions per super-step are attempts, fork lineage comes from checkpoint parents, an update source marks operator intervention, and topology comes from the compiled graph object rather than the checkpoint. Corrected: checkpoints carry no tokens and no model or effort labels at all, so cost and model data must be joined from callbacks. The plugin is therefore a checkpoint-saver wrapper plus a callbacks handler, not callbacks alone. The serialized wire format differs per backend and was not byte-verified. Roughly 300 to 500 lines.

### 5f. CrewAI (drafted mapping confirmed)

The notes verified usage metrics from source and guardrail semantics from docs. Confirmed: tasks with declared context give dep edges; each guardrail retry cycle is an attempt with a gate_failed cause (retries default to 3, with failure text sent back to the agent); guardrail results are gate verdicts; the human-input flag is an operator directive. The weak spot is confirmed too: usage metrics (total, prompt, cached prompt, completion tokens) aggregate at crew level, so per-attempt cost comes from the event bus at the point where a retry is decided, or is emitted from aggregates labeled `basis: allocated`. Roughly 300 to 500 lines.

### 5g. OpenTelemetry GenAI bridge (confirmed and upgraded)

The notes read the attribute names from the conventions repo, which now lives in its own repository with status Development, meaning unstable. The vocabulary OCP borrows for plugin-author familiarity: provider name, request and response model, reasoning level, and the usage token attributes including cache read and cache write splits, plus the workflow and agent invocation operation names. Confirmed floor-not-target: spans record executions, not tasks; no attempts, no gates, no declared dependencies. New since the first draft: the reasoning-level attribute exists, so the bridge can carry effort. Because the conventions are unstable, OCP pins the borrowed names to a snapshot date. Roughly 500 lines shared across every OTel-instrumented framework.

### 5h. Prior-art trajectory formats (calibration, not plugin targets)

The notes also surveyed OpenHands, SWE-agent, and MLflow. OpenHands trajectories carry per-event token usage, accumulated dollar cost, and cause links, but are linear, so no DAG. A SWE-agent trajectory maps to one node with a single attempt; its model stats map to a cost record with no cache split, and exit status plus the submitted patch make a verified-tier outcome when the harness evaluated it. MLflow traces contribute the best acceptance prior art: assessments with a source field that maps onto evidence tiers (code-sourced to verified, LLM-judge to heuristic, human to a directive). None of the six prior-art formats carries the task and attempt split, and none binds an acceptance verdict to a priced attempt, which is exactly the gap OCP exists to close.

### 5i. Unverified: OpenAI Agents SDK and AutoGen/AG2

The recovered notes do not cover these two frameworks. The first draft's mappings for them were written from general knowledge and remain unverified against current docs, so they are kept in one paragraph for shape only. Agents SDK: traces with agent, generation, and guardrail spans plus a custom trace processor would map cleanly if the drafted details hold. AutoGen: honest recording would be topology-as-executed via spawn edges. Neither is scoped for plugin work until someone verifies the extension points.

## 6. Adoption path

Order of attack, by leverage per line of code, revised against the survey notes and the stated priorities, which put BB, Orca, and Pi ahead of the generic frameworks:

1. **herdr-dagr converter: done.** About 250 lines, stdlib, in the repo. The follow-up that matters is the contract v3 usage amendment, and the notes confirm it is cheap: contract v3 ignores unknown fields at every level, so an optional per-attempt usage object can be added today without breaking loopmath check or loopmath view.
2. **Claude Code exporter.** A SubagentStart, SubagentStop, and SessionEnd hook script that appends OCP records and folds in token splits from the per-agent transcripts at run end. Roughly 100 to 300 lines. Running live matters twice: effort labels exist only in hook payloads, and run files are swept after about 30 days. The existing loopmath-data harvester, recast to emit OCP, stays the fallback for historical sessions and doubles as the reference implementation.
3. **Orca poller.** Reads the orchestration database plus the two usage caches and emits near-complete OCP: declared DAG, dispatch attempts, decision gates. A few hundred lines. Cost lands as `basis: allocated` through the time-window join, matching Orca's own attribution practice.
4. **Pi extension.** A pi package subscribing to core lifecycle events and summing per-message usage and dollar cost into attempts. Roughly 200 to 400 lines. Best-in-class cost data; topology is the weak axis. Optional adapters for the workflow packages come later, if at all, because their run formats are internal and unstable.
5. **BB backend plugin.** Tails thread events, diffs cumulative token breakdowns at turn boundaries, whitelists metadata. A few hundred lines. Gets per-turn model and the richest effort labels surveyed; topology is limited to the manager and worker lineage tree.
6. **OTel GenAI bridge.** Roughly 500 lines, shared across everything instrumented, delivering cost, timing, model, and effort without topology, attempts, or gates. A floor, not the target.
7. **LangGraph and CrewAI listeners.** Each roughly 300 to 500 lines, accepting the documented lossiness: a checkpoint-saver wrapper plus callback join for LangGraph, event-bus cost attribution for CrewAI.

OpenAI Agents SDK and AutoGen drop off the numbered path until their mappings are verified; see 5i.

Each plugin's definition of done: emits schema-valid metadata_only OCP for a real run, passes `spec/ocp_conformance.py` (schema plus referential checks), and loopmath's loader ingests it with no framework-specific code.

## 7. Open questions

1. **Cost attribution boundary (acted on in the 08-30 revision, veto open).** OCP says per-attempt cost is the producer's job. When a harness only reports usage per session and a session spans several attempts, the plugin must allocate, and the E0 join fragility reappears inside the producer where it is invisible. The survey notes settled the lean: Orca persists its own time-window attribution heuristic, CrewAI reports only crew-level aggregates, and Claude Code hook payloads carry no tokens, so allocation is the norm across producers, not the edge case. The schema now carries an optional `cost.basis` enum, measured or allocated; absence of the whole cost record covers the third case. Say the word to revert.
2. **Snapshot document versus append-only stream.** v0 is a document. Streaming producers rewrite the file per event or buffer until run end. Is a JSONL profile (one entity per line plus a reduce rule) worth speccing in v0, or does it wait for a producer that actually needs it? My lean: wait, but keep the flat arrays so the reduce rule stays trivial.
3. **Who is the second consumer?** The MCP analogy earns its keep only if something other than loopmath reads OCP. Candidates: a cost dashboard, a run differ, other researchers' pipelines. If no second consumer is plausible by the paper deadline, the honest pitch downgrades from protocol to "loopmath's documented import format, open to producers." Which claim goes in the paper?
4. **Canonical model and effort registry.** OCP preserves raw labels and leaves canonicalization to consumers. Should the spec ship a maintained mapping table (E0 already has one) as a non-normative companion file, and who keeps it current as providers rename models?
5. **Acceptance without gates.** Most frameworks have no gate object, so their acceptance signals arrive as reviewer decisions or guardrail results attached via `outcome.via`. Does the gate-grade slice in the analyses accept those as gate-grade, or does that dilute the evidence-tier story that makes the loopmath corpus credible?
6. **metadata_only trust model.** The schema can cap string lengths but cannot detect prompt text inside a 400-character title from a third-party producer. Is declared conformance plus spot-checking enough for corpora we did not produce?
7. **Framework notes: recovered and reconciled (resolved 2026-08-30).** The survey notes arrived after the first draft, and section 5 plus the adoption path were rewritten against them. The diffs, logged honestly: the first draft guessed at OpenAI Agents SDK and AutoGen, which the notes do not cover, so both are demoted to an unverified stub (5i). It missed BB, Orca, and the Pi ecosystem entirely, and each now has a grounded entry with concrete repo and package names; they also moved ahead of the generic frameworks in the adoption path. The drafted LangGraph and CrewAI mappings survived with corrections, the main one being that LangGraph checkpoints carry no tokens or model labels at all. The OTel entry gained the reasoning-level effort attribute the drafter did not know about. Still unverified after the notes: Pi workflow-package internal schemas, Claude Code agent-teams file shapes, BB upstream drift past the pinned commit, exact Orca CLI output shapes, and the two frameworks in 5i.

## 8. Version 0.3

Added 2026-09-23 (loopmath 0.1). v0.3 is a strict
superset of v0.2: every valid v0.2 document is a valid v0.3 document once its
`ocp` field says `0.3`. It records what ran as a configuration (a workflow
graph plus a setting per piece), the task, the acceptance rule and its signals,
pair slates and preferences, and loopmath's predicted-against-actual receipt.
The schema's description strings define each field; this section holds the
canonical id, the numbered rules, migration and the namespace.

New fields, by object:

- `run`: `task`, `configuration`, `provenance`, `acceptance_rule`, `signals[]`,
  `slate`, `preferences[]`, `rescue`, `receipt`.
- `node`: `vertex` (the workflow piece it executes), `gate`.
- `attempt`: `vertex`, `round` (repair round, from 1), `setting` (only when it
  differs from the configuration), `cwd`; `attempt.cost.tariff`.
- `artifact`: `vertex`, `version`, `supersedes`, `transfer_tokens`.
- event types `signal_observed`, `receipt_written`, `run_finished`; artifact
  kinds `commit` and `merge` (a late event is matched to its run by commit),
  and `issue`, `repo`, `diff`, `verdict` and `test_record` (the artifact kinds
  of workflow pieces, recorded as the workflow names them); producer
  capabilities `task`, `configuration`, `signals`, `slate`, `receipt`.

Two readings the schema text states and every reader applies: `control.budget`
counts rounds including the first, and a missing or 0 budget reads as 1; a
setting without `context_policy` reads as `fresh`, and loopmath always writes
it. Piece roles are open strings; loopmath writes its own role words
(`planner`, `implementer`, `reviewer`, `tester`, `referee`, `worker`) as they
are, and readers never rewrite a role.

Examples: one run per catalog shape in `spec/examples/v0.3/` (`solo`,
`best_of_n`, `plan_implement`, `implement_review`, `plan_implement_review`,
`swarm`) and `full-fields.ocp.json`, which fills every field v0.3 adds. Each
inlines its shape exactly as loopmath's catalog defines it, so the `{ref,
version}` form of its workflow gives the same `configuration.id`.

### 8.1 Canonical configuration id

`configuration.id` is `cfg_` plus the first 12 hex digits of the SHA-256 of the
canonical JSON of `{"workflow": W, "settings": S}`, encoded as UTF-8. The
canonical JSON has sorted keys, no whitespace (separators `,` and `:`) and
non-ASCII characters written as themselves. It keeps what ran and drops labels:

- W is the workflow with its `id`, `version`, `title` and `ext` removed. A
  workflow reference `{ref, version}` is replaced by the catalog workflow it
  names before hashing.
- `pieces`: `{id, width, role, workflow}`, with `width` 1 when absent, `role`
  only when present, and a nested workflow canonicalized the same way. Pieces
  are sorted by id.
- `artifacts`: `{id, kind}`, sorted by id. `edges` are sorted as pairs.
- `control`: `gates` (sorted), `repair`, `budget` read as max(1, budget) with 1
  when absent, `rescue` reduced to `kind` and `ref`, and `gate_rules` from
  `control.ext["dev.loopmath.gate_rules"]` when it is present and not empty (a
  non-default gate rule changes what ran). Other `ext` keys stay out.
- S maps each piece id to `{harness, model, effort, context_policy, options}`,
  where `model` is the modelRef `id`, else its `raw`, else the bare string;
  `effort` is `default` when absent, `context_policy` `fresh` and `options`
  `{}`. Setting `ext` stays out.

The one implementation is `loopmath.ocp.canonical.config_id`;
`tests/ocp/test_canonical.py` pins a literal id so the hash cannot drift across
Python versions.

### 8.2 Numbered rules

Each rule has a pass and a fail file in `spec/examples/golden/v0.3/`
(`<rule>-pass.ocp.json`, `<rule>-fail.ocp.json`; expected codes in
`expected.json`). The rules apply to documents at v0.3 or later.

| Rule | Level | What it checks |
|---|---|---|
| E190 | error | `run.configuration.id` equals the canonical id (8.1) of its workflow and settings. A workflow reference is resolved through the reader's catalog; a reader that cannot resolve it skips this rule and E191. |
| E191 | error | Every piece of the workflow has an object in `configuration.settings` under its id. A piece that holds a nested workflow and has no role needs none. |
| E192 | error | Piece and artifact ids are unique in one namespace; every edge joins a piece and an artifact that exist (piece to artifact produces, artifact to piece consumes); the edges are acyclic, since repair loops belong in `control.repair`. Nested inline workflows are checked the same way. |
| E193 | error | Every `control.gates` entry is a piece; every `control.repair` key is a gate and every value is a piece. |
| E194 | error | The run is among `slate.members`; `task.base_commit` equals `slate.base_commit` when both are present; each preference names this run's slate and its winner is a member or `tie`. Across files (several files to `loopmath ocp validate`), documents with the same `slate.id` share `task.id` and `base_commit`. |
| E195 | error | Signal values fit their kind: a verdict is `accept`, `reject`, `pass`, `fail` or `error`; a score is a number, or null when declared and not measured yet; `scale: fraction` values and targets lie in [0, 1]; an event carries a reference string. `acceptance_rule.score.name` names no verdict or event signal, and `acceptance_rule.requires` names no score or event signal. |
| W196 | warning | `run.receipt` is present and `producer.name` is not loopmath; only loopmath writes receipts. |
| W182 | warning | An `ext` key begins with `dev.dagr.`, the pre-rename namespace. It is still read through v0.4; write `dev.loopmath.`. |

A loopmath store takes only v0.3 documents and says so with E005, a store rule
rather than an OCP rule (`loopmath.ocp.emit.validate_strict`).

### 8.3 Migration

`loopmath ocp migrate FILE... [--out DIR]` takes v0.1, v0.2 and herdr-dagr
contract run files (through the contract converter, `loopmath.ocp.contractv3`)
to v0.3. It never mutates or overwrites its input, and it is idempotent. Each
addition is made only when the field is absent:

- `ocp` becomes `0.3`, and `run.ext["dev.loopmath.migration"]` records
  `{"from": <version>}` (`dagr/<n>` for a contract file);
- a v0.1 edge without a tier gets `reported`;
- `run.task` from `run.labels` `task`, `type` and `repo` (id from the `task`
  label, else the run id), labeled `{how: inferred, tier: heuristic}`;
- `node.vertex` from its attempts' `role.value` when they agree on one value;
- `run.provenance` `{kind: logged, chooser: habit}`;
- `run.configuration` `{source: habit}`, plus the inferred workflow and
  settings when loopmath's workflow inference returns one, with its confidence
  in `ext["dev.loopmath.inferred"]` (`tier: heuristic`).

Extension keys are kept as they are, so `dev.dagr.` keys draw W182 until their
producer writes `dev.loopmath.`. Goldens: `tests/ocp/golden/migrate/`.

### 8.4 Namespace

loopmath's extension namespace is `dev.loopmath.` from v0.3. Documents written
before the rename carry `dev.dagr.` keys; readers accept them with W182 through
v0.4 (two minor versions), and producers switch as they move to v0.3. The
retired bare `dagr.` prefix is an error, as before, and its message names
`dev.loopmath.`.

### 8.5 Paper Table 3.1 coverage

Each symbol of the paper's Table 3.1 maps to a v0.3 field, or is named as
computed (derived from recorded fields), belief (the fitted model) or
intentionally absent. `tests/ocp/test_paper_coverage.py` reads this table and
checks that every field exists in the v0.3 schema and is filled in
`examples/v0.3/full-fields.ocp.json`.

| Symbol | Where |
|---|---|
| tau, P | `run.task.id`; P is the store's task population (computed) |
| psi(tau), eta_tau | `run.task.features`, `run.task.groups`; eta is belief |
| rho, pi | `pieces[].role`; `configuration.settings` |
| W = (V, E, Gamma), q_v, K_max | `configuration.workflow` (pieces, artifacts, edges, control, width, budget) |
| x, X(W) | `configuration.id`; X(W) computed by the candidate generator |
| a, tok_s(a), reads, writes, C(a) | `attempts[]`, `attempt.cost` streams, artifact edges, `cost.usd` with `cost.tariff` |
| w, h_j | nodes, attempts, artifacts, edges; h_j computed |
| zeta, J_accept, Z_accept | `acceptance_rule`, `signals[]` (`at_attempt`); J and Z computed |
| C_run, C_accept, C_rescue | computed; `run.rescue` records the fallback used |
| u, kind(u), prod(u), cons(u) | `artifacts[]` (`kind`, `producer`, `consumers`, `version`, `supersedes`) |
| C_production, C_transfer, val(u) | computed (`transfer_tokens` when known); val is belief (P1) |
| theta, phi, (Y, Z), P_theta, ell_theta | belief |
| xi_n, F_n, S_n | one run file is xi; F_n and S_n are belief |
| d_n, N, V(S), G_n, Net_n | `run.receipt` (gain, price, payback); N intentionally absent (Q4) |
