# ADE adapters

Each adapter converts one agentic development environment (ADE) store to one
OCP v0.2 document, whose specification is in `spec/OCP.md`. An adapter is a
no-argument `Adapter` subclass in `src/loopmath/adapters/<name>.py`, sets its
canonical lowercase kebab-case `name`,
and uses `@register`. Its `discover()` and `sessions()` results are
deterministically ordered, and `emit(selection)` returns a dictionary that has
zero errors under `spec/ocp_conformance.py`.

`loopmath adapt NAME` writes JSON to stdout. `--out FILE` writes inside the current
Git worktree. Common source and selection flags are `--store PATH`, `--session
ID`, and `--workspace WORKSPACE` (all repeatable), inclusive `--since RFC3339`
and `--until RFC3339` bounds, and a positive `--limit N`. Empty filters mean all
sessions. `--store` replaces that invocation's discovered defaults.

## Adapter inventory

| Adapter | Store path or input | Format | OCP mapping | Missing from source | Relation kinds and evidence tiers |
|---|---|---|---|---|---|
| pi | `~/.pi/agent/sessions/**/*.jsonl` | Append-only JSONL entry tree, session versions 1 through 3 | One session header becomes one `unknown` node and one attempt; `cwd` and session time map to workspace/group and start time; usage-bearing assistant, tool-result, compaction, and branch-summary entries map to measured request tokens and recorded USD; a uniform model maps with the weakest applicable verified/reported tier and explicit thinking-level state maps to effort as reported; without a terminal signal, nodes and attempts remain `working` and omit outcome and end time | Task/node kind, terminal session signal or time, acceptance outcome, and explicit dependency, fan-in, subagent, launch, or artifact relations. `parentId` is conversation-entry ancestry and `parentSession` is a session fork, not a v0.2 graph relation | None emitted. This is a format-level decision: Pi emits no edges because the Pi format records no OCP relation, not because none were found in the selected sessions. No relation evidence tier applies |
| OpenCode | `~/.local/share/opencode/opencode.db` inventories ids; selected sessions come from `opencode export <sessionID>`. `--store` also accepts an export JSON file or a directory of export JSON files. | SQLite inventory plus supported OpenCode JSON export | Session to node and attempt; directory to workspace and origin; created/archive times to attempt events; assistant-message input, cache-read, cache-write, output and reasoning usage to measured cost streams; session cost to measured USD; one recorded model/variant to model/effort; mixed models to the documented attempt extension. Prompt, response, reasoning, tool, title and diff text are excluded. | No acceptance result, dependency or fan-in relation, artifact read/write lineage, role/phase evidence, or reliable terminal signal for an unarchived session. | `spawn` - `verified` only when the child's explicit `session.parentID` equals another selected session id. No inferred edges are emitted. |

The Pi adapter applies session id, workspace, and inclusive session-start time
filters before `--limit`. A filter is applied only when the session provides its
required field; sessions with unknown workspace or start time are retained and
counted separately from omissions. It orders sessions by ascending parsed start
time, then id and path, with unknown start times last. Explicit `--store` paths
replace discovery and may name either a JSONL file or a directory searched
recursively.
| `otel-genai` | Explicit repeatable `--store FILE`; no canonical local store is discovered. Claude Code 2.1.257 trace capture requires `CLAUDE_CODE_ENABLE_TELEMETRY=1`, `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`, `OTEL_TRACES_EXPORTER=otlp`, `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/json`, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:4318/v1/traces`, and `OTEL_LOG_TOOL_DETAILS=1`. Claude accepts trace exporters `console`, `otlp`, or `none`; protocol variables `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` and `OTEL_EXPORTER_OTLP_PROTOCOL` accept `grpc`, `http/json`, or `http/protobuf`; endpoint variables are `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` and `OTEL_EXPORTER_OTLP_ENDPOINT`. Claude has no file exporter, so an OpenTelemetry Collector receives OTLP and uses a `file` exporter with `path: /capture/claude-traces.otlp.jsonl` on a bind-mounted capture directory. `CLAUDE_CODE_PROPAGATE_TRACEPARENT=1` is needed when using a custom `ANTHROPIC_BASE_URL`. | Whole OTLP JSON object or Collector file-exporter JSONL; `resourceSpans` / `scopeSpans` / `spans`, typed `AnyValue`, and decimal-string 64-bit integers. | Complete traces become groups; Claude sessions and agents plus Codex conversations become nodes/attempts; timestamps, reported model/provider/effort, four measured token streams, request count, workspace, lifecycle events, and exact lineage map to OCP core. The injected base-class pricing hook adds settled loopmath-priced core USD for complete known-model usage; unpriced and provisional attempts retain measured tokens, are counted and surfaced, and never default USD or emit an estimate. Span `cost_usd` is only the documented cross-check extension because the OTel convention defines no cost attribute. Codex supports `[otel].trace_exporter` values `otlp-http` and `otlp-grpc`; the current binary emits `codex.*` telemetry and vendor-neutral `gen_ai.usage.*` fields, while this adapter maps only unambiguous vendor-neutral fields and current aliases. | No acceptance signal. The minimal real Claude Code 2.1.257 capture emitted `agent_id` but not `parent_agent_id`, `effort`, or `cost_usd`; all three beta fields are supported when present. Codex public documentation does not define a stable complete span/attribute vocabulary or guarantee inbound `TRACEPARENT` consumption. | `spawn` is `verified` for exact `parent_agent_id` to `agent_id` resolution or exact Claude Agent/Task span ancestry. `launch` is `verified` only when a cross-harness root's `parentSpanId` exactly matches a recognized launcher tool-execution `spanId` in the same `traceId`. No timing, text, command, path, workspace, or trace-id-only relation is emitted. |
| omp | `~/.omp/agent/sessions/<dir-encoded>/<timestamp>_<sessionId>.jsonl`; task children use `<timestamp>_<parentSessionId>/<agent>.jsonl` | Append-only JSONL v3 grouped by encoded canonical working directory; OMP 18.1.5 writes a fixed-width `title` slot before the `session` header | One OCP node and attempt per session; map header time and cwd, model changes, message roles, validated token usage, host-priced settled cost, and task tool calls/results/jobs; classify child sessions from `session_init` | Task-spawned child headers have no durable parent-session foreign key; parent task job ids do not equal child session ids; interrupted async completion may be unavailable; incomplete usage and unsettled pricing remain explicit extension diagnostics | `spawn`: heuristic only when exactly one parent job matches the non-null child `session_init` task and agent plus overlapping timestamps; entry ancestry from `parentId`, task call/result pairing from `toolCallId`, and child-session classification from `session_init` are verified source facts but are not additional inter-node relation kinds |
| Orca | `~/Library/Application Support/orca/orchestration.db` | SQLite, undocumented and read-only | Orca runs become groups, tasks become nodes, local worker dispatches become attempts keyed by native ids, effective `start_options` agent/model/effort become reported harness configuration, and typed `escalation` (send-back), `worker_done` and `merge_ready` (completion or approval-ready), and `decision_gate` (gate) messages become metadata-only events | No tokens or cost. The dispatch-to-vendor-session join remains open after read-only schema, profile, hook, and CLI context checks found no stable historical key; attempts therefore omit `session` and `origin`. Task, result, and message text is omitted. `start_options.agent` identifies a harness, not an OCP task role. | `spawn` from `tasks.parent_id`: verified. Effective agent/model/effort: reported launch configuration. Selected message enum values: verified as stored; completion or readiness remains reported, not verified acceptance. Worker state becomes a reported OCP status/outcome; uncertain start or stop states stay working. No heuristic relations. |
| Paseo | `~/.paseo/agents/<flattened-cwd>/<agent-id>.json` | JSON object per agent | Each agent becomes one node and one attempt. `provider` maps to harness, `config.model` and `config.thinkingOptionId` to reported model and effort, `cwd` to reported origin workspace, timestamps to attempt bounds, and a corroborated `persistence.sessionId` to `attempt.session`. A closed agent is `settled_unverified`, never accepted work. | No tokens, cost, turns, tools, acceptance result, or parent/launcher field. Titles, system prompts, feature descriptions, and unrelated persistence metadata are omitted. | No graph edge is emitted: the store has no distinct parent or launcher endpoint. The vendor-session identity correlation is verified when `persistence.sessionId` equals the available native handles; all 3 local records also matched vendor-session filenames in a metadata-only 2026-09-02 smoke check. Model, effort, workspace, and terminal state are reported. No heuristic relations. |
| Atrium | `~/.atrium/timeline.db`, `~/.atrium/tasks.db` | SQLite, hook-fed and read-only | Selected `session-end`, `agent-message`, and `subagent-stop` timeline records and populated `task_run_segments.adapter_session_id` records become metadata-only events. Their workspace, source pane, optional related endpoint kind and id, and evidence tier use the documented event extension below. Atrium workspace ids are deterministic adapter session ids. | No tokens, cost, model, effort, stable task node, or acceptance result. Runtime pane ids are not promoted to OCP nodes, so no attempt or core graph edge is emitted. Prompt, message, title, body, transcript, actor, and task text is never selected. Segment closure, session end, and subagent stop do not imply acceptance. The inspected local source is stale since 2026-08-14, its pane-to-session coverage is sparse, and all 86 local `subagent-stop` rows have an empty `subagent` value. | Pane-to-session and source-to-target-pane message endpoint pairs are verified because both endpoints are stored in the same record. A `subagent-stop` kind and source pane are verified stored facts, but an empty subagent id is omitted and never promoted to a `spawn` relation. `agent-message` is preserved as a typed send-back-capable event, not interpreted as rejection or retry. No reported or heuristic relation and no core graph edge is emitted. |
| Orca fix-round addendum | Same store and mapping as the Orca row above | Same read-only SQLite input | Document diagnostics count malformed `start_options` and selected typed-message payloads by fixed reason. The affected attempt or event remains, but unverified configuration or linkage fields stay absent. Counts also print to stderr in fixed order. | Malformed values themselves are never emitted. | No relation or evidence tier changes. This addendum adds observability only. |
| Paseo fix-round addendum | Same store and mapping as the Paseo row above | Same read-only JSON input | Records with missing, invalid-type, or unknown `lastStatus` are excluded by a named count instead of becoming working. `attempt.session` now requires `persistence.sessionId`, `persistence.nativeHandle`, and `runtimeInfo.sessionId` all to be present strings and equal. Timestamp defects and every selection exclusion are counted; all fixed counts print to stderr. | Unknown status values and malformed timestamp values are never emitted. A session correlate is absent unless both documented corroborators match. | No graph relation is added. The session correlation remains verified only when all three stored values match exactly. |
| Atrium fix-round addendum | Same stores and mapping as the Atrium row above | Same read-only SQLite input | Unsupported timeline rows, unusable segment session ids, and every selection exclusion are counted in document diagnostics and printed to stderr in fixed order. | Excluded row content is never selected or emitted. | No relation or evidence tier changes. This addendum adds observability only. |
| Paseo vendor-session ruling addendum | Same store and mapping as the Paseo row above | Same read-only JSON input | Each Paseo agent remains one node keyed by its own `node.id`; a fully corroborated vendor session remains only the opaque `attempt.session` correlate. Readers must never key or deduplicate nodes on a bare session id. | Cross-document qualified session reconciliation is out of scope tonight. | No `launch` edge is emitted. A separate vendor node or `launch` edge is allowed only for a distinct independently parsed execution with evidenced causality. |
| bb | `~/.bb/bb.db` | SQLite (Drizzle schema); current `events.provider_thread_id` with legacy `data.providerThreadId` fallback | One node/attempt per bb thread; `environments.path` is used for exact selection and a stable opaque workspace group under `metadata_only`; model and effort overrides are reported when present. A uniquely resolved provider UUID adds the existing Claude Code or Codex session as a second node/attempt, whose vendor-log tokens are priced through `prices.toml`. bb running token snapshots are ignored. | No native cost or acceptance result. bb merges cache reads and writes in its running token snapshot, so that snapshot cannot supply loopmath's four streams. Queued cross-thread messages have no truthful OCP v0.2 edge mapping and are counted in adapter coverage. | Current `threads.parent_thread_id` hierarchy -> `spawn` (`verified`, mutable relation rather than immutable creation provenance); `threads.source_thread_id` with `origin_kind = fork` -> `spawn` (`verified`, native kind retained in edge ext); one UUID across bb events joined to the matching vendor session -> `launch` (`heuristic`). Vendor session correspondence is correlation, not causal launch evidence. |

Every adapter row must name each relation kind it emits and its evidence tier.
Relations recorded explicitly by the store are `verified`; relations inferred
from timing, text, or path patterns are `heuristic`; claims accepted from a
session, harness, or model are `reported`.

## Binding OCP v0.2 extension-key policy

Adapters emit OCP v0.2. They do not introduce another schema version. Data that
OCP v0.2 cannot represent may appear only in an `ext` object at the narrowest
applicable entity, under the dotted namespace `dev.dagr.adapter.<name>`. Extension
data must not duplicate or weaken a core field, must obey the document's privacy
profile, and must not be required to understand the core graph. Consumers may
ignore it as OCP v0.2 permits.

Every extension namespace and every member path an adapter emits must have a
row in the proposal table below before the key ships. Each row is explicitly an
OCP v0.3 proposal, not a claim that the key is part of OCP v0.2 or guaranteed to
enter v0.3. Undocumented adapter extension keys are forbidden.

## OCP v0.3 extension proposals

| Adapter | Ext namespace | Member path | OCP entity | JSON type | Semantics and evidence | Why OCP v0.2 is insufficient | Proposed v0.3 disposition |
|---|---|---|---|---|---|---|---|
| `otel-genai` | `dev.dagr.adapter.otel-genai` | `reported_cost_usd` | attempt | number | Nonnegative sum of every valid source-reported `cost_usd` or `cost_usd_micros` value over mapped requests. When `missing_reported_cost` is nonzero this is the partial measured component; core `cost.usd` adds host pricing only for uncovered requests. | OCP v0.2 has no distinct standard field for the measured component of a mixed reported-and-priced total. | Consider a standard source-reported cost component distinct from computed cost. |
| `otel-genai` | `dev.dagr.adapter.otel-genai` | `pricing_refusal_reason` | attempt | string | Explicit reason that uncovered request usage was not sent to host pricing: `mixed request models` when mapped requests have multiple valid models, or `missing request model` when any mapped request lacks a valid model. | OCP v0.2 has no attempt-level pricing refusal reason. | Consider a standard pricing diagnostics field and reason vocabulary. |
| `otel-genai` | `dev.dagr.adapter.otel-genai` | `input_span_count` | run | integer | Count of valid, deduplicated OTLP spans observed before selection and mapping. | OCP v0.2 has no adapter-ingestion accounting. | Consider a standard ingestion diagnostics record. |
| `otel-genai` | `dev.dagr.adapter.otel-genai` | `mapped_span_count` | run | integer | Count of selected spans assigned to emitted logical attempts. | OCP v0.2 has no adapter-ingestion accounting. | Consider a standard ingestion diagnostics record. |
| `otel-genai` | `dev.dagr.adapter.otel-genai` | `omissions` | run | array | Deterministically sorted summaries of source data omitted or filtered during conversion. | OCP v0.2 has no structured conversion-loss report. | Consider a standard ingestion diagnostics record. |
| `otel-genai` | `dev.dagr.adapter.otel-genai` | `omissions[].reason` | run | string | Stable machine-readable omission reason code. | OCP v0.2 has no structured conversion-loss report. | Consider a standard ingestion diagnostics reason vocabulary. |
| `otel-genai` | `dev.dagr.adapter.otel-genai` | `omissions[].count` | run | integer | Positive occurrence count for the associated omission reason. | OCP v0.2 has no structured conversion-loss report. | Consider a standard ingestion diagnostics count. |
| omp | `dev.dagr.adapter.omp` | `message_role_counts.<role>` | attempt | integer | Count of messages carrying each recognized OMP role; verified from message records. Unknown role labels are combined under `other`. | OCP v0.2 has no aggregate message-role fields. | Consider a standard per-attempt interaction summary. |
| omp | `dev.dagr.adapter.omp` | `record_parent_links` | attempt | integer | Count of non-null `parentId` values that name a record id in the same file; verified by exact id equality. | OCP nodes and attempts do not represent native transcript-entry trees. | Keep producer-specific unless entry-tree summaries become common. |
| omp | `dev.dagr.adapter.omp` | `unresolved_parent_links` | attempt | integer | Count of non-null `parentId` values with no record id match in the same file; verified by exact id comparison. | OCP v0.2 has no native-store integrity counters. | Consider a standard source-integrity summary. |
| omp | `dev.dagr.adapter.omp` | `task_calls` | attempt | integer | Count of assistant `task` tool-call records; verified from typed tool-call entries. | OCP v0.2 represents cross-node edges, not native tool activity. | Consider a standard tool-activity summary. |
| omp | `dev.dagr.adapter.omp` | `linked_task_results` | attempt | integer | Count of task results whose `toolCallId` exactly names a task call in the same session; verified by id equality. | OCP v0.2 has no call/result pairing entity. | Consider standard call/result provenance. |
| omp | `dev.dagr.adapter.omp` | `linked_task_jobs` | attempt | integer | Count of linked task results whose async `jobId` exactly names a progress entry; verified by id equality. | OCP v0.2 has no native async-job entity. | Consider standard async-job provenance. |
| omp | `dev.dagr.adapter.omp` | `source_files.candidates` | run | integer | Count of unique JSONL source paths considered after deterministic recursive discovery. | OCP v0.2 has no source-ingestion coverage summary. | Consider a standard extraction-coverage record. |
| omp | `dev.dagr.adapter.omp` | `source_files.loaded` | run | integer | Count of valid v3 files retained after file validation and native session-id deduplication. | OCP v0.2 has no source-ingestion coverage summary. | Consider a standard extraction-coverage record. |
| omp | `dev.dagr.adapter.omp` | `source_findings[].reason` | run | string | Stable source finding: `bad_header`, `decode_failure`, `duplicate_record_id`, `duplicate_session_id`, `duplicate_source_path`, `invalid_header_timestamp`, `invalid_record_id`, `invalid_record_timestamp`, `invalid_session_id`, `malformed_json`, `malformed_parent_id`, `malformed_record_type`, `non_object_record`, `read_failure`, `store_path_unavailable`, `store_scan_failure`, `unsupported_version`, or `workspace_unknown`. | OCP v0.2 cannot explain source exclusions or unknown fields. | Consider a standard extraction-diagnostic vocabulary. |
| omp | `dev.dagr.adapter.omp` | `source_findings[].effect` | run | string | Effect of the source finding: `file_excluded`, `record_excluded`, `field_unknown`, or `path_omitted`. | OCP v0.2 cannot distinguish exclusion scope from an unknown value. | Consider a standard extraction-diagnostic scope. |
| omp | `dev.dagr.adapter.omp` | `source_findings[].count` | run | integer | Number of source findings with the paired reason and effect; verified from parsing and validation. | OCP v0.2 has no source-integrity counters. | Consider a standard extraction-diagnostic count. |
| omp | `dev.dagr.adapter.omp` | `selection_filters[].name` | run | string | Filter name in fixed application order: `session_id`, `workspace`, `since`, then `until`. | OCP v0.2 does not describe extraction selection. | Consider a standard selection-coverage record. |
| omp | `dev.dagr.adapter.omp` | `selection_filters[].active` | run | boolean | Whether the named selection filter was requested. | OCP v0.2 does not describe extraction selection. | Consider a standard selection-coverage record. |
| omp | `dev.dagr.adapter.omp` | `selection_filters[].unknown` | run | integer | Sessions retained because the source field needed by an active filter was unknown; filters are never applied to missing source fields. | OCP v0.2 cannot distinguish retained unknowns from matches. | Consider a standard selection-coverage record. |
| omp | `dev.dagr.adapter.omp` | `selection_filters[].excluded` | run | integer | Sessions excluded by the active filter after earlier filters; verified from available source values. | OCP v0.2 cannot report filter exclusion counts. | Consider a standard selection-coverage record. |
| omp | `dev.dagr.adapter.omp` | `limit_omitted` | run | integer | Sessions omitted after field filters solely because of the requested limit. | OCP v0.2 cannot report bounded-selection omissions. | Consider a standard selection-coverage record. |
| omp | `dev.dagr.adapter.omp` | `spawn_matching.child_sessions` | run | integer | Sessions carrying an explicit `session_init` child marker. | OCP v0.2 edges do not expose candidate coverage. | Consider a standard relation-diagnostic summary. |
| omp | `dev.dagr.adapter.omp` | `spawn_matching.matched_children` | run | integer | Child sessions receiving a heuristic spawn edge after exactly one job matched across all parents. | OCP v0.2 edges do not expose candidate coverage. | Consider a standard relation-diagnostic summary. |
| omp | `dev.dagr.adapter.omp` | `spawn_matching.zero_candidate_children` | run | integer | Child sessions with no eligible exact task-and-agent job match, including missing signatures or timestamps. | OCP v0.2 cannot explain omitted inferred edges. | Consider a standard relation-diagnostic summary. |
| omp | `dev.dagr.adapter.omp` | `spawn_matching.ambiguous_children` | run | integer | Child sessions matching jobs from more than one parent; no edge is emitted. | OCP v0.2 cannot explain omitted inferred edges. | Consider a standard relation-diagnostic summary. |
| omp | `dev.dagr.adapter.omp` | `spawn_matching.multiple_job_children` | run | integer | Child sessions matching more than one job across all parents; no edge is emitted even when all jobs belong to one parent. | OCP v0.2 cannot explain omitted inferred edges. | Consider a standard relation-diagnostic summary. |
| omp | `dev.dagr.adapter.omp` | `spawn_matching.missing_signature_children` | run | integer | Child sessions whose `session_init` lacks a non-empty string task or agent; no edge is emitted. | OCP v0.2 cannot explain omitted inferred edges. | Consider a standard relation-diagnostic summary. |
| omp | `dev.dagr.adapter.omp` | `malformed_parent_links` | attempt | integer | Count of non-null, non-string `parentId` values; each is also counted as unresolved. | OCP v0.2 has no native-store integrity counters. | Consider a standard source-integrity summary. |
| omp | `dev.dagr.adapter.omp` | `task_jobs` | attempt | integer | Count of unique valid job ids parsed across every `TaskToolDetails.progress[]` and `results[]` entry, including unmatched jobs. | OCP v0.2 has no native task-job entity. | Consider standard async-job provenance. |
| omp | `dev.dagr.adapter.omp` | `task_findings[].reason` | attempt | string | Stable task finding: `duplicate_task_call_id`, `duplicate_task_job`, `duplicate_task_result`, `malformed_async_job_id`, `malformed_task_call`, `malformed_task_details`, `malformed_task_job`, `malformed_task_result`, `unmatched_async_job_id`, `unmatched_task_call`, `unmatched_task_job`, or `unmatched_task_result`. | OCP v0.2 cannot explain native task exclusions and failed linkages. | Consider a standard tool-linkage diagnostic vocabulary. |
| omp | `dev.dagr.adapter.omp` | `task_findings[].count` | attempt | integer | Number of task findings carrying the paired reason; verified from typed record parsing and exact id joins. | OCP v0.2 has no tool-linkage diagnostic counts. | Consider a standard tool-linkage diagnostic count. |
| omp | `dev.dagr.adapter.omp` | `model_calls` | attempt | integer | Count of all assistant-message and `model_usage` calls encountered. | OCP v0.2 cost lacks source-call coverage. | Consider a standard usage-coverage summary. |
| omp | `dev.dagr.adapter.omp` | `usage_complete_calls` | attempt | integer | Calls with complete finite non-negative integer token fields and a valid matching `totalTokens`. | OCP v0.2 cannot report measured-call coverage. | Consider a standard usage-coverage summary. |
| omp | `dev.dagr.adapter.omp` | `usage_unknown_calls` | attempt | integer | Calls whose token usage is missing or invalid; any nonzero value suppresses the attempt's aggregate core cost rather than creating a partial total. | OCP v0.2 cannot distinguish absent cost from incomplete source measurements. | Consider a standard usage-coverage summary. |
| omp | `dev.dagr.adapter.omp` | `usage_findings[].reason` | attempt | string | Stable reason: provider/model `missing`, `malformed`, or `oversized`; usage `missing` or `not_object`; a required `input`, `cacheRead`, `cacheWrite`, or `output` field `missing` or `invalid`; or `totalTokens` `missing`, `invalid`, or `mismatch`, with the emitted reason prefixed by `provider_`, `model_`, or `usage_`. | OCP v0.2 cannot explain incomplete usage or unknown attribution. | Consider a standard usage-diagnostic vocabulary. |
| omp | `dev.dagr.adapter.omp` | `usage_findings[].count` | attempt | integer | Number of usage findings carrying the paired reason. | OCP v0.2 has no usage-diagnostic counts. | Consider a standard usage-diagnostic count. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].provider` | attempt | string or null | OMP provider label for one accounting group, or null with a corresponding unknown reason. | OCP v0.2 has one attempt model and cannot retain mixed-provider usage attribution. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].model` | attempt | string or null | OMP model label for one accounting group, or null with a corresponding unknown reason. | OCP v0.2 has one attempt model and cannot retain mixed-model usage attribution. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].calls` | attempt | integer | Assistant-message and `model_usage` calls in this exact provider/model group. | OCP v0.2 cost has only an aggregate request count. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].usage_complete_calls` | attempt | integer | Calls with complete validated token usage in this provider/model group. | OCP v0.2 cannot report per-model measured-call coverage. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].usage_unknown_calls` | attempt | integer | Calls with missing or invalid usage in this provider/model group. | OCP v0.2 cannot report per-model unknown-call coverage. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].known_input_tokens` | attempt | integer | Sum of validated uncached input tokens from complete calls in this provider/model group. | OCP v0.2 cannot retain mixed-model usage attribution. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].known_cached_input_tokens` | attempt | integer | Sum of validated cache-read tokens from complete calls in this provider/model group. | OCP v0.2 cannot retain mixed-model usage attribution. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].known_cache_creation_tokens` | attempt | integer | Sum of validated cache-write tokens from complete calls in this provider/model group. | OCP v0.2 cannot retain mixed-model usage attribution. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].known_output_tokens` | attempt | integer | Sum of validated output tokens from complete calls in this provider/model group. | OCP v0.2 cannot retain mixed-model usage attribution. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].unknown_reasons[].reason` | attempt | string | Usage or provider/model reason affecting this accounting group, using the `usage_findings[].reason` vocabulary. | OCP v0.2 cannot explain per-model unknown attribution. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `usage_by_model[].unknown_reasons[].count` | attempt | integer | Number of occurrences of the paired unknown reason in this provider/model group. | OCP v0.2 has no per-model diagnostic counts. | Consider a standard per-model usage breakdown. |
| omp | `dev.dagr.adapter.omp` | `pricing_by_model[].provider` | attempt | string or null | Provider label of the usage group passed to host pricing. | OCP v0.2 cannot retain mixed-model pricing attribution. | Consider a standard per-model pricing breakdown. |
| omp | `dev.dagr.adapter.omp` | `pricing_by_model[].model` | attempt | string or null | Model label passed through the public adapter pricing hook. | OCP v0.2 cannot retain mixed-model pricing attribution. | Consider a standard per-model pricing breakdown. |
| omp | `dev.dagr.adapter.omp` | `pricing_by_model[].priced` | attempt | boolean | Exact `PricingResult.priced` state; true only for settled USD. | OCP v0.2 cost cannot explain why USD is absent. | Consider a standard pricing-result record. |
| omp | `dev.dagr.adapter.omp` | `pricing_by_model[].provisional` | attempt | boolean | Exact `PricingResult.provisional` state. Provisional OMP pricing stays provisional, never becomes settled USD, and never collapses into ordinary unpriced. | OCP v0.2 has no provisional-price state. | Consider a standard provisional pricing state. |
| omp | `dev.dagr.adapter.omp` | `pricing_by_model[].reason` | attempt | string or null | Exact host pricing reason; null only for settled pricing, the standard provisional reason for estimates, and non-null for ordinary unpriced results. | OCP v0.2 cost cannot preserve pricing disposition. | Consider a standard pricing-result record. |
| omp | `dev.dagr.adapter.omp` | `pricing_by_model[].usd` | attempt | number or null | Exact settled `PricingResult.usd`; only settled values may also enter aggregate core cost. It is always null for provisional and ordinary unpriced results. | OCP v0.2 cannot retain per-model settled prices. | Consider a standard per-model pricing breakdown. |
| omp | `dev.dagr.adapter.omp` | `pricing_by_model[].estimate_usd` | attempt | number or null | Exact provisional estimate. It remains only `estimate_usd` in the extension, never becomes core or settled `usd`, and is null for settled and ordinary unpriced results. | OCP v0.2 has no field for explicitly provisional estimates. | Consider a standard provisional pricing field. |

## OMP 18.1.5 evidence notes

- A 2026-09-02 isolated probe found no pre-existing `omp` executable or default
  OMP session directory. It installed `@oh-my-pi/pi-coding-agent` 18.1.5 and a
  compatible Bun runtime under a throwaway prefix, used a zero-cost local model,
  and did not modify Pi files.
- Relocating the agent base created
  `<probe-agent-dir>/sessions/<dir-encoded>/<timestamp>_<sessionId>.jsonl`. This
  verifies that the default agent base `~/.omp/agent` yields the store root
  `~/.omp/agent/sessions/`.
- The probe produced two JSONL files and 17 records. Each `title` occupied a
  256-byte first line, the v3 `session` header was second, and later entries
  formed an explicit tree through `id` and `parentId`.
- The parent task result named its call exactly through `message.toolCallId`,
  and its progress id exactly matched `details.async.jobId`. The child was
  explicitly classified by `session_init`; its task and agent label matched the
  parent progress entry.
- No parent-to-child session key was present. The child had no `parentSession`,
  the parent job id differed from the child session id, and
  `session_init.spawns` was a policy rather than a parent pointer. Any emitted
  `spawn` edge therefore remains heuristic.
- The committed synthetic fixture is smaller than the probe and contains only
  invented values: two files and 13 records comprising two `title`, two
  `session`, two `model_change`, one `session_init`, and six `message` records.
- The synthetic child file uses the observed nested
  `<parent-session-stem>/<agent>.jsonl` shape rather than a second top-level
  session-file shape.
| Atrium | `dev.dagr.adapter.atrium` | `workspace_id` | event | string | Atrium workspace id stored on the source record; verified | Events cannot name a workspace or group without pretending that a runtime event belongs to a task node | Add an optional event workspace/group reference |
| Atrium | `dev.dagr.adapter.atrium` | `pane_id` | event | string | Source or observing runtime pane id stored on the source record; verified | A pane is a runtime locator, explicitly not an OCP node id | Add a runtime-endpoint record distinct from task nodes |
| Atrium | `dev.dagr.adapter.atrium` | `related_kind` | event | string | Optional endpoint category `session`, `target-pane`, or `subagent`, selected mechanically when the typed source record has a non-empty related id; verified | The closed OCP edge vocabulary has no neutral runtime correlation or message relation | Add typed runtime endpoint relations without graph scheduling semantics |
| Atrium | `dev.dagr.adapter.atrium` | `related_id` | event | string | Optional opaque session, target-pane, or subagent identifier co-recorded with `pane_id`; verified and omitted when empty | Core events have at most one node and one attempt reference, neither of which may stand for a runtime pane | Add a second typed runtime endpoint to events |
| Atrium | `dev.dagr.adapter.atrium` | `tier` | event | string | Evidence tier for the stored event correlation fields; currently always `verified` | Core events have no evidence tier | Add an event relation evidence tier using the existing tier vocabulary |
| Atrium | `dev.dagr.adapter.atrium` | `diagnostics` | document | array | Fix-round proposal: a fixed-order record for every Atrium diagnostic reason, including zero counts; mechanically counted from read-only rows and selection decisions | OCP v0.2 has no extraction-diagnostic record | Add document-level extraction diagnostics |
| Atrium | `dev.dagr.adapter.atrium` | `diagnostics[].reason` | document | string | One of `timeline_kind_unsupported`, `segment_adapter_session_id_missing`, `segment_adapter_session_id_invalid_type`, `filter_session`, `filter_workspace`, `filter_before_since`, `filter_after_until`, or `filter_limit`; categories are fixed and contain no source content | OCP v0.2 has no reason vocabulary for omitted source records | Add a closed adapter-diagnostic reason vocabulary |
| Atrium | `dev.dagr.adapter.atrium` | `diagnostics[].count` | document | integer | Nonnegative count for the paired reason, always emitted and printed even when zero; mechanically measured | OCP v0.2 has no extraction exclusion counts | Add a nonnegative diagnostic count |
| Orca | `dev.dagr.adapter.orca` | `diagnostics` | document | array | Fix-round proposal superseding the pre-review no-extension statement retained below: a fixed-order record for every Orca malformed-JSON reason, including zero counts; mechanically counted from selected read-only rows | OCP v0.2 has no extraction-diagnostic record | Add document-level extraction diagnostics |
| Orca | `dev.dagr.adapter.orca` | `diagnostics[].reason` | document | string | One of `start_options_missing`, `start_options_invalid_type`, `start_options_invalid_json`, `start_options_json_not_object`, `message_payload_missing`, `message_payload_invalid_type`, `message_payload_invalid_json`, or `message_payload_json_not_object`; categories contain no source content | OCP v0.2 has no reason vocabulary for malformed adapter fields | Add a closed adapter-diagnostic reason vocabulary |
| Orca | `dev.dagr.adapter.orca` | `diagnostics[].count` | document | integer | Nonnegative count for the paired reason, always emitted and printed even when zero; mechanically measured | OCP v0.2 has no malformed-field counts | Add a nonnegative diagnostic count |
| Paseo | `dev.dagr.adapter.paseo` | `diagnostics` | document | array | Fix-round proposal: a fixed-order record for every Paseo unknown, malformed, session-omission, and selection-filter reason, including zero counts; mechanically counted from read-only records and selection decisions | OCP v0.2 has no extraction-diagnostic record | Add document-level extraction diagnostics |
| Paseo | `dev.dagr.adapter.paseo` | `diagnostics[].reason` | document | string | One of `record_last_status_missing`, `record_last_status_invalid_type`, `record_last_status_unknown`, `session_id_missing_or_invalid`, `session_corroborator_missing_or_invalid`, `session_corroborator_mismatch`, `timestamp_missing`, `timestamp_invalid_type`, `timestamp_invalid_value`, `timestamp_timezone_missing`, `filter_session`, `filter_workspace`, `filter_created_at_unknown`, `filter_before_since`, `filter_after_until`, or `filter_limit`; categories contain no source content | OCP v0.2 has no reason vocabulary for adapter omissions or filters | Add a closed adapter-diagnostic reason vocabulary |
| Paseo | `dev.dagr.adapter.paseo` | `diagnostics[].count` | document | integer | Nonnegative count for the paired reason, always emitted and printed even when zero. Source-field and session-correlation counts cover every scanned record. Each `filter_*` count uses the first matching selection reason in session, workspace, unknown creation time, before since, after until, then limit order; records with unusable status emit no entity and are named by `record_last_status_*`. | OCP v0.2 has no unknown-field or selection-exclusion counts | Add a nonnegative diagnostic count |
| Paseo | `dev.dagr.adapter.paseo` | `diagnostics[].reason` | document | string | Fixture-contract addendum superseding only the preceding Paseo reason vocabulary: the fixed-order vocabulary additionally includes `record_unreadable`, `record_invalid_json`, `record_json_not_object`, and `record_not_agent`; each names one safely skipped file without retaining its path or content | OCP v0.2 has no reason vocabulary for unreadable or non-agent source records | Extend the closed adapter-diagnostic reason vocabulary |
| bb | `dev.dagr.adapter.bb` | `coverage.selected_threads` | document | integer | Number of bb thread rows remaining after all common selection filters and the limit. | OCP lists emitted nodes but does not distinguish bb wrapper threads from mechanically joined vendor nodes in an aggregate. | Add a standard producer coverage record, or retain as adapter extension. |
| bb | `dev.dagr.adapter.bb` | `coverage.joined_vendor_sessions` | document | integer | Selected bb threads whose single event UUID matched exactly one parseable vendor session. | OCP carries each launch edge but no aggregate join-coverage count. | Add a standard relation coverage record, or retain as adapter extension. |
| bb | `dev.dagr.adapter.bb` | `coverage.unjoined_vendor_sessions` | document | integer | Selected bb threads not joined because the UUID was missing/conflicting, the provider was unsupported, ownership was ambiguous, or no matching parseable vendor session existed. | OCP cannot count source relations withheld because an endpoint was unavailable. | Add a standard omitted-relation count with reasons in a future diagnostics entity. |
| bb | `dev.dagr.adapter.bb` | `coverage.unpriced_vendor_sessions` | document | integer | Joined vendor sessions whose parsed model/tokens could not be priced by loopmath's price table. | OCP permits cost without USD or no cost, but has no aggregate pricing-coverage count. | Add a standard pricing coverage record, or retain as adapter extension. |
| bb | `dev.dagr.adapter.bb` | `coverage.filtered_relations` | document | integer | Explicit parent/fork relations withheld because a selection filter excluded their other endpoint. | OCP edges cannot name a filtered-out node and has no omitted-edge counter. | Add a standard omitted-relation count in a future diagnostics entity. |
| bb | `dev.dagr.adapter.bb` | `coverage.unrepresented_queued_messages` | document | integer | Queued cross-thread messages touching selected threads that were not mapped because no OCP v0.2 edge kind has send-back message semantics. Only endpoint metadata is counted; content is never read. | The closed OCP v0.2 edge vocabulary has no cross-thread message/send-back relation. | Consider a typed message relation; retain the count until then. |
| bb | `dev.dagr.adapter.bb` | `coverage.ignored_bb_token_snapshots` | document | integer | `thread/tokenUsage/updated` rows on selected threads intentionally excluded from tokens and cost; joined vendor usage is authoritative. | OCP has no provenance field saying a conflicting/coarser token source was deliberately ignored. | Add token-source provenance, or retain as adapter extension. |
| bb | `dev.dagr.adapter.bb` | `native_relation` | edge | string (`fork`) | Records that an OCP `spawn` edge came specifically from bb's explicit `source_thread_id` plus `origin_kind = fork`, both checked in the SQLite row. | OCP v0.2 has no fork edge kind, and projecting fork to `spawn` otherwise loses the native distinction. | Consider a core `fork` lineage kind; retain this refinement if the edge vocabulary stays closed. |

## Vendor session correlation and launch semantics

`attempt.session` is the canonical opaque vendor-session correlate whenever it is
reliably known. It is not a globally unique Graph identity key, and it does not by
itself establish topology or causality. Use the field only when the ADE attempt is the
vendor execution, or when a reliable session id is the only available correlate.

Emit a separate vendor node only when a genuinely distinct controller or wrapper
attempt exists and the vendor execution record was independently ingested. Put
`session` on the vendor attempt, not on the controller. An independently ingested
vendor execution is emitted and priced once. Multiple references to that execution do
not justify duplicate cost or ambiguous launch claims.

Emit a `launch` edge and matching `origin.launched_by` only when the launcher is a
distinct node in the same document and established causal evidence connects it to the
vendor attempt. Never fabricate a node or edge from a session id alone. When the
launcher is missing or ambiguous, emit no launch edge and set `launched_by` to null.

The reader does not deduplicate on bare `session`. Provenance-aware reconciliation is
deferred and must qualify the correlate by namespace plus session, preserve aliases,
count cost once, and diagnose conflicting measurements. Tonight's measured known gap
is that overlapping A2 and native sources may collide on shared node ids, while A4 and
native sources can double-count despite carrying the same session correlate.
## Extraction boundary and host services

Adapter modules are extraction-ready: they may import the OCP schema/API, the
adapter base and registry API, and the Python standard library, but never loopmath
graph, ingest, pricing, or report internals. The stdlib-only base constructor is
`Adapter(services: AdapterServices | None = None, /)`. Its narrow host methods
are `configure_services(services: AdapterServices, /) -> None`,
`price_usage(model: str | None, usage: Usage | None, /) -> PricingResult`,
`resolve_vendor_session(kind: str, correlate: str, /) -> VendorSession | None`,
and `producer_version() -> str | None`. `Usage` has the OCP cost fields
`input_tokens`, `cached_input_tokens`, `cache_creation_tokens`, `output_tokens`,
and `basis`; `PricingResult.to_ocp()` returns exactly those fields plus `usd`
when priced, retains measured tokens without `usd` when unpriced, and returns
`None` when no complete usage exists. `VendorSession` contains only `kind`, the
host-normalized opaque `correlate`, `run_id`, `started_at`, `wall_s`, `model`,
`model_tier`, `effort`, `usage`, and `match_evidence`; resolution returns a
session only for one parseable match, while an absent service, unsupported
kind, invalid correlate, missing match, or ambiguous match returns `None`.
The injected callable signatures are `PriceUsageHook = Callable[[str | None,
Usage | None], PricingResult]` and `VendorSessionHook = Callable[[str, str],
VendorSession | None]`; `AdapterServices` stores those as `price_usage` and
`vendor_session` alongside `producer_version`. Loopmath's host factory is
`default_adapter_services(*, claude_code_root: str | Path | None = None,
codex_root: str | Path | None = None, price_path: str | Path | None = None,
producer_version: str = loopmath.__version__) -> AdapterServices`; the loopmath type in
that default is evaluated only in the host module and never imported by an
adapter.
No-argument construction installs no services: pricing returns an explicit
unpriced result while preserving valid usage, vendor resolution returns
`None`, and producer version returns `None` so the optional version member is
omitted. The loopmath CLI injects `default_adapter_services()`, whose implementation
lives outside the adapter package, uses loopmath's parser and pricing internals, and
injects loopmath's version explicitly; tests may inject callables and a fixed
version without package metadata or global state.

Vendor resolution now has the non-lossy signature
`resolve_vendor_session(kind: str, correlate: str, /) -> VendorSessionResult`.
The frozen result carries `session: VendorSession | None` and one stable
machine-readable `reason`: `resolved`, `service_unavailable`,
`unsupported_kind`, `invalid_correlate`, `not_found`, `unparseable`, or
`ambiguous`. `session` is present exactly when `reason == "resolved"`; adapters
branch on the reason for honest coverage and consume the session only in that
case. This paragraph supersedes the earlier `VendorSession | None` behavior
without changing its text.

Pricing now separates provisional estimates from settled USD structurally.
`PricingResult` adds `provisional: bool` and `estimate_usd: float | None`: a
price-table `todo` rate produces `priced=False`, `usd=None`,
`provisional=True`, the estimate only in `estimate_usd`, and reason exactly
`price-table entry is provisional`. `PricingResult.to_ocp()` preserves complete
usage but emits no USD or estimate for provisional or unpriced results; only a
settled `priced=True` result may emit `usd`. This paragraph supersedes any
earlier implicit treatment of provisional rates without changing prior text.

## Lane O vendor-session identity convention

Cross-document qualified vendor-session reconciliation is out of scope tonight.
`node.id` remains identity, and bare `attempt.session` is only an opaque correlate,
so independently ingested documents may not reconcile duplicates without qualified
provider/provenance identity.

## Lane O fixture selection convention

The fixture hook is `Adapter.fixture_selection(self, fixture_dir: Path, /) ->
AbstractContextManager[Selection]`. Its default context yields
`Selection(stores=(fixture_dir,))` without changing the directory. The gate keeps
the context entered until `emit(selection)` returns, so an override may materialize
a temporary database and remove it only after emission finishes. Before entering the
hook, the gate configures the no-argument adapter with fixture-scoped host services.
The vendor roots remain `fixture_dir/vendor/claude-code` and
`fixture_dir/vendor/codex`, so bb may select only its materialized temporary database
while its sibling Claude Code and Codex fixture logs remain visible to vendor joins.
| pi | `dev.dagr.adapter.pi` | `(namespace root)` | run, attempt | object | Pi-only metadata, mechanically extracted without prompt, response, tool, or summary text | Core entities have no producer-specific metadata fields | Keep the sanctioned extension namespace |
| pi | `dev.dagr.adapter.pi` | `selection` | run | object | Accounting for discovery, parsing, filters, and limit | v0.2 has no ingest-selection record | Consider a generic ingest accounting record |
| pi | `dev.dagr.adapter.pi` | `selection.files_seen` | run | integer | JSONL files found in the selected stores | v0.2 does not record source-file accounting | Consider a generic ingest accounting record |
| pi | `dev.dagr.adapter.pi` | `selection.sessions_parsed` | run | integer | Files with a usable Pi session header, supported version, and id | v0.2 does not record source parse counts | Consider a generic ingest accounting record |
| pi | `dev.dagr.adapter.pi` | `selection.sessions_selected` | run | integer | Sessions emitted after filters and limit | v0.2 does not record selection counts | Consider a generic ingest accounting record |
| pi | `dev.dagr.adapter.pi` | `selection.retained_unknown_workspace` | run | integer | Sessions retained during a workspace-filtered emission because `cwd` was unavailable | v0.2 cannot state that a requested filter was inapplicable | Consider a generic ingest filter-inapplicability count |
| pi | `dev.dagr.adapter.pi` | `selection.retained_unknown_timestamp` | run | integer | Sessions retained during a time-filtered emission because start time was absent or invalid | v0.2 cannot state that a requested filter was inapplicable | Consider a generic ingest filter-inapplicability count |
| pi | `dev.dagr.adapter.pi` | `selection.omitted` | run | object | Counts for every reason a discovered file or parsed session was not emitted | v0.2 does not record adapter omissions | Consider a generic ingest accounting record |
| pi | `dev.dagr.adapter.pi` | `selection.omitted.unreadable_or_missing_id` | run | integer | Files unreadable as a Pi session or lacking a stable session id | v0.2 cannot distinguish source damage from an empty selection | Consider a generic ingest omission reason |
| pi | `dev.dagr.adapter.pi` | `selection.omitted.unsupported_or_missing_version` | run | integer | Session headers whose version is missing or is not integer version 1, 2, or 3 | v0.2 cannot distinguish unsupported source versions from an empty selection | Consider a generic ingest omission reason |
| pi | `dev.dagr.adapter.pi` | `selection.omitted.duplicate_session_id` | run | integer | Later files suppressed because an earlier deterministically ordered file had the same Pi session id | v0.2 requires unique identities but has no deduplication accounting | Consider a generic ingest omission reason |
| pi | `dev.dagr.adapter.pi` | `selection.omitted.id_filter` | run | integer | Parsed sessions excluded by `--session` | v0.2 does not record source filters | Consider a generic ingest omission reason |
| pi | `dev.dagr.adapter.pi` | `selection.omitted.workspace_filter` | run | integer | Sessions with known `cwd` excluded by `--workspace` | v0.2 does not record source filters | Consider a generic ingest omission reason |
| pi | `dev.dagr.adapter.pi` | `selection.omitted.time_filter` | run | integer | Sessions with a known start outside the inclusive time window | v0.2 does not record source filters | Consider a generic ingest omission reason |
| pi | `dev.dagr.adapter.pi` | `selection.omitted.limit` | run | integer | Otherwise selected sessions omitted by `--limit` | v0.2 does not record truncation | Consider a generic ingest omission reason |
| pi | `dev.dagr.adapter.pi` | `usage_records` | attempt | array | Ordered per-entry usage records whose valid fields are aggregated into core `cost`; values are directly recorded unless a tier member says otherwise | v0.2 cost is attempt-aggregate only | Add an optional per-request usage collection |
| pi | `dev.dagr.adapter.pi` | `usage_records[].id` | attempt | string | Pi entry id, directly recorded | v0.2 has no request identity | Add request identity to per-request usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].kind` | attempt | string | `assistant_message`, `tool_result`, `compaction`, or `branch_summary`, directly recorded from entry type and role | v0.2 has no request subtype | Add request kind to per-request usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].at` | attempt | string | Entry RFC 3339 timestamp when valid, directly recorded | v0.2 has no per-request timestamp | Add request time to per-request usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].provider` | attempt | string | Provider from an assistant message, or configured provider for a summarization entry | v0.2 has only one attempt-level model reference | Add provider to per-request usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].model` | attempt | string | Response model when present, otherwise assistant request model, otherwise configured model for summarization | v0.2 has only one attempt-level model reference | Add model to per-request usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].model_tier` | attempt | string | `verified` for a model on the usage-bearing assistant message; `reported` for configured state applied to a summarization entry | v0.2 model evidence is attempt-level only | Add evidence tier to per-request model |
| pi | `dev.dagr.adapter.pi` | `usage_records[].effort` | attempt | string | Nearest explicit `thinking_level_change` on the entry branch | v0.2 has only one attempt-level effort | Add effort to per-request usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].effort_tier` | attempt | string | Always `reported`: Pi records configured thinking state but not an independent verification of provider execution | v0.2 effort has no evidence tier and is attempt-level only | Add evidence tier to per-request effort |
| pi | `dev.dagr.adapter.pi` | `usage_records[].tokens` | attempt | object | Directly recorded token breakdown for one usage-bearing entry | v0.2 cost is attempt-aggregate only | Add per-request token usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].tokens.input` | attempt | integer | Directly recorded uncached input tokens | v0.2 cost is attempt-aggregate only | Add per-request token usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].tokens.cache_read` | attempt | integer | Directly recorded cache-read tokens | v0.2 cost is attempt-aggregate only | Add per-request token usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].tokens.cache_write` | attempt | integer | Directly recorded total cache-creation tokens | v0.2 cost is attempt-aggregate only | Add per-request token usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].tokens.cache_write_1h` | attempt | integer | Optional directly recorded subset written with 1-hour retention | v0.2 has aggregate cache buckets but no per-request breakdown | Add per-request cache-retention buckets |
| pi | `dev.dagr.adapter.pi` | `usage_records[].tokens.output` | attempt | integer | Directly recorded output tokens, including reasoning where the provider defines it that way | v0.2 cost is attempt-aggregate only | Add per-request token usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].tokens.reasoning` | attempt | integer | Directly recorded reasoning subset of output; its absence makes the usage record incomplete | v0.2 reasoning tokens are attempt-aggregate only | Add per-request reasoning usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].tokens.total` | attempt | integer | Pi's directly recorded total-token value | v0.2 has no total-token field or per-request record | Add per-request reported total |
| pi | `dev.dagr.adapter.pi` | `usage_records[].usd` | attempt | object | Directly recorded per-entry USD breakdown | v0.2 USD is attempt-aggregate only | Add per-request money usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].usd.input` | attempt | number | Directly recorded input cost | v0.2 USD is attempt-aggregate only | Add per-request money usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].usd.cache_read` | attempt | number | Directly recorded cache-read cost | v0.2 USD is attempt-aggregate only | Add per-request money usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].usd.cache_write` | attempt | number | Directly recorded cache-write cost | v0.2 USD is attempt-aggregate only | Add per-request money usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].usd.output` | attempt | number | Directly recorded output cost | v0.2 USD is attempt-aggregate only | Add per-request money usage |
| pi | `dev.dagr.adapter.pi` | `usage_records[].usd.total` | attempt | number | Directly recorded total cost used in the core attempt sum only when complete for every usage record | v0.2 USD is attempt-aggregate only | Add per-request money usage |
| pi | `dev.dagr.adapter.pi` | `incomplete_usage_records` | attempt | integer | Usage records lacking input, cache-read, cache-write, output, or reasoning tokens, or recorded total USD; incomplete fields are omitted rather than treated as zero | v0.2 cannot explain a partial measured aggregate | Consider generic cost-completeness accounting |
| pi | `dev.dagr.adapter.pi` | `malformed_records` | attempt | integer | Non-object or invalid JSON lines skipped after locating a valid session header | v0.2 cannot record source-level parse damage | Consider generic ingest parse accounting |
| OpenCode | `dev.dagr.adapter.opencode` | `(namespace root)` | run, attempt | object | OpenCode-only metadata and ingest accounting mechanically extracted without prompt, response, reasoning, tool, title, or diff text | Core entities have no producer-specific metadata or selection-accounting fields | Keep the sanctioned extension namespace |
| OpenCode | `dev.dagr.adapter.opencode` | `selection_excluded_by_session_id` | run | integer | Count of inventoried sessions excluded by `--session` before any selected session is materialized. | OCP records selected entities but has no source-filter accounting. | Add a standard adapter-selection diagnostics record. |
| OpenCode | `dev.dagr.adapter.opencode` | `selection_excluded_by_workspace` | run | integer | Count of sessions with a known directory excluded by `--workspace` before materialization. | OCP records selected entities but has no source-filter accounting. | Add a standard adapter-selection diagnostics record. |
| OpenCode | `dev.dagr.adapter.opencode` | `selection_excluded_by_time` | run | integer | Count of sessions with a known creation time excluded by the inclusive `--since` or `--until` window before materialization. | OCP records selected entities but has no source-filter accounting. | Add a standard adapter-selection diagnostics record. |
| OpenCode | `dev.dagr.adapter.opencode` | `selection_excluded_by_limit` | run | integer | Count of otherwise selected sessions excluded by `--limit` before materialization. | OCP records selected entities but has no source-filter accounting. | Add a standard adapter-selection diagnostics record. |
| OpenCode | `dev.dagr.adapter.opencode` | `selection_unknown_created_at` | run | integer | Count of candidate sessions retained because a `--since` or `--until` filter could not be evaluated without `session.time.created`. | OCP records selected entities but has no source-filter accounting. | Add a standard adapter-selection diagnostics record. |
| OpenCode | `dev.dagr.adapter.opencode` | `selection_unknown_workspace` | run | integer | Count of candidate sessions retained because a `--workspace` filter could not be evaluated without `session.directory`. | OCP records selected entities but has no source-filter accounting. | Add a standard adapter-selection diagnostics record. |
| OpenCode | `dev.dagr.adapter.opencode` | `parent_relations_omitted` | run | integer | Count of explicit `session.parentID` relations not emitted because the known parent was excluded by selection. The source relation remains verified; omission is mechanical. | An OCP edge cannot name an endpoint absent from the document. | Add a standard omitted-relation count grouped by relation kind and reason. |
| OpenCode | `dev.dagr.adapter.opencode` | `parent_relations_unresolved` | run | integer | Count of explicit `session.parentID` values not emitted because the value was malformed, named the child itself, or named no session in any selected store. | An OCP edge cannot name an unknown endpoint. | Add a standard unresolved-relation count grouped by relation kind and reason. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage` | attempt | array of object | Per-model assistant-message usage, emitted when a session used multiple models, multiple variants, or lacks a model on some requests. All values are copied or summed mechanically from message records. | OCP v0.2 has one model and one aggregate cost record per attempt. Assigning one model to a mixed session would misattribute usage. | Add repeated model-usage records beneath attempt cost. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage[].provider` | attempt | string | OpenCode `assistant.providerID` for this usage group, verified from message records. | The core attempt model is singular. | Field of the proposed repeated model-usage record. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage[].model` | attempt | string | OpenCode `assistant.modelID` for this usage group, verified from message records. | The core attempt model is singular. | Field of the proposed repeated model-usage record. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage[].variants` | attempt | array of string | Distinct sorted OpenCode `assistant.variant` or legacy `assistant.mode` labels in the usage group, verified from message records. | Core effort is singular and cannot represent changes within an attempt. | Field of the proposed repeated model-usage record. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage[].requests` | attempt | integer | Number of assistant message requests in the provider/model group, counted mechanically. | Core requests aggregate the whole attempt. | Field of the proposed repeated model-usage record. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage[].input_tokens` | attempt | integer | Sum of complete nonnegative input-token fields for the provider/model group. | Core input tokens aggregate the whole attempt. | Field of the proposed repeated model-usage record. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage[].cached_input_tokens` | attempt | integer | Sum of complete nonnegative cache-read token fields for the provider/model group. | Core cached input tokens aggregate the whole attempt. | Field of the proposed repeated model-usage record. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage[].cache_creation_tokens` | attempt | integer | Sum of complete nonnegative cache-write token fields for the provider/model group. OpenCode does not expose cache-retention buckets. | Core cache creation tokens aggregate the whole attempt. | Field of the proposed repeated model-usage record. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage[].output_tokens` | attempt | integer | Sum of complete nonnegative output-token fields for the provider/model group. | Core output tokens aggregate the whole attempt. | Field of the proposed repeated model-usage record. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage[].reasoning_tokens` | attempt | integer | Sum of complete nonnegative reasoning-token fields for the provider/model group. | Core reasoning tokens aggregate the whole attempt. | Field of the proposed repeated model-usage record. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_usage[].usd` | attempt | number | Sum of complete nonnegative per-message costs for the provider/model group. The core attempt USD remains the source session's cost. | Core USD aggregates the whole attempt and cannot be partitioned by model. | Field of the proposed repeated model-usage record. |
| OpenCode | `dev.dagr.adapter.opencode` | `model_requests_unattributed` | attempt | integer | Count of assistant message requests with no complete provider/model pair. | OCP has no accounting field for requests deliberately excluded from model attribution. | Add a standard unknown-model request count. |
| OpenCode | `dev.dagr.adapter.opencode` | `usage_incomplete_messages` | attempt | object | Counts assistant messages missing or invalid in one or more token streams. Affected aggregate core streams are omitted, not undercounted. | OCP cost fields are optional but cannot explain why a stream was omitted. | Add standard cost-completeness diagnostics. |
| OpenCode | `dev.dagr.adapter.opencode` | `usage_incomplete_messages.input_tokens` | attempt | integer | Count of assistant messages without a valid nonnegative input-token value. | OCP cost fields cannot carry per-stream completeness counts. | Child field of the proposed cost-completeness diagnostics record. |
| OpenCode | `dev.dagr.adapter.opencode` | `usage_incomplete_messages.cached_input_tokens` | attempt | integer | Count of assistant messages without a valid nonnegative cache-read token value. | OCP cost fields cannot carry per-stream completeness counts. | Child field of the proposed cost-completeness diagnostics record. |
| OpenCode | `dev.dagr.adapter.opencode` | `usage_incomplete_messages.cache_creation_tokens` | attempt | integer | Count of assistant messages without a valid nonnegative cache-write token value. | OCP cost fields cannot carry per-stream completeness counts. | Child field of the proposed cost-completeness diagnostics record. |
| OpenCode | `dev.dagr.adapter.opencode` | `usage_incomplete_messages.output_tokens` | attempt | integer | Count of assistant messages without a valid nonnegative output-token value. | OCP cost fields cannot carry per-stream completeness counts. | Child field of the proposed cost-completeness diagnostics record. |
| OpenCode | `dev.dagr.adapter.opencode` | `usage_incomplete_messages.reasoning_tokens` | attempt | integer | Count of assistant messages without a valid nonnegative reasoning-token value. | OCP cost fields cannot carry per-stream completeness counts. | Child field of the proposed cost-completeness diagnostics record. |
| none (proposal only) | `dev.dagr.artifact` | `kind` | artifact | string | PROPOSAL, NOT IMPLEMENTED. Open vocabulary naming what the artifact is, so that any agent output is representable and not only a file: `file`, `decision` (a workflow gate verdict is an artifact), `review`, `metric` (for example the number of critical issues a review raised, or the views a published post received), and values not yet enumerated. A consumer must round-trip a value it does not recognize rather than dropping or coercing it. | OCP v0.2 artifacts are implicitly files identified by a path, so a gate verdict, a review, or a measured quantity has no representable artifact form at all. | Adopt as a core artifact field with an open vocabulary and an explicit unknown-value round-trip rule. |
| loopmath graph | `dev.dagr.artifact` | `bytes` | artifact | integer | IMPLEMENTED AS AN OCP v0.2 EXTENSION. Size in bytes of this artifact version when the write was observed. Null when the store does not record it; zero is a real size and must never stand in for unknown. | OCP v0.2 records that an artifact was written but never how large it is, so effort and cost per artifact cannot be normalized. | Adopt as a core artifact-version field, nullable, with unknown distinct from zero. |
| loopmath graph | `dev.dagr.artifact` | `lines_added` | artifact | integer | IMPLEMENTED AS AN OCP v0.2 EXTENSION. Lines added by the write that produced this artifact version. Tier follows the source: verified when the store records the diff, heuristic when the adapter computes it against the earlier version. | OCP v0.2 has no change-magnitude field, so a one-character edit and a full rewrite are indistinguishable. | Adopt as a core artifact-version field paired with `lines_removed`, with a required evidence tier because the value is often adapter-computed. |
| loopmath graph | `dev.dagr.artifact` | `lines_removed` | artifact | integer | IMPLEMENTED AS AN OCP v0.2 EXTENSION. Lines removed by the write that produced this artifact version, tiered on the same rule as `lines_added`. | OCP v0.2 has no change-magnitude field, so deletion work is invisible. | Adopt as a core artifact-version field paired with `lines_added`, with a required evidence tier. |
| loopmath graph | `dev.dagr.artifact` | `language` | artifact | string | IMPLEMENTED AS AN OCP v0.2 EXTENSION. Lowercase final filename-extension identifier without the dot, emitted at heuristic tier. For example, `py` records the path suffix only; it is not a semantic claim that the artifact contains Python. | OCP v0.2 carries no artifact typing, so extension-based aggregation otherwise requires re-reading local paths that a consumer may not have. | Adopt as a core artifact field with a required evidence tier and a stable extension-identifier vocabulary rather than language names or free text. |
| loopmath graph | `dev.dagr.artifact` | `tests_touched` | artifact | integer | IMPLEMENTED AS AN OCP v0.2 EXTENSION. How many test files or test cases the write touched. Recorded open question: a count answers whether a change carried tests, but an array of test identifiers would also answer which ones. | OCP v0.2 cannot distinguish a change that carried tests from one that did not. | Adopt as a core artifact-version field; settle count versus identifier array before v0.3 freezes. |
| loopmath graph | `dev.dagr.artifact` | `fate` | artifact | string | IMPLEMENTED AS AN OCP v0.2 EXTENSION. What became of this artifact version after it was written: `kept`, `edited`, `reverted`, `deleted`, or `unknown`. `unknown` is a real value and the default, and must never be inferred silently from the mere absence of a later write. | OCP v0.2 records writes but no outcome, so work that was reverted is indistinguishable from work that shipped. | Adopt as a core artifact-version field with exactly these five values and `unknown` required as the default. |
| none (proposal only) | `dev.dagr.artifact` | `version` | artifact | integer or string | PROPOSAL, NOT IMPLEMENTED. Identifier of this version within its artifact lineage, ordered monotonically. Each observed write produces a new version rather than mutating the previous one. | OCP v0.2 artifacts have no versions, so repeated writes to one path collapse into a single entity and per-write measurements have nowhere to attach. | Adopt artifact versions as first-class entities, created by write edges. |
| none (proposal only) | `dev.dagr.artifact` | `supersedes` | artifact | string | PROPOSAL, NOT IMPLEMENTED. Identifier of the artifact version this one replaces, forming the lineage chain. Verified when the store records the ordering; heuristic when the adapter orders by timestamp alone. | OCP v0.2 has no link between successive states of the same artifact. | Adopt as a core artifact-version link with a required evidence tier. |
| none (proposal only) | `dev.dagr.artifact` | `edge.creates_version` | edge | string | PROPOSAL, NOT IMPLEMENTED. On an artifact write edge, the identifier of the artifact version that write created. This states the lineage rule explicitly: a write edge creates a new artifact version instead of mutating an existing artifact. | OCP v0.2 write edges point at one mutable artifact, so the graph cannot say which state of the artifact a given attempt actually produced. | Adopt together with artifact versions; a write edge must name the version it created. |
| loopmath graph | `dev.dagr.artifact` | `meta` | artifact | object | IMPLEMENTED AS AN OCP v0.2 EXTENSION. Free-form object of arbitrary JSON keys carrying per-artifact detail that no standard field covers. Consumers must ignore keys they do not recognize, and nothing in the core graph may depend on it. | OCP v0.2 `ext` is namespaced per adapter and per entity; an artifact has no sanctioned free-form slot for detail that is genuinely local and unmodellable. | Adopt as a core artifact field, explicitly free-form and explicitly ignorable, governed by the export privacy rule in the next row. |
| none (proposal only) | `dev.dagr.artifact` | `meta export privacy level` | document | string | PROPOSAL, NOT IMPLEMENTED. `meta` may hold local-only detail such as absolute paths, hostnames, branch names, or prompt fragments. At export the document is stripped or anonymized to a level the user chooses, and the document records which level was applied so a consumer can tell a redacted document from a complete one. | OCP v0.2 has a document privacy profile but no per-field redaction level and no record of what was removed, so a stripped document is indistinguishable from one that never held the detail. | Adopt an export redaction level recorded in the document alongside the free-form `meta` field. |

For the implemented `dev.dagr.artifact` measurements, evidence is stored in a sibling
`<member>_tier` key with value `verified`, `reported`, or `heuristic`. A null
measurement has a null tier and a present measurement has a non-null tier;
`fate: unknown` similarly pairs with `fate_tier: null`, while a known fate has a non-null
tier. The extension always emits these keys, including null and zero values.

## OpenCode source choice

OpenCode 1.17.11 stores sessions, messages and parts in SQLite, while its
supported exporter returns one object with `info` and `messages` and preserves
the optional `parentID`. A local structural comparison on 2026-09-02 found the
sampled export's message and part counts identical to its SQLite rows. The
database's flattened five token totals and session cost also equaled sums of
assistant-message usage and cost for every local session row. The export
contained the directory, timestamps, model fields, session cost and all five
token streams needed by this adapter; its message parts remain available but
their text is deliberately ignored.

The supported export is therefore the preferred ingest boundary. For the
discovered default store, SQLite is used read-only only to inventory ids and
selection metadata; each selected session is fetched with sanitized
`opencode export`, and the sanitizer-redacted directory locator is restored
from that inventory row. Explicit export JSON is portable and is the fixture
format. Direct SQLite reading is retained only for an explicitly selected
non-default database, because the OpenCode CLI cannot target another database.
The live store inspected for this choice had no non-null parent rows, so the
verified parent case is exercised only by synthetic shape-equivalent fixtures.

The price table covers every model id observed during that structural check:
`gpt-5.3-codex-spark`, `gpt-5.4-mini`, `gpt-5.5`, `big-pickle`,
`deepseek-v4-flash-free`, and `north-mini-code-free`. OpenCode offers
`big-pickle`, `deepseek-v4-flash-free`, and `north-mini-code-free` free as of
the measured local catalog date, 2026-09-02. The price validator emits a
zero-rate warning for their intentional input/output zeros by design, and a
regression test pins that warning.

Known gap: future lanes adding a free model will meet the same zero-rate warning.

## Lane A3 post-review pricing dispositions

| Status | Trigger | False emitted field | Direction and consequence |
|---|---|---|---|
| RESOLVED by refusal | Uncovered requests in an attempt lack one common valid model. | `attempt.ext.dev.dagr.adapter.otel-genai.pricing_refusal_reason` | The adapter omits core USD instead of pricing aggregate usage under one model; mixed-model partitioned pricing remains intentionally unimplemented. |
| RESOLVED by component accounting | Some mapped requests carry valid span cost and others do not. | `attempt.ext.dev.dagr.adapter.otel-genai.reported_cost_usd` | The extension preserves the measured component, pricing receives only uncovered requests, and core USD is emitted only when both components are available. |

The Orca adapter emits no `ext` member, so it requires no extension proposal.


For the bb adapter, `coverage.unpriced_vendor_sessions` also includes joined
sessions whose matching price-table row has `todo = true`. Their measured token
streams remain in OCP, but the provisional USD estimate is omitted.

## bb host-service migration corrections

For bb, this paragraph supersedes the broader unjoined-reason claim in the
earlier `coverage.unjoined_vendor_sessions` proposal row. The adapter itself
preclassifies only a missing, malformed, or conflicting bb event correlate, an
unsupported bb provider, or duplicate ownership of one correlate by bb
threads. For an otherwise eligible correlate, host-visible failure is taken
only from `VendorSessionResult.reason`: `service_unavailable`,
`unsupported_kind`, `invalid_correlate`, `not_found`, `unparseable`, or
`ambiguous`. All of these cases increment the existing aggregate; bb does not
infer a host parser or log outcome.

A bb session correlate establishes correspondence, never causality by itself.
The bb adapter keeps a distinct controller node and independently parsed vendor
node, places the canonical opaque correlate in `attempt.session` on the vendor
attempt only, and under the bb-specific operator ruling keeps the `launch` edge
plus matching `origin.launched_by` at heuristic tier. Their vendor session
correspondence is correlation, not causal launch evidence. The vendor run id
remains the vendor node id; the reader identity and session-based deduplication
rules are unchanged.

For bb pricing, any `PricingResult` with `priced = false`, including a result
with `provisional = true` and an `estimate_usd`, increments
`coverage.unpriced_vendor_sessions`. `PricingResult.to_ocp()` retains complete
measured usage but emits neither `usd` nor `estimate_usd` for those results.

Known gap: cross-document qualified-session reconciliation is out of scope for
this integration. Node ids are not globally deduplicated from bare
`attempt.session` values; future reconciliation must qualify the session by
its vendor namespace and provenance.
