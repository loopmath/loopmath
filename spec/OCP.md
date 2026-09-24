# Orchestration Context Protocol (OCP)

Current version: 0.3. Earlier versions: 0.2 and 0.1.

## 1. What OCP is

OCP is a file format for orchestration runs. One JSON document describes one
run of agents on a task: the units of work (nodes) and how they depend on each
other (edges), every try at a unit (attempts) with its model, reasoning effort,
token cost and outcome, the files attempts wrote and read (artifacts), events,
the producer that wrote the document, and a declared privacy profile. From 0.3
a document can also record the task, the configuration that ran it (a workflow
shape plus a setting for each piece), how that configuration was chosen, the
signals and the rule that decide acceptance, and comparisons between runs on
the same task.

Any orchestrator can write OCP and any tool can read it. loopmath is the
reference reader and ships the reference checker.

## 2. What is normative

Two things define OCP:

1. The JSON Schema (draft 2020-12) for the document's version, including the
   meaning its description strings give each field: `ocp-v0.3.schema.json`,
   `ocp-v0.2.schema.json`, and `ocp-v0.schema.json` for 0.1, all in this
   folder.
2. The rules in section 6, which JSON Schema cannot express.

A document conforms when it validates against the schema its `ocp` value
selects and breaks no error rule of section 6 for its version. Warnings do not
affect conformance.

The reference checker validates against the schemas and implements every rule
in section 6. Run it as `loopmath ocp validate FILE...`, `python3 -m
loopmath.ocp.conformance FILE...` or `python3 spec/ocp_conformance.py
FILE...`. It exits 0 when no file has an error. A requirement stated only in a
description string is normative too, but the checker does not test every one,
so a clean result does not prove that a document meets every description.

## 3. Versions and migration

`ocp` is the version string: `0.1`, `0.2` or `0.3`. It selects the schema and
decides which rules apply: a rule introduced in a version applies to documents
at that version or later. In an older document, a field that only a later
version defines is an unknown field and is ignored.

- A conforming reader accepts 0.1, 0.2 and 0.3 documents.
- 0.3 is a strict superset of 0.2: raising `ocp` from 0.2 to 0.3 keeps a valid
  document valid, as long as the document does not use a name that 0.3
  defines as an unknown field of its own (for example a string in `run.task`,
  which 0.3 defines as an object).
- 0.2 is not a superset of 0.1. 0.2 requires `tier` on every edge, so a 0.1
  document with edges fails once only its `ocp` value is raised. Convert 0.1
  documents with `loopmath ocp migrate`.
- A field or namespace is removed only after a deprecation that stays in the
  specification for at least two minor versions and names the replacement.
  See `VERSIONING.md`.

`loopmath ocp migrate FILE... [--out DIR]` is the reference migration. It
writes 0.3, never changes its input, and returns its input unchanged when run
on its own output. Each addition is made only when the field is absent:

- `ocp` becomes `0.3`, and `run.ext["dev.loopmath.migration"]` records
  `{"from": <old version>}`;
- a 0.1 edge without a tier gets `tier: reported`;
- `run.task` is built from the run labels `task`, `type` and `repo` when any
  is present (its id is the `task` label, else the run id), with `labeled_by`
  `{how: inferred, tier: heuristic}`;
- `node.vertex` is set when the node's attempts agree on one `role.value`;
- `run.provenance` becomes `{kind: logged, chooser: habit}`;
- `run.configuration` becomes `{source: habit}`, plus the workflow and settings
  loopmath infers from the run when it can, with the inference confidence in
  `ext["dev.loopmath.inferred"]` (`tier: heuristic`).

Existing fields and extension keys are kept as they are.

## 4. Reading a document

- Ignore fields you do not know, at every level: the top level, `run`, nodes,
  edges, attempts, artifacts, events and `ext` objects.
- Producer-specific data goes in the `ext` object that every entity accepts.
  Its keys use reverse-domain namespaces. Ignore namespaces you do not
  understand.
- A string field is an open set with a recommended vocabulary unless the
  schema gives it an `enum`. Accept any value of an open field.
- A missing `outcome.evidence` means `asserted`. A tier is never upgraded
  without new mechanical evidence.
- A node's `state` is the producer's summary of its attempts. When learning
  from settled runs, trust attempt outcomes over node state.
- `model.raw` keeps the producer's model label as written. Canonicalizing
  model names is the reader's job.

## 5. Privacy

Every document declares `privacy.profile`: `metadata_only` or `full`.

- `metadata_only`: no prompt, completion or transcript text anywhere in the
  document. Free-text fields hold short producer-authored labels, and the
  schema caps the designated label fields at 500 characters. Other strings,
  such as `task.features` values, setting `options` and `ext` payloads, are
  not capped. The no-text rule is the producer's promise; no checker can
  verify it. Paths and workspace names can still be sensitive.
- `full`: text may appear in `ext` namespaces, never in core fields.

## 6. Rules the schema cannot express

Codes starting with E are errors; codes starting with W are warnings. Each
rule is listed under the version that introduced it and applies to that
version and every later one.

### 6.1 Every version (0.1 and later)

| Code | Rule |
|---|---|
| E001 | The file can be read and parsed as JSON. |
| E002 | The top-level JSON value is an object. |
| E003 | `ocp` is present and is `0.1`, `0.2` or `0.3`. |
| E010 | The document validates against the schema its `ocp` value selects. |
| E104 | A top-level list (`nodes`, `edges`, `attempts`, `events`, `groups`, and from 0.2 `artifacts`) is a list when present, not null or another type. |
| E100 | Node ids are unique. |
| E101 | Attempt ids are unique. |
| E102 | Group ids are unique. |
| E110 | `edge.from` names an existing node. |
| E111 | `edge.to` names an existing node. |
| E112 | `attempt.node` names an existing node. |
| E113 | An event's `node`, when given, names an existing node. |
| E114 | An event's `attempt`, when given, names an existing attempt. |
| E115 | `node.group`, when given, names an existing group. |
| E120 | `group.parent`, when given, names an existing group. |
| E121 | Group parent links have no cycle. |
| E122 | `dep` and `fan_in` edges together form a DAG. An edge without `kind` is a `dep` edge. `spawn`, `launch` and `artifact` edges record lineage, not scheduling, and may form cycles. |
| E130 | An attempt with a terminal status (`done`, `failed`, `rejected`, `canceled`, `settled_unverified` or `lost`) that carries an outcome has `outcome.result` equal to its status. |
| E181 | No `ext` key begins with a retired prefix: `ocp.` (use `io.orchestrationcontextprotocol.`) or `dagr.` (use `dev.loopmath.` from 0.3, `dev.dagr.` before). The keys of every `ext` object in the document are checked; the values stored under those keys are not. |
| W131 | An attempt with a terminal status has no outcome. |
| W132 | An attempt whose status is not terminal carries an outcome. |
| W140 | `cause.ref` names no attempt in the document. |
| W141 | `outcome.via` names no node in the document. |
| W150 | Events are not in ascending time order. |

### 6.2 From 0.2

| Code | Rule |
|---|---|
| E103 | Artifact ids are unique. |
| E160 | Every edge carries `tier`, one of `verified`, `heuristic` or `reported`, whatever its kind and whoever the producer is. |
| E116 | An `artifact` edge names an artifact in `artifact`. |
| E117 | That artifact exists. |
| E125 | An `artifact` edge carries both `from_attempt` and `to_attempt`. |
| E123 | An edge's `from_attempt` and `to_attempt`, when given, name existing attempts. |
| E124 | The `from_attempt` is an attempt at the edge's `from` node, and the `to_attempt` an attempt at its `to` node. |
| E164 | On an `artifact` edge, `from_attempt` is the artifact's `producer` and `to_attempt` is one of its `consumers`. |
| E118 | `artifact.producer` and every entry of `writers` and `consumers` name existing attempts. |
| E162 | `artifact.producer` equals `writers[0]`. |
| E166 | `writers` is not empty, and neither `writers` nor `consumers` lists an attempt twice. |
| E167 | `writers` and `consumers`, when present, are lists, not null. |
| E163 | A known `n_writes` is at least the number of writers, and a known `n_reads` at least the number of consumers. A null count is unknown and is not checked. |
| E119 | `origin.launched_by`, when not null, names another existing attempt. |
| E161 | An attempt whose `origin.launched_by` names a launcher has a `launch` edge from the launcher's node to its own node. |
| E165 | `origin.external: true` does not appear with a non-null `origin.launched_by`. |
| E170 | When `cache_creation_5m_tokens`, `cache_creation_1h_tokens` and `cache_creation_tokens` are all present, the first two sum to the third. |
| E171 | When `producer.name` names loopmath's log extractor (`loopmath`, or `dagr` in documents written before its rename, alone or followed by a non-word character, as in `loopmath/graph`), every cost record has `basis: measured`. One exception, from 0.3 and only when `producer.name` is loopmath's: `basis: allocated` is allowed when `cost.ext["dev.loopmath.logmatch"]` is an object whose `tier` is `heuristic` or whose `shared_session` is `true` or a non-empty object. |
| E180 | A capability that `producer.capabilities` declares `false` has no matching element in the document. The capabilities are `groups`, `events`, `artifacts`, `edges_dep` (any `dep` or `fan_in` edge), `edges_spawn`, `edges_launch`, `edges_artifact`, `cost_usd`, `cost_tokens` and `outcome_evidence`, and from 0.3 `task`, `configuration`, `signals`, `slate` and `receipt`. |
| W180 | `producer.capabilities` is absent, so readers cannot tell an unsupported element from an empty one. |
| W200 | A value lies outside the recommended vocabulary of an open field: `node.kind`, `node.state`, `edge.kind`, `artifact.kind.value`, `role.value`, `phase.value`, `cost.basis`, `cause.type`, `outcome.evidence`, `event.type`, and from 0.3 `task.type`. Readers must accept it. |

### 6.3 From 0.3

| Code | Rule |
|---|---|
| E190 | `run.configuration.id` equals the canonical id (section 7) of its workflow and settings. A workflow given by reference is resolved through the reader's catalog; a reader that cannot resolve it skips E190 and E191. |
| E191 | Every top-level piece of the workflow has an object in `configuration.settings` under its id, except a piece that holds a nested workflow and has no role. |
| E192 | In an inline workflow, piece and artifact ids are unique across both lists; every edge joins an existing piece and an existing artifact, in either direction (a piece produces an artifact, or an artifact feeds a piece); and the edges have no cycle, since a repair loop belongs in `control.repair`. Nested inline workflows are checked the same way. |
| E193 | In an inline workflow, every `control.gates` entry names a piece, every `control.repair` key is one of `control.gates`, and every `control.repair` value names a piece. Nested inline workflows are checked the same way. |
| E194 | The run's id is among `slate.members`. `task.base_commit` equals `slate.base_commit` when both are present. A preference that names a slate names this run's slate. A preference's `winner` is `tie` or one of the preference's `members`, or of the slate's members when the preference lists none. Across documents checked together, documents with the same `slate.id` share `task.id` and the base commit (`task.base_commit`, else `slate.base_commit`). |
| E195 | Signal values fit their kind: a verdict is `accept`, `reject`, `pass`, `fail` or `error`; a score is a number, or null when declared and not yet measured; a score with `scale: fraction` lies in [0, 1]; an event's value is a reference string. `acceptance_rule.score.name` names no verdict or event signal; when `acceptance_rule.score.scale` is `fraction`, its `target` lies in [0, 1]; and `acceptance_rule.requires` names no score or event signal. |
| W196 | `run.receipt` is present and `producer.name` is not loopmath's; only loopmath writes receipts. |
| W182 | An `ext` key begins with `dev.dagr.`, loopmath's namespace before its rename. It is still read through 0.4; write `dev.loopmath.` instead. |

The reference checker also reports W001 when the `jsonschema` package is
missing (schema validation is skipped) and E011 when it cannot load a schema
file. These describe the checker's environment, not the document.

## 7. Configuration id

`run.configuration.id` is `cfg_` followed by the first 12 hexadecimal digits
of the SHA-256 of the canonical JSON of `{"workflow": W, "settings": S}`,
encoded as UTF-8. The canonical JSON has sorted keys at every level, no
whitespace (separators `,` and `:`), and non-ASCII characters written as
themselves. The reference implementation is `loopmath.ocp.canonical.config_id`.

A workflow given by reference, `{"ref": ..., "version": ...}` without
`pieces`, is replaced by the workflow it names before hashing. W then has
exactly four members; the workflow's `id`, `version`, `title`, `ext` and
unknown members are left out:

- `pieces`: for each piece object, `{id, width}` with `width` 1 when absent,
  plus `role` when present and `workflow` when present (a nested workflow is
  made canonical the same way). Sorted by id. `[]` when absent.
- `artifacts`: for each artifact object, `{id}` plus `kind` when present.
  Sorted by id. `[]` when absent.
- `edges`: the edge pairs, sorted by the string forms of their two ends. `[]`
  when absent.
- `control`: `gates` sorted (`[]` when absent); `repair` (`{}` when absent);
  `budget`, which counts rounds including the first, read as max(1, budget)
  when it is an integer and as 1 otherwise (a JSON number with a fraction
  part, such as `2.0`, reads as 1); `rescue` reduced to its `kind` and `ref`
  when present; and `gate_rules`, the value of
  `control.ext["dev.loopmath.gate_rules"]` when that key is present and not
  empty, because a non-default gate rule changes what runs. Other `ext` keys
  are left out. An absent `control` reads as
  `{"budget": 1, "gates": [], "repair": {}}`.

S maps every key of `configuration.settings` to `{harness, model, effort,
context_policy, options}`: `harness` as given, else null; `model` is the model
reference's `id`, else its `raw`, else a bare string as given, else null;
`effort` is `default` when absent; `context_policy` is `fresh` when absent;
`options` is `{}` when absent. The setting's `ext` is left out.

The id covers the declared configuration only. An attempt may carry its own
`setting` where it differed from the configuration's setting for its piece,
so two runs with the same id can still differ in what individual attempts
used.

Numbers are written as the reference implementation's JSON encoder writes
them. A producer that computes ids in another language should check its
output against the reference implementation, for example on the documents in
`examples/v0.3/`, whose ids all verify.

## 8. Namespaces

OCP's own extension namespace is `io.orchestrationcontextprotocol.`. loopmath
writes its extensions under `dev.loopmath.` from 0.3. Documents written
before loopmath's rename carry `dev.dagr.` keys: readers accept them through
0.4, with W182 in 0.3 documents. The bare `dagr.` and `ocp.` prefixes are
retired (E181).

## 9. Examples and golden files

- `examples/v0.3/`: one run for each workflow shape in loopmath's catalog
  (`solo`, `best_of_n`, `plan_implement`, `implement_review`,
  `plan_implement_review`, `swarm`) and `full-fields.ocp.json`, which uses the
  fields 0.3 adds. Each validates with no findings.
- `examples/golden/v0.3/`: a pass file and a fail file for each 0.3 rule
  (`<code>-pass.ocp.json`, `<code>-fail.ocp.json`); `expected.json` lists the
  codes each file must produce.
- `examples/golden/`: 0.2 documents: a minimal run, artifacts, groups and
  events, and `broken-tier.ocp.json`, which fails E010 and E160 on an edge
  tier outside the enum.
- `examples/minimal.ocp.json` (0.1) and `examples/swarm-v02.ocp.json` (0.2).
- `examples/review-loop-v01.ocp.json`: a synthetic 0.1 run still in progress,
  with a failed attempt, a rejected follow-up and a send-back.
  `examples/review-loop-v03.ocp.json` is what `loopmath ocp migrate` makes of
  it: no errors and one warning, W180, because a 0.1 producer declares no
  capabilities.

## 10. Full text

The OCP website will host the full text of the specification, with
explanations, a field reference and guides for producers and readers. Until
it is published, the schemas and this note are the whole specification.
