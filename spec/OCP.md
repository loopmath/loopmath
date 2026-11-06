# Orchestration Context Protocol (OCP)

Current version: 0.4. Earlier versions: 0.3, 0.2 and 0.1.

## 1. What OCP is

OCP is a file format for orchestration runs. One JSON document describes one
run of agents on a task: the units of work (nodes) and how they depend on each
other (edges), every try at a unit (attempts) with its model, reasoning effort,
token cost and outcome, the files attempts wrote and read (artifacts), events,
the producer that wrote the document, and a declared privacy profile. From 0.3
a document can also record the task, the configuration that ran it (a workflow
shape plus a setting for each piece), how that configuration was chosen, the
signals and the rule that decide acceptance, and comparisons between runs on
the same task. From 0.4 a second kind of document, a flow document,
describes an experiment on a fixed workflow (section 12).

Any orchestrator can write OCP and any tool can read it. loopmath is the
reference reader and ships the reference checker.

## 2. What is normative

Two things define OCP:

1. The JSON Schema (draft 2020-12) for the document's version, including the
   meaning its description strings give each field: `ocp-v0.4.schema.json`,
   `ocp-v0.3.schema.json`, `ocp-v0.2.schema.json`, and `ocp-v0.schema.json`
   for 0.1, all in this folder.
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

`ocp` is the version string: `0.1`, `0.2`, `0.3` or `0.4`. It selects the
schema and decides which rules apply: a rule introduced in a version applies to
documents at that version or later. In an older document, a field that only a
later version defines is an unknown field and is ignored.

- A conforming reader accepts 0.1, 0.2, 0.3 and 0.4 documents.
- 0.4 is a strict superset of 0.3: a valid 0.3 document stays valid when its
  `ocp` value is raised to 0.4, and its configuration id (section 7) does not
  change, as long as the document does not use a name that 0.4 defines (a
  top-level `kind`, a top-level `flow` or `control.checks`) as an unknown
  field of its own.
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

`loopmath ocp migrate FILE... [--out DIR] [--to 0.4]` is the reference
migration. It writes 0.3 by default, or 0.4 with `--to 0.4`, never changes its
input, and returns its input unchanged when run on its own output. With
`--to 0.4` it also reads 0.3 and 0.4 documents; it never writes a 0.4 document
down to 0.3. Each addition is made only when the field is absent:

- `ocp` becomes the target version, and `run.ext["dev.loopmath.migration"]`
  records `{"from": <old version>}`;
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
  model names is the reader's job. A producer never builds a label with the
  effort fused in (such as `fable·xhigh`) itself; when the source wrote one,
  `raw` keeps it, and effort known separately goes in `effort`.
- `cost.output_tokens` includes reasoning tokens; `reasoning_tokens` is the
  reasoning share of it, a breakdown. A producer whose source reports
  reasoning apart from output adds it into `output_tokens`. A reader never
  adds the two when totalling.

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
| E001 | The file can be read and parsed as JSON. The reference reader also refuses, by path, an integer with more decimal digits than its active budget (Python's `sys.get_int_max_str_digits()`, 4300 by default), in a file or in a document built in code; JSON itself sets no such bound. |
| E002 | The top-level JSON value is an object. |
| E003 | `ocp` is present and is `0.1`, `0.2`, `0.3` or `0.4`. |
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
| E171 | When `producer.name` names loopmath's log extractor (`loopmath`, or `dagr` in documents written before its rename, alone or followed by a non-word character, as in `loopmath/graph`), every cost record has `basis: measured`. One exception, from 0.3 and only when `producer.name` is loopmath's: `basis: allocated` is allowed when `cost.ext["dev.loopmath.logmatch"]` is an object whose `tier` is `heuristic` or whose `shared_session` is `true` or a non-empty object. A second, from 0.4 and only when `producer.name` is loopmath's: `basis: simulated` is allowed when `run.ext["dev.loopmath.simulate"]` is an object, the mark of a run a simulator drew (section 12.5). |
| E180 | A capability that `producer.capabilities` declares `false` has no matching element in the document. The capabilities are `groups`, `events`, `artifacts`, `edges_dep` (any `dep` or `fan_in` edge), `edges_spawn`, `edges_launch`, `edges_artifact`, `cost_usd`, `cost_tokens` and `outcome_evidence`, from 0.3 `task`, `configuration`, `signals`, `slate` and `receipt`, and from 0.4 `flow` (a flow document or `run.flow`). In a flow document, the `configuration` element is `flow.configuration`. |
| W180 | `producer.capabilities` is absent, so readers cannot tell an unsupported element from an empty one. |
| W200 | A value lies outside the recommended vocabulary of an open field: `node.kind`, `node.state`, `edge.kind`, `artifact.kind.value`, `role.value`, `phase.value`, `cost.basis`, `cause.type`, `outcome.evidence`, `event.type`, and from 0.3 `task.type`. From 0.4, `simulated` joins the recommended values of `cost.basis`. Readers must accept it. |

### 6.3 From 0.3

| Code | Rule |
|---|---|
| E190 | `run.configuration.id` equals the canonical id (section 7) of its workflow and settings. A workflow given by reference is resolved through the reader's catalog; a reader that cannot resolve it skips E190 and E191. |
| E191 | Every top-level piece of the workflow has an object in `configuration.settings` under its id, except a piece that holds a nested workflow and has no role. |
| E192 | In an inline workflow, piece and artifact ids are unique across both lists; every edge joins an existing piece and an existing artifact, in either direction (a piece produces an artifact, or an artifact feeds a piece); and the edges have no cycle, since a repair loop belongs in `control.repair`. Nested inline workflows are checked the same way. |
| E193 | In an inline workflow, every `control.gates` entry names a piece, every `control.repair` key is one of `control.gates`, and every `control.repair` value names a piece. Nested inline workflows are checked the same way. |
| E194 | The run's id is among `slate.members`. `task.base_commit` equals `slate.base_commit` when both are present. A preference that names a slate names this run's slate. A preference's `winner` is `tie` or one of the preference's `members`, or of the slate's members when the preference lists none. Across documents checked together, documents with the same `slate.id` share `task.id` and the base commit (`task.base_commit`, else `slate.base_commit`). |
| E195 | Signal values fit their kind: a verdict is `accept`, `reject`, `pass`, `fail` or `error`; a score is a number a float can hold, or null when declared and not yet measured; a score with `scale: fraction` lies in [0, 1]; an event's value is a reference string. A number a float can hold converts to a 64-bit float without overflow and is neither infinite nor NaN. `acceptance_rule.score.name` names no verdict or event signal; its `target` is a number a float can hold, and lies in [0, 1] when `acceptance_rule.score.scale` is `fraction`; `acceptance_rule.window_days` is at most 999999999; and `acceptance_rule.requires` names no score or event signal. |
| W196 | `run.receipt` is present and `producer.name` is not loopmath's; only loopmath writes receipts. |
| W182 | An `ext` key begins with `dev.dagr.`, loopmath's namespace before its rename. It is still read through 0.4; write `dev.loopmath.` instead. |

The reference checker also reports W001 when the `jsonschema` package is
missing (schema validation is skipped) and E011 when it cannot load a schema
file. These describe the checker's environment, not the document.

### 6.4 From 0.4

0.4 adds the rules E212 to E218, W219 and W220, for flow documents, `control.checks`
and `run.flow`; each rule's row is added here with the part of 0.4 it checks.

| Code | Rule |
|---|---|
| E212 | `flow.digest`, when present, equals the flow's digest (section 12.4). |
| E213 | In `flow.allow` (section 12.3): every `pieces` key names a piece of the base workflow; every merge's `replace` names 2 or more base pieces, each joined to the next through an artifact the first produces and the second consumes; every merge's `into.id` is new among the base workflow's piece and artifact ids; merge ids are unique; every `checks.remove` entry names a check id of the base workflow when it uses `control.checks`, else a gate piece; and every `widths` list with a value above 1 comes with a `select` setting. A reader that cannot resolve a base workflow given by reference skips the parts that read it. |
| E214 | Judge ids are unique. Every judge `rate` lies in [0, 1]. A judge with `scale: linear` has `range` [lo, hi], two finite numbers of absolute value at most 2**53, with lo < hi (integers up to 2**53 are exact as floats). |
| E215 | An objective of kind `cheapest_within_margin` has `score` and a `margin`. An objective of kind `best_under_cost` has `score` and a `cost_cap_usd` above 0. Other kinds need nothing here. A `margin` or `cost_cap_usd`, for any kind, is a number a float can hold (as in E195). A `margin`, for any kind, is above 0, and the planner's check figure for it at a first plan's assumed spreads (with k the flow's judges that are not person judges, at least 1) is within its pair limit: the margin a first plan refuses is refused here, in the planner's words. `objective.score` and every judge `score` name no verdict of `acceptance_rule.requires`. The flow's `acceptance_rule.score.target` is a number a float can hold, and lies in [0, 1] when its `scale` is `fraction`; its `window_days` is at most 999999999. |
| E216 | `inputs.ids` and `inputs.holdout` have no duplicates, and every `holdout` id is among `ids` when both are given. |
| E217 | In an inline workflow with `control.checks` (section 11): the workflow has no `control.gates`, `control.repair` or `control.ext["dev.loopmath.gate_rules"]`; check ids are unique; every `after` and every non-null `on_fail` names a piece; and no two checks share both `after` and `rule`. Nested inline workflows are checked the same way. Below 0.4, `checks` is an unknown field and E217 does not apply. |
| E218 | A run with `run.flow` (section 12.5) has `run.configuration`. Across documents checked together, when a flow document with the `id` and `version` that `run.flow` names is among them: `run.flow.digest`, when given, is that flow's digest as section 12.4 computes it, and `task.id` is among the flow's `inputs.ids` when it lists ids. With several such flow documents, each part holds when it holds for one of them. Below 0.4, `run.flow` is an unknown field and E218 does not apply. |
| W219 | Across documents checked together, on a run whose `run.flow` names a flow document among them: every score signal whose `source.kind` is `judge` or `user` names one of that flow's judges in `source.ref`. |
| W220 | A score signal whose `source.kind` is `judge` or `user` lies in the linear range it is read by; a bound itself is in range. Alone: when the signal has `scale: linear` and `ext["dev.loopmath.judge"].range` is a range E214 accepts, its value lies in that range. Across documents checked together, on a run whose `run.flow` names a flow document among them: when `source.ref` names a judge with `scale: linear` and a range E214 accepts, the value lies in that range; with several such flow documents, it holds when it holds for one of them. A value no float can hold is E195's and gives no warning. |

In a flow document, E190 to E193 and E217 apply to `flow.configuration` as
they apply to `run.configuration`. The rules about the run, nodes, edges,
attempts, artifacts, events and groups find nothing there, since a flow
document carries none of them; the producer rules (E180, W180) and the
namespace rules apply as in a run document.

E218's comparison with the flow, W219 and W220's judge range need the run and
its flow in one check (`loopmath ocp validate FLOW RUN...`), as E194 needs
every member of a slate: a run of a flow checked alone gets none of them.
W220's signal range applies to a run checked alone.

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
- `control`: `gates`, the pieces after which a check runs (section 11),
  sorted (`[]` when absent); `repair`, which maps a gate to the piece that
  reruns when its check fails (`{}` when absent); `budget`, which counts
  rounds including the first, read as max(1, budget) when it is an integer
  and as 1 otherwise (a JSON number with a fraction part, such as `2.0`,
  reads as 1); `rescue` reduced to its `kind` and `ref` when present; and
  `gate_rules`, the value of `control.ext["dev.loopmath.gate_rules"]` when
  that key is present and not empty, because a rule other than the role
  default changes what runs. Other `ext` keys are left out. An absent
  `control` reads as `{"budget": 1, "gates": [], "repair": {}}`.
- `control` when `control.checks` is present (from 0.4, section 11):
  - When no piece has more than one check, the checks are turned into the
    form above before hashing: `gates` is the `after` values, sorted;
    `repair` maps each `after` to its non-null `on_fail`; `gate_rules` maps
    each `after` whose `rule` differs from its piece's role default to that
    rule, and is included only when not empty (a check whose `after` names
    no piece takes the default of any other role); `budget` and `rescue` are
    read as above. The id is then exactly the one of the same workflow
    written with gates, when its `gate_rules` names only rules that differ
    from the role default, which is what loopmath writes. A gates workflow
    that names a piece's default rule in `gate_rules` keeps its 0.3 id, and
    its checks twin has the id of the same workflow without that entry.
  - When some piece has two or more checks, `control` is
    `{"budget": B, "checks": C}` plus `rescue` when present, both read as
    above. C lists `{after, on_fail, rule}` for each check, with `on_fail`
    null when absent, sorted by `after`, then `rule`, then `on_fail` (null
    first). `gates`, `repair` and `gate_rules` do not appear.

  Check `id` and `ext` are left out in both cases, as gate ids are in 0.3,
  and `gates`, `repair` and `gate_rules` next to `checks` are ignored (E217
  reports them). Nested workflows follow the same rule. The rule reads
  `checks` in a document of any version; 0.3 does not define the field, so
  no 0.3 id changes.

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
- `examples/v0.4/`: `lit-brief.flow.ocp.json`, a flow document (section 12),
  `lit-brief.run-1.ocp.json` and `lit-brief.run-2.ocp.json`, two runs of that
  flow (section 12.5), and `two-checks.ocp.json`, a run whose review piece has
  two checks (section 11). Each validates with no findings, alone and with the
  others.
- `examples/golden/v0.4/`: a pass file and a fail file for each rule of
  section 6.4, with `expected.json` as for 0.3, and for E171's `simulated`
  exception. The E218 and W219 files are checked together with
  `examples/v0.4/lit-brief.flow.ocp.json`, flow first; `E218-fail` and
  `W219-fail` pass when checked alone.
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

## 11. Several checks after one piece

From 0.4, `control.checks` lists the checks after a workflow's pieces. Each
check is an object `{id, after, rule, on_fail, ext}`:

- `id` is unique among the workflow's checks.
- `after` names the piece whose finished attempt the check judges.
- `rule` is an open string. The recommended forms are `tests_pass`,
  `review_approve`, `referee_pick`, `command:<name>` and
  `judge:<judge id>>=<target>`.
- `on_fail` names the piece that reruns when the check fails, or is null to
  stop. An absent `on_fail` reads as null.

When an attempt at piece P finishes, every check with `after: P` runs, and P
passes when all of them pass. When any fails, a rerun follows:

- The piece that reruns is the failing checks' `on_fail` piece.
- When failing checks name different pieces, the earliest of those pieces in
  topological order reruns.
- When every failing check's `on_fail` is null, the run stops.

Each rerun is a new round and counts against `control.budget`.

A workflow states its checks in one of two forms:

- the 0.3 form: `control.gates`, `control.repair` and
  `control.ext["dev.loopmath.gate_rules"]`;
- the 0.4 form: `control.checks`.

Using both in one workflow is an error (E217). The 0.3 form keeps its
meaning. `control.gates` are the pieces after which a check runs. The check's
rule is the piece role's default (`reviewer`: `review_approve`, `referee`:
`referee_pick`, `tester`: `tests_pass`, any other role: `review_approve`)
unless `control.ext["dev.loopmath.gate_rules"]` names another.
`control.repair` maps a gate to the piece that reruns when its check fails.

A producer may write either form when every piece has at most one check,
because both give the same configuration id (section 7). This holds for gates
whose `gate_rules` names only rules that differ from the role default, which
is what loopmath writes. A gates workflow that names a piece's default rule
there keeps its 0.3 id, and its checks twin has the id of the same workflow
without that entry. A producer writes `checks` when a piece has several
checks, or when a check carries an `id` or `ext` it must keep. In
`examples/v0.4/two-checks.ocp.json`, two checks follow the review piece: the
reviewer must approve and `command:lint` must pass, and either failure sends
the work back to the implementer.

## 12. Flow documents

From 0.4 a document has a `kind`: `run` or `flow`. `kind` is `run` when
absent, so every earlier document is a run document.

- A run document is as in 0.3. It requires `ocp`, `producer`, `privacy`,
  `run` and `nodes`.
- A flow document requires `ocp` (0.4 or later), `kind: "flow"`, `producer`,
  `privacy` and `flow`. It carries no `run`, `nodes`, `edges`, `attempts`,
  `artifacts`, `events` or `groups`. `ext` is allowed at the top level and in
  `flow`.

The schema enforces the kinds, the required and forbidden members and the
field types below (E010). A flow document at 0.3 fails, because the 0.3 schema
requires `run` and `nodes`.

A flow document describes an experiment on a fixed multi-step workflow: the
base configuration, the inputs, the acceptance rule, the objective, the
hypothesis, the judges and the edits a planner may try. A reader that learns
from runs (fits, comparisons, run lists) skips flow documents. A store refuses
to import a flow document as a run and says why.

### 12.1 The flow object

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Stable flow id, with the length limit of a run id. |
| `version` | yes | Integer, at least 1. A producer raises it on any change except to `title` or `ext`. |
| `title` | no | Short text. |
| `digest` | no | `flw_` plus 12 hex digits (section 12.4). When present it must match (E212). |
| `configuration` | yes | The base configuration, an object exactly as `run.configuration`: id, workflow inline or by reference, settings, source. Its id follows section 7, and E190 to E193 and E217 apply to it. |
| `inputs` | yes | `{ref, type, ids, holdout, n}`. `ref` is short text naming where the inputs live, for example `questions.jsonl`. `type` uses the `task.type` vocabulary, for example `research`. `ids` is an optional list of task ids, `holdout` an optional list of task ids kept for the final check, and `n` an optional count (E216). |
| `acceptance_rule` | yes | The same object as `run.acceptance_rule`. |
| `objective` | yes | `{kind, score, margin, reference, cost_cap_usd, ext}`. `kind` is an open string; the recommended values follow this table (E215). |
| `hypothesis` | no | Short text: what the user expects, in their words. |
| `judges` | no | A list of judge objects (section 12.2). |
| `allow` | no | The edits a planner may try (section 12.3). |
| `ext` | no | Producer data. |

The recommended `objective.kind` values:

- `cheapest_within_margin`: the cheapest configuration whose score `score` is
  no worse than the `reference` configuration's by `margin`, in the score's
  units. This is loopmath's default, with margin 0.10 on `quality`.
- `cost_per_accepted`: the lowest dollars per accepted run under
  `acceptance_rule`.
- `best_under_cost`: the highest `score` with a mean run cost under
  `cost_cap_usd`.

`reference` is `base`, the flow's configuration, when absent.

A loopmath producer keeps the edits the hypothesis names in
`flow.ext["dev.loopmath.hypothesis_edits"]`: a list of edit objects (`kind`,
and `piece`, `key` and `value` as the kind needs), in the order the hypothesis
names them. `loopmath flow init` writes only single edits that `allow`
permits. loopmath's planner puts the factors they name first. Like the rest
of `ext`, the member is outside the digest (section 12.4).

A flow that `loopmath flow init --from-ocp` or `--from-rows` prefilled from
logged runs records them in `flow.ext["dev.loopmath.prefill"]`: `from`, the
run ids in the order read; `how`, `declared` (the runs declared one
configuration, taken exactly) or `inferred` (the steps were inferred from
them); and `confidence`, a heuristic figure of how far the runs agree, 1.0
when declared, not a probability. It is outside the digest too.

### 12.2 Judges

| Field | Meaning |
|---|---|
| `id` | Unique within the flow (E214). Score signals name it in `source.ref`. |
| `kind` | Open: `model`, `person`, `command`. |
| `model` | A model reference, for `kind: model`. |
| `rubric` | Short text naming the rubric and its version, for example `rubric.md#v2`. |
| `score` | The score name the judge writes. `objective.score` when absent. |
| `scale` | `fraction` (the default) or `linear`. A linear judge gives `range: [lo, hi]`, and readers map its scores to `(value - lo) / (hi - lo)`. |
| `rate` | A fraction in [0, 1]: the share of runs this judge scores. 1 when absent; a person who spot-checks 1 run in 5 has 0.2. |
| `blinded` | Boolean: the judge does not see which arm made the output. |

### 12.3 Allowed edits

`allow` lists the edits a planner may try (E213):

- `pieces`: a piece id of the base workflow to
  `{models, efforts, options, widths, drop, select}`.
  - `models` is a list of model references.
  - `efforts` is a list of effort labels.
  - `options` maps an option key to a list of values. A prompt or tool
    version is one, for example `{"prompt": ["v3", "v4"]}`.
  - `widths` is a list of integers of at least 1.
  - `drop: true` lets the planner remove the piece.
  - `select` is the referee setting used when a width above 1 is tried.
- `budgets`: a list of round budgets of at least 1.
- `checks`: `{add, remove}`, a list of check objects (section 11) to add and
  a list of checks to remove: check ids, or gate pieces when the base
  workflow uses `control.gates`.
- `merges`: a list of `{id, replace, into, setting}`.
  - `replace` lists 2 or more pieces that form a chain.
  - `into` is the new piece, `{id, role}`.
  - `setting` is its setting. The merged prompt's version goes in
    `setting.options`.

How each edit rewires the workflow is the producer's business.

### 12.4 Digest

`flow.digest` is `flw_` followed by the first 12 hexadecimal digits of the
SHA-256 of the canonical JSON (the encoding of section 7) of this object:

```
{"acceptance_rule": ..., "allow": ...,
 "configuration": <flow.configuration.id>, "hypothesis": ...,
 "inputs": ..., "judges": ..., "objective": ...}
```

An absent member is null. Members are hashed as written, with no defaults
filled in. The flow's `id`, `version`, `title`, `digest` and `ext` are left
out, and the configuration enters through its id alone. Any other edit
changes the digest, so a reader can tell when a flow was edited and kept its
version number. The reference implementation is
`loopmath.ocp.canonical.flow_digest`.

`examples/v0.4/lit-brief.flow.ocp.json` is a flow for a research workflow
whose pieces scout, extract, synthesize and critique, from a question to a
one-page brief. It runs on 20 questions and holds 8 back, looks for the
cheapest configuration within 0.10 of the base on `quality`, has two model
judges and a person who scores one run in five, and allows other models at
extract, other efforts at synthesize, dropping the critic, a width of 2 at
scout, budgets 1 to 3 and one merge of extract and synthesize.

### 12.5 Runs of a flow

From 0.4 a run document may name the flow it is a run of, in `run.flow`:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | The flow's `flow.id`, 1 to 200 characters. |
| `version` | yes | The flow's `flow.version`, an integer from 1. |
| `digest` | no | The flow's digest (section 12.4). Recommended: checked with the flow, it must match (E218). |
| `arm` | no | The arm the run ran, at most 200 characters: `base` for the flow's configuration, or the id of an arm a planner made. |
| `plan` | no | The id of the plan that chose the arm, at most 200 characters. |
| `split` | no | The part of the flow's inputs the task came from, at most 100 characters. The recommended values are `train` and `holdout`. |
| `ext` | no | Producer data, as in every `ext`. |

- The run carries `run.configuration`: the configuration of its arm, so a run
  with a send-back and one without are runs of one configuration (E218).
- loopmath's `flow record`, given `--allow-unplanned`, records a run its
  current plan did not ask for as arm `unplanned`, with the run's own
  configuration, and keeps the arm the run named in
  `run.flow.ext["dev.loopmath.flow"].named_arm`. It refuses such a run whose
  configuration is `base`'s or a plan arm's, naming that arm. Under a stale
  plan, one made for another digest of the flow, it refuses a run of any arm
  but `base`, with the flag or without: the refusal says the plan is stale
  and asks for `loopmath flow plan`.
- Checked with its flow, the run gives the flow's digest and a task among the
  flow's `inputs.ids` (E218), and its judge and user scores name the flow's
  judges (W219) and lie in a linear judge's range (W220).
- A run a simulator drew, not billed, says so on its costs with
  `basis: simulated`. A loopmath producer adds the object
  `run.ext["dev.loopmath.simulate"]`, without which E171 refuses that basis.

`examples/v0.4/lit-brief.run-1.ocp.json` and `lit-brief.run-2.ocp.json` are
runs of the base arm of `lit-brief.flow.ocp.json` on question `q07`: one pass,
then one where the critic sent the brief back to synthesize once. Both have
the configuration id `cfg_9bc4f6e876f9`. Run 1 has a score from each of the
flow's three judges, the person's among them; run 2 from the two model judges.
