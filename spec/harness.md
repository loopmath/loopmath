# The harness record, `loopmath.harness/0.1`

A harness is what an agent runs with besides its model: the prompt files, skills, plugins, MCP servers, hooks, sub-agents and commands in its working tree. The harness record names those parts, so the belief model can learn what each part does to cost and success and price a tree it has not run.

This is the 0.1 record. Changes from here are additive only: a new optional field, a new factor kind or a new layout rule.

## 1. Where it sits

- On an OCP run document, in `run.ext["dev.loopmath.harness"]`. OCP section 4 asks for reverse-domain `ext` keys.
- Readers also accept a bare `run.ext["harness"]` holding the same record. `dev.loopmath.harness` wins when both exist. Writers write `dev.loopmath.harness` only.
- No OCP version bump. A reader that does not know the key ignores it.
- One record per run, applying to every piece of the run. In the model, each piece's factors sit under that piece's own harness label (section 8), so a Claude Code piece and a Codex piece in one run learn separately.
- `loopmath share` does not export the record. The share format is an allow-list, and factor names can be private.

## 2. The record

```json
{"schema": "loopmath.harness/0.1",
 "digest": "sha256:<64 hex of the normalized tree>",
 "parent": null,
 "source": "hand",
 "layouts": ["claude-code"],
 "factors": ["hook:PostToolUse:Edit|Write", "mcp:github", "prompt:3f2a9c1d", "skill:pdf"],
 "tool": {"name": "...", "version": "...", "run_id": "...", "task_id": "...", "stage": "..."}}
```

| field | required | meaning |
|---|---|---|
| `schema` | yes | `loopmath.harness/0.1`. A reader reads any `loopmath.harness/0.x`, and later majors' `factors`, the same way. |
| `digest` | yes | `sha256:` and 64 lowercase hex characters: the digest of the normalized tree (section 3). A producer with only a factor list may write any stable `sha256:` id of its own; the model never reads `digest`. |
| `parent` | no | The digest of the tree this one was edited from, or null. |
| `source` | no | Free string, default `hand`: who made the tree. Any value is kept. |
| `layouts` | no | The layouts that found files (section 4). Informational. |
| `factors` | yes | The named parts of the tree (section 5). May be empty. |
| `tool` | no | The producing tool's ids, all strings, every member optional. |

Unknown fields are kept as they are, at every level.

## 3. The digest

The digest covers the harness files only, never the rest of a repository: changing `src/app.py` leaves it as it was.

1. **Which files.** The files the matched layouts name (section 4), under the tree's root. Ignored everywhere: `.git/`, `node_modules/`, `__pycache__/`, `*.pyc`, `.DS_Store`, and `README.md` stubs directly inside `scripts/`, `skills/`, `sub_agents/`, `plugin/` and `.codex/`.
2. **Paths.** Relative to the root, `/` separators, Unicode NFC, sorted by their UTF-8 bytes.
3. **Content.** A file that decodes as UTF-8 loses a leading BOM, and CRLF and lone CR become LF. Nothing else changes: trailing spaces and the final newline count. A file that does not decode is read as its raw bytes. A symlink is never followed: its entry is the link text, marked as a link.
4. **Hash.** For each file in order, the line `<path>\0<f or l>\0<sha256 hex of the content>\n`. The digest is `sha256:` plus the SHA-256 hex of those lines joined. An empty tree has a fixed digest, the SHA-256 of the empty string.

## 4. Layouts

`record_from_tree(path, layout="auto")` reads one of these rule sets:

- `agents`: the Claude Code and Codex files that a repository's working tree holds. Always on in `auto`.
- `harness2`: a research harness's root-level surface files (`systemprompt.md`, `guardrails.md`, `LongTermMEMORY.md`, `tool_descriptions/tool_guidance.md`, `skills/`, `scripts/`, `sub_agents/`, `plugin/`). In `auto` it is on only when one of the first four files is at the root, so a repository's own `scripts/` and `skills/` do not count. `layout="harness2"` turns on both rule sets.

The record's `layouts` lists `claude-code`, `codex` and `harness2` for the rule sets that found at least one file.

## 5. Factors

**Naming.** A factor is `<kind>:<name>`.

- `kind` is lowercase `[a-z0-9-]+`. A reader lowercases it.
- `name` is any non-empty text without a newline, NFC and stripped of outer whitespace, and otherwise kept as written: case and spaces matter, and `hook:PostToolUse:Edit|Write` is one name.
- `factors(record)` turns a string without a kind into `other:<string>` and drops a value that is not a non-empty one-line string. The list is sorted and has no repeats.
- Any kind is kept and modelled, known or not.

Named parts (skill, plugin, mcp, hook, subagent, command, script, memory) are named by their name, not their content. Evidence therefore pools across trees and across edits of the same part. The always-on prose files have no name. They are named by the first 8 hex characters of the SHA-256 of their normalized text (section 3), so an edit to them is a new factor, which teaches little until it repeats. An empty file gives no factor.

**The map.**

| factor | Claude Code (`agents`) | Codex (`agents`) | `harness2` |
|---|---|---|---|
| `prompt:<sha8>` | `CLAUDE.md`, `.claude/CLAUDE.md`, `CLAUDE.local.md`, each file its own factor | `AGENTS.md`, `AGENTS.override.md` | `systemprompt.md` |
| `guardrails:<sha8>` | | | `guardrails.md` |
| `memory:<name>` | | | `LongTermMEMORY.md` gives `memory:LongTermMEMORY` |
| `tool-guidance:<sha8>` | | | `tool_descriptions/tool_guidance.md` |
| `subagent:<name>` | `.claude/agents/<name>.md` (its frontmatter `name:`, else the file stem) | | `sub_agents/<name>/AGENT.md` |
| `skill:<name>` | `.claude/skills/<name>/` (every file under it is in the digest) | `.agents/skills/<name>/`, `.codex/skills/<name>/` | `skills/<name>/SKILL.md` and `skills/<name>/scripts/*` |
| `script:<name>` | | | `scripts/<name>.py` |
| `plugin:<name>` | `enabledPlugins` entries set to true in `.claude/settings.json` and `.claude/settings.local.json`, as `plugin:<name>@<marketplace>` | | `plugin/<name>.ts` |
| `hook:<event>[:<matcher>]` | `hooks` in `.claude/settings.json` and `.claude/settings.local.json`: one factor per event and matcher; no matcher, or `*`, gives `hook:<event>` | `.codex/hooks.json` | `.codex/hooks.json` |
| `mcp:<name>` | `mcpServers` keys in `.mcp.json` | `[mcp_servers.<name>]` tables in `.codex/config.toml` | |
| `command:<name>` | `.claude/commands/<name>.md` | | |

The whole of `.claude/settings.json`, `.claude/settings.local.json`, `.mcp.json`, `.codex/config.toml` and `.codex/hooks.json` is in the digest. A permission change therefore moves the digest but makes no factor. Hook commands, environment values and MCP server commands never appear in the record; only names do.

Nested `CLAUDE.md` and `AGENTS.md` files below the root are not read in 0.1. Neither are the user's home files (`~/.claude`, `~/.codex`): the record describes the repository, not the machine.

## 6. Python API

```python
from loopmath import harness
harness.SCHEMA                     # "loopmath.harness/0.1"
harness.EXT_KEY                    # "dev.loopmath.harness"
harness.digest(path, layout="auto") -> str                       # "sha256:<hex>"
harness.record_from_tree(path, source="hand", *, parent=None, tool=None, layout="auto") -> dict
harness.factors(record) -> tuple[str, ...]                      # sorted, unique, unknown kinds kept
harness.normalize_record(record) -> dict                         # factors cleaned, every other field kept
harness.run_record(doc) -> dict | None                           # the record on a run document, either key
```

`record_from_tree` reads files and never runs anything. It never follows a symlink and never reads outside `path`.

## 7. Recording from the CLI

- `loopmath run record ... --harness PATH|auto|none` and `loopmath run finish ... --harness PATH|auto|none` write the record.
  - `PATH` is a tree.
  - `auto` is the run's working tree: the git top level of the folder the run's attempts ran in (`--cwd`, or the session's own folder), else of the folder the command runs in. When the attempts ran in two different trees, `auto` refuses with exit 1 and asks for `--harness PATH`. When no git working tree holds the folder, `auto` refuses with exit 1 too and names it. In both cases nothing is written and the run stays open. This is the flag's behaviour; for the config key, see below.
  - `none` writes nothing. A record already on the run stays.
  - The record holds no path.
- When a record is written, the text output has one line naming it: `harness: recorded sha256:<first 12 hex> (<source>; <layouts>; <n> parts: ...) from <tree>`. `--json` adds a key `harness` with `tree`, `digest`, `source`, `layouts` and `factors`. Without a record written, neither appears.
- `loopmath runs RUN` shows a run's record in a `harness` line, in the same form without the tree.
- The default is `none`. The config key `harness.record` (`none` or `auto`) sets it; the flag wins.
  - When `auto` comes from the config key, not the flag, and finds no single tree (no git working tree, or two trees), nothing is written and the run finishes as it would without the key.
  - The text output adds one note line, `harness: no record: <why> (harness.record is auto; --harness PATH records one)`, and `--json` is unchanged.
- `run import` keeps whatever record a producer wrote.

## 8. What the model does with it

- **Terms.** In the cost head, and in the success and score heads, each factor `f` of a piece with harness label `h` adds a node with parent `harness:<h>` and scale group `hfactor`. Each node is an independent deviation from its label, `N(0, phi_hfactor^2)`; the default scales are 0.2 on log cost and 0.3 on the success and score scales, and the fit sets `phi_hfactor` by empirical Bayes, as it does for every group.
  - Cost rows, one per attempt, carry the factors at 1. Success and score rows carry each piece's factors at `1/n`, as that piece's harness term, so a run whose pieces share a label carries each factor at 1.
  - The gate head has no harness term, and no factor terms.
- **The node id** is `hfactor:<esc(h)>|<f>`, where `esc` writes `%` as `%25` and `|` as `%7C` in the label. The first bare `|` ends the label, and the factor is not escaped. So `hfactor:claude-code|skill:pdf` reads as written, and two (label, factor) pairs never share a node. Outputs carry the raw `harness` and `factor` strings beside the id; no reader needs to parse it.
- **A part with no data** has effect 0, so a tree with unseen parts keeps its harness label's mean, with the factor scale's variance added per unseen part. Integrating that extra log-cost uncertainty raises the predicted dollar mean slightly, by about `exp(phi^2/2)`.
- **No record, no change.** A run with no record adds no node, and a store with no record anywhere gives the numbers it gave before records existed.

### `posterior --json`

When the fit has factor nodes, the payload has a key `factors` with one item per node, sorted by harness then factor, and a one-line `factor_note` beside it:

```json
{"factor": "skill:pdf", "harness": "claude-code", "kind": "skill", "name": "pdf",
 "node": "hfactor:claude-code|skill:pdf", "runs": 12,
 "cost": {"mean": 0.87, "lo": 0.74, "hi": 1.02, "level": 0.8, "lo95": 0.68, "hi95": 1.11},
 "success": {"mean": 4.1, "lo": -1.0, "hi": 9.0, "level": 0.8, "lo95": -3.6, "hi95": 11.5,
             "log_odds": {"mean": 0.18, "lo": -0.04, "hi": 0.41, "level": 0.8, "lo95": -0.16, "hi95": 0.52}}}
```

- `cost` is the multiplier on an attempt's dollars when the part is present, everything else equal.
- `success` is the change in percentage points at the head's baseline, with the log-odds beside it.
- Ranges are the 10/90 and 2.5/97.5 percentiles of the 400 draws. A head without the node gives `null`.
- `success` carries `calibrated: false`: its ranges missed their nominal coverage in simulation (section 9). The text output says so in one line.
- The nodes also appear in the `harness` section of `levels`, beside their labels, and `--level harness` shows the labels and their parts.

### `recommend`

The search is unchanged: every candidate is priced with its bare harness label.

- `--harness-variants [N]` adds a key `harness_variants` after `search`. It prices the N trees the fit saw on most runs (default 5), with the tree's factors on every piece: the chance of success, the run cost, the cost per accepted result, `p_better` and `delta_p_success`.
  - A tree is priced on the configurations its runs used, at most two, most runs first. The fit lists them per tree in `design.json`, with one run to read each from. It is also priced on the default pick's and the goal's configurations.
  - Each priced configuration is an entry of the variant's `configs`, by id, with `label`, `roles` (`runs`, `default_pick`, `goal`) and, for a run configuration, `runs`.
  - A configuration is not priced when no part of the tree has a node under its harness labels in any head. There the tree's price would be the factor scale's spread around the bare label, not evidence. The same goes for a tree with no named parts.
  - The variant's `not_priced` lists each configuration it is not priced on, with `id`, `role`, `label` and a `reason`. That includes a run configuration whose run is no longer in the store, and a fit made before the per-tree list (refit to price the tree on its runs' configurations).
  - `p_better` is the share of the 400 joint draws in which the variant's cost per accepted result is below the bare label's.
  - `delta_p_success` is the change in the chance of success on the same draws.
  - Both are paired comparisons.
- `--harness PATH|auto` adds that tree as the first variant (`asked: true`), even when the fit never saw it. It is priced as above: on its runs' configurations when the fit saw it, and on the recommendation's where some part has data.
- In a priced variant, `p_success`, `delta_p_success` and `cost_per_accepted_usd` carry `calibrated: false`, and each priced configuration has `p_better_calibrated: false` (section 9).
- The text output prints one line that says how many variants are in `--json` and what is not calibrated. When no variant is priced, it says so instead. It then prints one line for each of the default pick and goal that some variants are not priced on, with the reason.
- Without either flag, the payload has no new key.

## 9. What an estimate means, and its limits

- A factor's estimate is the change in cost and success when a named part is present, against the same harness without it, pooled over every recorded version of that part. It is a conditional association, not the effect of one edit: an edited skill body is the same factor.
- A part on every run of a harness cannot be told apart from the harness label. Shrinkage then splits the credit by the scales, which is not evidence of an effect. Parts that always appear together are identified only as a sum.
- **Success intervals are not calibrated.** Success factor effects are shrunk toward 0, more at large effect sizes and by more than the factor scale alone explains. In the recovery check (40 simulated stores of 1,200 runs; per head, 10 named factor effects drawn from the factor scale and 6 fixed at 0), the success head's 80% and 95% intervals held 60.0% (240 of 400) and 78.5% (314 of 400) of the nonzero effects. They held 95.0% and 99.6% of the zero effects. Both bands missed their nominal coverage. Effects set by hand at up to 2 scales fared worse: 36.2% and 61.5%. The model's other success effects of that size show the same pull without any harness record, so it is not particular to harness parts. The JSON marks these ranges `calibrated: false`.
- **Cost intervals met the stated rates.** On the same check, the cost head's 80% intervals held 78.9% of the effects and its 95% intervals 95.8%, for zero and nonzero effects alike, and its estimates were about 0.97 of the truth. The stated rates (80% intervals cover 70 to 90% of the truths, 95% intervals 90 to 99%) are claimed for the cost head only, and only for effects drawn from the factor scale, as the model assumes. They are not a claim about any one fixed effect.
- **What the check covers.** The simulation holds the group scales, the fixed coefficients and the residual sigmas constant; no scale is drawn from its hyperprior. These are recovery rates for factors' own additive deviations in this fixed-hyperparameter simulation; they do not validate full prior-predictive or run-level success-probability coverage.
- **Variant comparisons.** In `recommend --harness-variants`, `p_success`, `delta_p_success`, `cost_per_accepted_usd` and `p_better` rest on the success effects, so these comparisons lack validated calibration. They carry `calibrated: false` (`p_better_calibrated: false`). `run_cost_usd` rests on the cost effects alone.
- A factor applies to every piece in the run, whatever its harness. A Codex piece gets a node for a `.claude/skills` entry it never reads. That node stays near 0 only when the trees vary, and otherwise shares credit with the parts it always appears with.
- Main effects only. Interactions between factors, and between a factor and a model or a task type, are not modelled.
- Shared exports and prior bundles drop the record, so shared and shipped runs carry no factors.
