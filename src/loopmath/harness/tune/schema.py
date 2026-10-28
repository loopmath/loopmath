"""FIELDS: every config field of `loopmath harness`, the one table the init file, the flags, the JSON Schema, the
skill doc and the mod's configure pane are generated from, so none of them can drift from the file.

A field is (key, type, default, help, choices, source, note). `key` is `<table>.<name>`; fields of the `editors`
array of tables are `editors.<name>` with `per_editor` set. `source` is `study` when the default is the published
study design's value, `product` when it is a product choice (a guard, a path, a switch the study had no use for);
`note` says how a product default departs from the study.

Types: str, int, float, bool, str-list, range (two numbers), duration (a number of seconds, or text such as 90m,
6h or 1h30m).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from . import CONFIG_SCHEMA

__all__ = ["CONFIG_SCHEMA", "EFFORTS", "FIELDS", "EDITORS", "Field", "TABLES", "TYPES", "by_key", "defaults",
           "editor_defaults", "flag", "json_schema", "parse_duration", "parse_value", "table_fields"]

TYPES = ("str", "int", "float", "bool", "str-list", "range", "duration")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
FOCUSES = ("procedure", "domain", "file")


@dataclass(frozen=True)
class Field:
    key: str
    type: str
    default: Any
    help: str
    choices: tuple = ()
    source: str = "study"
    note: str = ""
    minimum: float | None = None

    @property
    def table(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.key.split(".", 1)[1]

    @property
    def per_editor(self) -> bool:
        return self.table == "editors"


def _f(key, type, default, help, choices=(), source="study", note="", minimum=None) -> Field:
    return Field(key, type, default, help, tuple(choices), source, note, minimum)


FIELDS: tuple[Field, ...] = (
    # task
    _f("task.repo", "str", ".", "git working tree the agent works in", source="product"),
    _f("task.base_ref", "str", "HEAD", "commit every run starts from", source="product"),
    _f("task.prompt", "str", "", "the task text", source="product"),
    _f("task.prompt_file", "str", "", "a file holding the task text", source="product"),
    _f("task.set", "str", "", "a JSONL task set: one {id, prompt, base_ref?, score_command?} per line",
       source="product"),
    _f("task.id", "str", "", "stable task id; default: sha256 of the prompt and the base commit, 12 hex",
       source="product"),
    # score
    _f("score.command", "str", "", "your own check, run in the run's worktree after the agent stops",
       source="product"),
    _f("score.kind", "str", "auto", "how the command's result reads: auto (a number if the last line parses as "
       "one, else pass), pass (exit 0 is 100), number (last stdout line, rescaled from score.range), junit",
       ("auto", "pass", "number", "junit"), source="product"),
    _f("score.range", "range", [0, 100], "what a number from the command means: [low, high] maps to 0..100",
       source="product"),
    _f("score.junit_path", "str", "", "the JUnit XML file a junit command writes, in the worktree",
       source="product"),
    _f("score.timeout_s", "int", 900, "seconds the score command may run", source="product", minimum=1),
    _f("score.combine", "str", "auto", "how the command and the judge combine: auto (gate with both, else the one "
       "present), gate (a failed command scores 0, else the judge), command, judge, mean",
       ("auto", "gate", "command", "judge", "mean"), source="product"),
    # judge
    _f("judge.kind", "str", "tester", "the judge: tester (one test-writer session per task writes an acceptance "
       "bundle that scores every run), rubric (a rubric file), none", ("tester", "rubric", "none")),
    _f("judge.model", "str", "claude-sonnet-5-5", "the test writer's model"),
    _f("judge.effort", "str", "high", "the test writer's effort", EFFORTS),
    _f("judge.attempts", "int", 3, "test-writer attempts: a bundle that fails on the starting repo for another "
       "reason than the missing feature is written again", minimum=1),
    _f("judge.tries", "int", 2, "tries to score one run; a run with no score after them is left out of the fit",
       minimum=1),
    _f("judge.rubric_file", "str", "", "the rubric file, for judge.kind = rubric", source="product"),
    # executor
    _f("executor.agent", "str", "claude-code", "the task agent", ("claude-code",), source="product"),
    _f("executor.model", "str", "claude-haiku-5-5", "the task agent's model"),
    _f("executor.effort", "str", "high", "the task agent's effort", EFFORTS),
    _f("executor.max_turns", "int", 200, "turn limit per run", minimum=1),
    _f("executor.timeout_s", "int", 3600, "seconds per run", minimum=1),
    _f("executor.tries", "int", 2, "tries per run when the executor fails outside the harness; then the run is "
       "left out of the fit", minimum=1),
    _f("executor.fallback_model", "str", "", "Claude Code's --fallback-model, empty for none", source="product"),
    _f("executor.permission_mode", "str", "bypassPermissions", "Claude Code's --permission-mode",
       ("acceptEdits", "auto", "bypassPermissions", "manual", "dontAsk", "plan")),
    _f("executor.sandbox", "bool", True, "sandbox the agent's shell: writes only in its folder (its clone's .git "
       "among it), TMPDIR and Claude Code's temp folder, and an executor's reads only in its own run's "
       "folders (the file tools are limited either way)",
       source="product", note="the study ran each task in its own container instead"),
    # base
    _f("base.runs", "int", 2, "base runs, which also give the first noise read", minimum=1),
    _f("base.reuse", "str", "", "a folder of earlier scored base runs to import instead of running them",
       source="product"),
    # signal
    _f("signal.enabled", "bool", True, "check the base runs for a ceiling and a floor"),
    _f("signal.on_ceiling", "str", "stop", "every base run scores the maximum: warn and go on, or stop and deliver "
       "the base before any editor runs", ("warn", "stop"), source="product", note="the study warned and went on"),
    _f("signal.on_floor", "str", "warn", "every base run scores 0: warn and go on, or stop", ("warn", "stop")),
    _f("signal.min_t", "float", 0.0, "an edit is kept when its effect is above min_t standard errors; 0 keeps any "
       "positive effect"),
    # edit
    _f("edit.enabled", "bool", True, "run the editors; off needs edit.from"),
    _f("edit.from", "str", "", "your own edit folders (one subfolder per edit), used when the editors are off",
       source="product"),
    # editors (an array of tables)
    _f("editors.name", "str", "", "the editor's name, used in labels and folders; default: its focus, numbered "
       "when repeated"),
    _f("editors.focus", "str", "procedure", "procedure (how to go about this kind of task), domain (facts and "
       "conventions a newcomer gets wrong) or file (focus_file)", FOCUSES),
    _f("editors.focus_file", "str", "", "a file stating the editor's focus, for focus = file", source="product"),
    _f("editors.model", "str", "claude-opus-5-5", "the editor's model"),
    _f("editors.effort", "str", "high", "the editor's effort", EFFORTS),
    _f("editors.max_edits", "int", 5, "edits one editor may propose", minimum=1),
    _f("editors.attempts", "int", 2, "editor attempts; the retry runs only when an attempt yields no usable edit",
       minimum=1),
    _f("editors.timeout_s", "int", 3600, "seconds per editor session", source="product", minimum=1),
    # screen
    _f("screen.design", "str", "pb12", "the probe design: pb12 (12 single runs, Plackett-Burman), fold-over "
       "(2 x pairs runs), one-at-a-time (one run per edit), full (every subset, up to 4 edits)",
       ("pb12", "fold-over", "one-at-a-time", "full")),
    _f("screen.max_edits", "int", 10, "edits the design takes; the round robin order sets which are kept",
       minimum=1),
    _f("screen.repeats", "int", 1, "runs per probe row", minimum=1),
    _f("screen.pairs", "int", 8, "pairs, for the fold-over design", source="product", minimum=2),
    _f("screen.seed", "int", 0, "seed of the column assignment and the run order"),
    _f("screen.base_rows", "bool", True, "base runs enter the fit as all-off rows"),
    # fit
    _f("fit.select", "str", "least-squares", "what selects the kept edits: least-squares, or head (the shrinkage "
       "model)", ("least-squares", "head")),
    _f("fit.report", "str-list", ["head"], "fits reported beside the selector", ("head",)),
    # final
    _f("final.runs", "int", 1, "runs of the kept set's build; 0 needs deliver.rule fit or always-build",
       minimum=0),
    # check
    _f("check.max_pairs", "int", 3, "replicate base and final pairs the gate rule may add", source="product",
       minimum=0),
    # deliver
    _f("deliver.rule", "str", "final-vs-base-mean", "the delivery rule: final-vs-base-mean (deliver the build "
       "unless its final run scores below the base mean), fit (the build, no final run), gate "
       "(loopmath.gate.decide), always-build (the build whenever an edit was kept)",
       ("final-vs-base-mean", "fit", "gate", "always-build")),
    # reflector
    _f("reflector.enabled", "bool", True, "one session writes a lesson into the role library"),
    _f("reflector.model", "str", "claude-opus-5-5", "the reflector's model"),
    _f("reflector.effort", "str", "high", "the reflector's effort", EFFORTS),
    _f("reflector.when", "str", "end", "when the reflector runs: end, or after-fit", ("end", "after-fit")),
    _f("reflector.timeout_s", "int", 1800, "seconds for the reflector session", source="product", minimum=1),
    # library
    _f("library.enabled", "bool", True, "editors read, and the reflector writes, the role library"),
    _f("library.path", "str", "", "the library folder; default $LOOPMATH_HOME/harness/library", source="product"),
    _f("library.domain", "str", "", "the role key lessons are filed under; default: the repo name",
       source="product"),
    # surface
    _f("surface.scope", "str", "project", "where the harness lives: project (the repo's CLAUDE.md, .claude/skills, "
       ".claude/agents, settings hooks) or user (a clean config folder, as the study ran)", ("project", "user"),
       source="product", note="the study tuned at user scope; project scope measures what apply writes"),
    _f("surface.always_on", "str-list", ["CLAUDE.md"], "always-on files: added text only", source="product"),
    _f("surface.always_on_cap", "int", 8000, "characters the always-on files may hold", minimum=1),
    _f("surface.skills", "bool", True, "skills are editable"),
    _f("surface.agents", "bool", True, "agents are editable"),
    _f("surface.hooks", "bool", True, "hooks are not editable and do not run in tune sessions; true shows the base "
       "settings.json to editors"),
    _f("surface.allow_model_names", "bool", False, "harness files may name a model"),
    # budget
    _f("budget.usd", "float", 10.0, "dollar cap per task, checked before every session at its high estimate",
       source="product", note="a guard; the study capped whole streams", minimum=0),
    _f("budget.runs", "int", 40, "executor run cap per task", source="product", minimum=1),
    _f("budget.wall", "duration", "6h", "wall-clock cap per task", source="product"),
    _f("budget.recheck_after_base", "bool", True, "price the plan again from the base runs' measured cost",
       source="product"),
    _f("budget.set_usd", "float", 0.0, "dollar cap for a whole task set; 0: no set cap", source="product",
       minimum=0),
    # loop
    _f("loop.rounds", "int", 1, "rounds of edit to deliver, each on the last round's delivered harness",
       minimum=1),
    _f("loop.overlap", "bool", False, "not in this release; true is refused", source="product"),
    # run
    _f("run.width", "int", 4, "executor sessions at once (judge sessions count against it)", minimum=1),
    _f("run.editor_width", "int", 0, "editor sessions at once; 0: all", minimum=0),
    _f("run.runner", "str", "local", "where sessions run: local, or herdr (one pane each)", ("local", "herdr"),
       source="product"),
    _f("run.work_root", "str", "", "the tunes folder; default $LOOPMATH_HOME/harness/tunes", source="product"),
    _f("run.keep_worktrees", "str", "failed", "worktrees kept after scoring", ("failed", "all", "none"),
       source="product"),
    _f("run.record", "bool", True, "write each executor run into the loopmath store as a harness-runs row",
       source="product"),
    # auth
    _f("auth.mode", "str", "token-file", "how sessions log in: token-file (a `claude setup-token` token, passed by "
       "file descriptor) or api-key-env", ("token-file", "api-key-env"), source="product"),
    _f("auth.token_file", "str", "", "the token file, mode 600; read at start, never put in argv, env or files",
       source="product"),
    _f("auth.api_key_env", "str", "ANTHROPIC_API_KEY", "the variable holding the API key, for api-key-env",
       source="product"),
)

TABLES = tuple(dict.fromkeys(f.table for f in FIELDS))

EDITORS = (
    {"name": "procedure", "focus": "procedure", "focus_file": "", "model": "claude-opus-5-5", "effort": "high",
     "max_edits": 5, "attempts": 2, "timeout_s": 3600},
    {"name": "domain", "focus": "domain", "focus_file": "", "model": "claude-opus-5-5", "effort": "high",
     "max_edits": 5, "attempts": 2, "timeout_s": 3600},
)
"""The default editors: procedure, then domain."""

_BY_KEY = {f.key: f for f in FIELDS}


def by_key(key: str) -> Field:
    """The field of a dotted key; `editors.2.model` and `editors.model` both give `editors.model`. KeyError."""
    parts = key.split(".")
    if len(parts) == 3 and parts[0] == "editors" and parts[1].isdigit():
        key = f"editors.{parts[2]}"
    return _BY_KEY[key]


def table_fields(table: str) -> list[Field]:
    return [f for f in FIELDS if f.table == table]


def editor_defaults(index: int, taken: Iterable[str] = ()) -> dict:
    """The defaults of editor `index` (from 1) when `--editors N` adds it: focus cycles procedure, domain,
    procedure, ...; the name is the focus, numbered (procedure-2) when taken."""
    focus = FOCUSES[(index - 1) % 2]
    out = {f.name: _copy(f.default) for f in table_fields("editors")}
    out["focus"] = focus
    name, k, taken = focus, 1, set(taken)
    while name in taken:
        k += 1
        name = f"{focus}-{k}"
    out["name"] = name
    return out


def _copy(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def defaults() -> dict:
    """The default config as a nested dict, in FIELDS order: schema first, then the tables, editors as a list."""
    out: dict[str, Any] = {"schema": CONFIG_SCHEMA}
    for table in TABLES:
        if table == "editors":
            out["editors"] = [dict(e) for e in EDITORS]
        else:
            out[table] = {f.name: _copy(f.default) for f in table_fields(table)}
    return out


def flag(f: Field) -> str:
    """The field's flag: `--<table>-<name>`, underscores as dashes. Editor fields have none of their own (they take
    `--editor-model`, `--editor-effort` for every editor, and `--set editors.N.KEY=VALUE`)."""
    if f.per_editor:
        return f"--editor-{f.name.replace('_', '-')}"
    return f"--{f.table}-{f.name}".replace("_", "-")


_DURATION = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?\s*$")


def parse_duration(value: Any) -> float:
    """Seconds from a number or text such as 90m, 6h, 1h30m, 45s. ValueError."""
    if isinstance(value, bool):
        raise ValueError(f"not a duration: {value!r}")
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"a duration is not negative: {value!r}")
        return float(value)
    text = str(value).strip()
    try:
        return parse_duration(float(text))
    except ValueError:
        pass
    m = _DURATION.match(text)
    if not text or not m or not any(m.groups()):
        raise ValueError(f"not a duration: {value!r} (give seconds, or text such as 90m, 6h, 1h30m)")
    h, mi, s = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s


_TRUE = ("true", "yes", "on", "1")
_FALSE = ("false", "no", "off", "0")


def parse_value(f: Field, value: Any) -> Any:
    """`value` (from TOML, or a flag's text) as the field's type, checked against its choices and minimum.
    ValueError with a plain reason."""
    t = f.type
    if t == "bool":
        if isinstance(value, bool):
            out: Any = value
        elif isinstance(value, str) and value.strip().lower() in _TRUE + _FALSE:
            out = value.strip().lower() in _TRUE
        else:
            raise ValueError(f"{f.key}: true or false, not {value!r}")
    elif t == "int":
        if isinstance(value, bool):
            raise ValueError(f"{f.key}: a whole number, not {value!r}")
        try:
            out = int(value) if not isinstance(value, float) or value.is_integer() else None
        except (TypeError, ValueError):
            out = None
        if out is None:
            raise ValueError(f"{f.key}: a whole number, not {value!r}")
    elif t == "float":
        if isinstance(value, bool):
            raise ValueError(f"{f.key}: a number, not {value!r}")
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{f.key}: a number, not {value!r}") from None
    elif t == "str":
        if not isinstance(value, str):
            raise ValueError(f"{f.key}: text, not {value!r}")
        out = value
    elif t == "str-list":
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{f.key}: a list of text, not {value!r}")
        out = list(value)
    elif t == "range":
        if isinstance(value, str):
            value = [v.strip() for v in value.strip("[] ").split(",")]
        try:
            lo, hi = (float(v) for v in value)
        except (TypeError, ValueError):
            raise ValueError(f"{f.key}: two numbers [low, high], not {value!r}") from None
        if not lo < hi:
            raise ValueError(f"{f.key}: low must be below high, not {value!r}")
        out = [int(v) if float(v).is_integer() else v for v in (lo, hi)]
    elif t == "duration":
        parse_duration(value)  # raises
        out = value
    else:  # pragma: no cover
        raise ValueError(f"{f.key}: unknown type {t}")
    if f.choices:
        items = out if isinstance(out, list) else [out]
        bad = [v for v in items if v not in f.choices]
        if bad:
            raise ValueError(f"{f.key}: {bad[0]!r} is not one of {', '.join(map(str, f.choices))}")
    if f.minimum is not None and t in ("int", "float") and out < f.minimum:
        raise ValueError(f"{f.key}: at least {f.minimum:g}, not {out!r}")
    return out


def _json_type(f: Field) -> dict:
    t = f.type
    if t == "bool":
        out: dict = {"type": "boolean"}
    elif t == "int":
        out = {"type": "integer"}
    elif t == "float":
        out = {"type": "number"}
    elif t == "str":
        out = {"type": "string"}
    elif t == "str-list":
        out = {"type": "array", "items": {"type": "string"}}
    elif t == "range":
        out = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    else:  # duration
        out = {"type": ["string", "number"]}
    if f.choices:
        if t == "str-list":
            out["items"] = {"type": "string", "enum": list(f.choices)}
        else:
            out["enum"] = list(f.choices)
    if f.minimum is not None:
        out["minimum"] = f.minimum
    return out


def json_schema() -> dict:
    """The config's JSON Schema (draft 2020-12), with each field's default, help, flag and source; unknown keys are
    allowed, as the loader keeps them."""
    props: dict[str, Any] = {"schema": {"const": CONFIG_SCHEMA}}
    for table in TABLES:
        sub = {}
        for f in table_fields(table):
            sub[f.name] = {**_json_type(f), "default": _copy(f.default), "description": f.help,
                           "x-flag": flag(f), "x-source": f.source, **({"x-note": f.note} if f.note else {})}
        obj = {"type": "object", "properties": sub, "additionalProperties": True}
        if table == "editors":
            props["editors"] = {"type": "array", "items": obj, "default": [dict(e) for e in EDITORS],
                                "description": "the editors, in ownership order", "x-flag": "--editors"}
        else:
            props[table] = obj
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": CONFIG_SCHEMA,
            "title": "loopmath harness config", "type": "object", "properties": props,
            "additionalProperties": True}
