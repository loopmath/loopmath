"""Task types, features and the grouping chain (design/0.1/03-interfaces.md, section 3; Q6).

Types are fixed. Subtypes are the user's own (config `subtypes`). Features are a short
built-in list plus the ones the user declares in config `[features.<key>]` (a
`FeatureSet`); unknown is always allowed. `horizon_s`, the run's time budget in seconds,
is a feature key the fit models as two fixed terms, never as a feature level (spec 04
section 1).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .types import TASK_TYPE_IDS, Task

TASK_TYPES: dict[str, str] = {
    "bug_fix": "Fix a defect in existing behaviour.",
    "feature": "Add new behaviour.",
    "refactor": "Change structure without changing behaviour.",
    "tests": "Add or repair tests.",
    "docs": "Write or change documentation.",
    "research": "Investigate, prototype or answer a question; the output is a finding.",
    "infra": "Build, CI, deployment, tooling or configuration.",
    "data": "Data pipelines, migrations, analyses or notebooks.",
}
assert tuple(TASK_TYPES) == TASK_TYPE_IDS


FEATURE_KINDS = ("category", "bool", "number")
FEATURE_FILLS = ("labeller", "orchestrator")
MAX_CUSTOM_FEATURES = 16
MAX_CATEGORY_VALUES = 12
MIN_TASKS_DEFAULT = 2  # distinct tasks a feature value needs before it enters the fit (spec 04 section 1)
HORIZON_KEY = "horizon_s"
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_VALUE_RE = re.compile(r"^[a-z0-9][a-z0-9_.+-]{0,31}$")
_TRUE = ("true", "yes", "y", "1")
_FALSE = ("false", "no", "n", "0")


class FeatureConfigError(ValueError):
    """A bad `[features]` declaration in config.toml; the message names the key."""


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    values: tuple[str, ...]
    description: str
    default: str = "unknown"
    kind: str = "category"  # category | bool | number
    edges: tuple[float, ...] = ()  # number: ascending bucket edges
    labels: tuple[str, ...] = ()  # number: one label per bucket (len(edges) + 1)
    fill: str = "labeller"  # labeller | orchestrator (--feature, a task file, run start, a converter)
    types: tuple[str, ...] = ()  # the task types it applies to; empty: every type
    repos: tuple[str, ...] = ()  # the repos it applies to; empty: every repo
    builtin: bool = True

    def label_values(self) -> tuple[str, ...]:
        """The values a normalized value can take besides the default; empty means open."""
        if self.kind == "bool":
            return ("yes", "no")
        if self.kind == "number":
            return self.labels
        return self.values

    def normalize(self, value: Any) -> str:
        """Lowercase; a number into its bucket; anything outside a closed set is the default."""
        if isinstance(value, bool):
            v = "yes" if value else "no"
        else:
            v = str(value).strip().lower()
        if self.key == "size" and v.isdigit():
            return size_bucket(int(v))
        if self.key == "touches" and v.isdigit():
            return touches_bucket(int(v))
        if not v:
            return self.default
        if self.kind == "bool":
            return "yes" if v in _TRUE else ("no" if v in _FALSE else self.default)
        if self.kind == "number":
            if v in self.labels:
                return v
            try:
                x = float(v)
            except ValueError:
                return self.default
            if not math.isfinite(x):
                return self.default
            for edge, label in zip(self.edges, self.labels):
                if x < edge:
                    return label
            return self.labels[-1]
        if self.values and v not in self.values:
            return self.default
        return v

    def stored(self, value: Any) -> str:
        """The value a document keeps: a built-in as `normalize` gives it (its buckets never change);
        a declared key's value as given, lowercased, so a changed declaration applies at the next fit."""
        if self.builtin:
            return self.normalize(value)
        if isinstance(value, bool):
            return "yes" if value else "no"
        return str(value).strip().lower() or self.default

    def applies(self, task_type: str | None, repo: str | None = None) -> bool:
        """Whether the key is in scope for a task (scopes only steer the labeller and the notes)."""
        if self.types and task_type not in self.types:
            return False
        return not (self.repos and repo is not None and repo not in self.repos)

    def scope_text(self) -> str:
        parts = []
        if self.types:
            parts.append(", ".join(self.types) + " tasks")
        if self.repos:
            parts.append("repo " + ", ".join(self.repos))
        return " and ".join(parts)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"key": self.key, "kind": self.kind, "values": list(self.label_values()),
                             "description": self.description, "fill": self.fill,
                             "types": list(self.types), "repos": list(self.repos), "builtin": self.builtin}
        if self.kind == "number":
            d["edges"] = list(self.edges)
            d["labels"] = list(self.labels)
        return d


FEATURES: dict[str, FeatureSpec] = {
    "size": FeatureSpec("size", ("xs", "s", "m", "l", "xl"),
                        "Expected changed lines: under 20, 100, 400, 1500, above."),
    "lang": FeatureSpec("lang", (), "Dominant language, lowercase (python, typescript, rust, ...)."),
    "has_tests": FeatureSpec("has_tests", ("yes", "no"), "The touched code has tests that can judge the change."),
    "spec_clarity": FeatureSpec("spec_clarity", ("clear", "partial", "vague"), "How fully the task says what done means."),
    "needs_design": FeatureSpec("needs_design", ("yes", "no"), "A design choice must be made before coding."),
    "touches": FeatureSpec("touches", ("one", "few", "many"), "Files touched: 1, 2 to 5, more."),
}

SIZE_EDGES = ((20, "xs"), (100, "s"), (400, "m"), (1500, "l"))


def size_bucket(changed_lines: int) -> str:
    for edge, name in SIZE_EDGES:
        if changed_lines < edge:
            return name
    return "xl"


def touches_bucket(files: int) -> str:
    return "one" if files <= 1 else ("few" if files <= 5 else "many")


_HORIZON_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|sec|secs|m|min|mins|h|hr|hrs|hours?|d|days?)?$")
_HORIZON_UNIT = {"s": 1, "sec": 1, "secs": 1, "m": 60, "min": 60, "mins": 60, "h": 3600, "hr": 3600, "hrs": 3600,
                 "hour": 3600, "hours": 3600, "d": 86400, "day": 86400, "days": 86400}


def parse_horizon(value: Any) -> float | None:
    """A run's time budget in seconds: `8h`, `90m`, `2.5h`, `7200` (seconds) or a number;
    `none`, `open` or empty is open-ended (None). Raises ValueError on anything else."""
    if value is None or isinstance(value, bool):
        if isinstance(value, bool):
            raise ValueError(f"a horizon is a duration such as 8h, 90m or 7200; got {value!r}")
        return None
    if isinstance(value, (int, float)):
        secs = float(value)
    else:
        text = str(value).strip().lower()
        if text in ("", "none", "open", "open-ended", "0"):
            return None
        m = _HORIZON_RE.match(text)
        if not m:
            raise ValueError(f"a horizon is a duration such as 8h, 90m or 7200; got {value!r}")
        secs = float(m.group(1)) * _HORIZON_UNIT.get(m.group(2) or "s", 1)
    if not math.isfinite(secs) or secs < 0:
        raise ValueError(f"a horizon is a positive duration; got {value!r}")
    return secs or None


def horizon_text(secs: float | None) -> str:
    """`2 h`, `90 min`, `45 s`, or `open-ended`."""
    if not secs:
        return "open-ended"
    if secs % 3600 == 0:
        return f"{int(secs // 3600)} h"
    if secs >= 3600:
        return f"{secs / 3600:.1f} h"
    if secs % 60 == 0:
        return f"{int(secs // 60)} min"
    return f"{secs:g} s"


def _seconds_str(secs: float) -> str:
    return str(int(secs)) if float(secs).is_integer() else f"{secs:g}"


def horizon_value(value: Any) -> str | None:
    """`horizon_s` as a task keeps it: seconds, or `none` for a given open-ended horizon (it then wins
    over an inherited one). None, to drop it, when empty or unreadable."""
    if value is None or str(value).strip() == "":
        return None
    try:
        secs = parse_horizon(value)
    except ValueError:
        return None
    return _seconds_str(secs) if secs else "none"


class FeatureSet:
    """The feature keys a store models: the built-ins plus config `[features.<key>]`.

    One place validates declarations, normalizes values (numbers into buckets, bool
    spellings, closed sets) and says which keys the labeller fills. The fit records its
    set in `meta.json`, so predictions from that fit read features the same way.
    """

    def __init__(self, custom: Iterable[FeatureSpec] = (), *, min_tasks: int = MIN_TASKS_DEFAULT):
        self.custom: tuple[FeatureSpec, ...] = tuple(custom)
        self.specs: dict[str, FeatureSpec] = dict(FEATURES)
        for spec in self.custom:
            self.specs[spec.key] = spec
        self.min_tasks = int(min_tasks)

    # ---------------------------------------------------------------- declarations

    @classmethod
    def from_config(cls, table: Any) -> "FeatureSet":
        """The set a config `[features]` table declares. Raises FeatureConfigError."""
        if table is None:
            return cls()
        if not isinstance(table, Mapping):
            raise FeatureConfigError("[features] must be a table")
        min_tasks = table.get("min_tasks", MIN_TASKS_DEFAULT)
        if isinstance(min_tasks, bool) or not isinstance(min_tasks, int) or min_tasks < 1:
            raise FeatureConfigError(f"features.min_tasks takes a whole number of 1 or more; got {min_tasks!r}")
        custom = [_spec_from_table(key, value) for key, value in table.items() if key != "min_tasks"]
        if len(custom) > MAX_CUSTOM_FEATURES:
            raise FeatureConfigError(f"at most {MAX_CUSTOM_FEATURES} custom features; got {len(custom)}")
        return cls(custom, min_tasks=min_tasks)

    @classmethod
    def from_json(cls, d: Any) -> "FeatureSet":
        """The set a fit recorded (`meta.json` `features.declared`); the built-ins when absent or bad."""
        if not isinstance(d, Mapping):
            return cls()
        table: dict[str, Any] = {"min_tasks": d.get("min_tasks", MIN_TASKS_DEFAULT)}
        for item in d.get("custom") or []:
            if isinstance(item, Mapping) and item.get("key"):
                spec = {k: item[k] for k in ("kind", "fill", "types", "repos", "description") if k in item}
                if item.get("kind") == "number":
                    spec.update(edges=item.get("edges") or [], labels=item.get("labels") or [])
                elif item.get("kind", "category") == "category":
                    spec["values"] = item.get("values") or []
                table[str(item["key"])] = spec
        try:
            return cls.from_config(table)
        except FeatureConfigError:
            return cls()

    def to_json(self) -> dict[str, Any]:
        return {"min_tasks": self.min_tasks, "custom": [s.to_json() for s in self.custom]}

    def digest(self) -> str:
        text = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ---------------------------------------------------------------- values

    def normalize(self, raw: Mapping[str, Any] | None) -> dict[str, str]:
        """The model's reading: lowercase and bucket declared keys; `horizon_s` as `horizon_value`
        gives it; any other key kept as `extra:<key>`, stored but never modelled."""
        return self._read(raw, keep=False)

    def store(self, raw: Mapping[str, Any] | None) -> dict[str, str]:
        """The values a run document keeps: as `normalize`, except that a declared key keeps its raw
        value (`input_items: 50`, not its bucket), so the fit buckets it with its own declaration."""
        return self._read(raw, keep=True)

    def _read(self, raw: Mapping[str, Any] | None, *, keep: bool) -> dict[str, str]:
        out: dict[str, str] = {}
        later: list[tuple[str, Any]] = []
        for key, value in (raw or {}).items():
            k = str(key).strip().lower()
            base = k[len("extra:"):] if k.startswith("extra:") else None
            if base is not None and (base in self.specs or base == HORIZON_KEY):
                later.append((base, value))  # stored before the key was declared: read it as declared
                continue
            if k == HORIZON_KEY:
                h = horizon_value(value)
                if h:
                    out[k] = h
                continue
            spec = self.specs.get(k)
            if spec is not None:
                out[k] = spec.stored(value) if keep else spec.normalize(value)
            elif k.startswith("extra:"):
                out[k] = str(value).strip().lower()
            else:
                out[f"extra:{k}"] = str(value).strip().lower()
        for base, value in later:
            if base in out:
                continue
            if base == HORIZON_KEY:
                h = horizon_value(value)
                if h:
                    out[base] = h
            else:
                spec = self.specs[base]
                out[base] = spec.stored(value) if keep else spec.normalize(value)
        return out

    def model_features(self, raw: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
        """Declared keys with a known value, sorted: the task's feature levels."""
        out = []
        for key, value in sorted(self.normalize(raw).items()):
            spec = self.specs.get(key)
            if spec is None or not value or value == spec.default:
                continue
            out.append((key, value))
        return tuple(out)

    def notes(self, raw: Mapping[str, Any] | None, task_type: str | None = None,
              repo: str | None = None) -> list[str]:
        """Plain notes on given features: undeclared keys, values outside a closed set, keys out of scope."""
        notes = []
        for key, value in (raw or {}).items():
            k = str(key).strip().lower()
            if k == HORIZON_KEY:
                continue
            spec = self.specs.get(k)
            if spec is None:
                notes.append(f"{k} is not a declared feature; stored, not used by the model")
            elif spec.normalize(value) == spec.default and str(value).strip().lower() != spec.default:
                allowed = "|".join(spec.label_values())
                notes.append(f"{k}={value} is not one of {allowed}; read as unknown")
            elif not spec.applies(task_type, repo):
                notes.append(f"{k} is declared for {spec.scope_text()}")
        return notes

    def labeller_specs(self) -> list[FeatureSpec]:
        return [s for s in self.specs.values() if s.fill == "labeller"]


def _tokens(key: str, name: str, raw: Any, *, pattern: re.Pattern = _VALUE_RE) -> tuple[str, ...]:
    if isinstance(raw, str):
        raw = [v for v in raw.split(",") if v.strip()]
    if not isinstance(raw, (list, tuple)):
        raise FeatureConfigError(f"features.{key}.{name} takes a list")
    out = []
    for v in raw:
        t = str(v).strip().lower()
        if not pattern.match(t):
            raise FeatureConfigError(f"features.{key}.{name}: {v!r} is not a short lowercase word")
        if t in out:
            raise FeatureConfigError(f"features.{key}.{name}: {t!r} is listed twice")
        out.append(t)
    return tuple(out)


_SPEC_FIELDS = ("kind", "values", "edges", "labels", "fill", "types", "repos", "description")


def _spec_from_table(key: Any, table: Any) -> FeatureSpec:
    key = str(key)
    if not _KEY_RE.match(key) or key.startswith("extra"):
        raise FeatureConfigError(f"feature name {key!r}: use lowercase letters, digits and _, "
                                 "up to 32 characters, not starting with extra")
    if key == HORIZON_KEY:
        raise FeatureConfigError(f"{HORIZON_KEY} is built in (the run horizon); it cannot be declared")
    if key in FEATURES:
        raise FeatureConfigError(f"{key} is a built-in feature; it cannot be declared again")
    if not isinstance(table, Mapping):
        raise FeatureConfigError(f"features.{key} must be a table")
    extra = sorted(set(table) - set(_SPEC_FIELDS))
    if extra:
        raise FeatureConfigError(f"features.{key}: unknown field {extra[0]!r} (known: {', '.join(_SPEC_FIELDS)})")
    kind = str(table.get("kind", "category")).strip().lower()
    if kind not in FEATURE_KINDS:
        raise FeatureConfigError(f"features.{key}.kind is one of {', '.join(FEATURE_KINDS)}; got {kind!r}")
    fill = str(table.get("fill", "orchestrator")).strip().lower()
    if fill not in FEATURE_FILLS:
        raise FeatureConfigError(f"features.{key}.fill is one of {', '.join(FEATURE_FILLS)}; got {fill!r}")
    types = _tokens(key, "types", table.get("types", ()), pattern=re.compile(r"^[a-z_]+$"))
    bad = [t for t in types if t not in TASK_TYPES]
    if bad:
        raise FeatureConfigError(f"features.{key}.types: {bad[0]!r} is not a task type ({', '.join(TASK_TYPES)})")
    repos_raw = table.get("repos", ())
    if isinstance(repos_raw, str):
        repos_raw = [r for r in repos_raw.split(",") if r.strip()]
    if not isinstance(repos_raw, (list, tuple)) or not all(str(r).strip() for r in repos_raw):
        raise FeatureConfigError(f"features.{key}.repos takes a list of repo names")
    repos = tuple(str(r).strip() for r in repos_raw)
    description = str(table.get("description", "")).strip()
    values: tuple[str, ...] = ()
    edges: tuple[float, ...] = ()
    labels: tuple[str, ...] = ()
    if kind == "category":
        values = _tokens(key, "values", table.get("values", ()))
        if len(values) > MAX_CATEGORY_VALUES:
            raise FeatureConfigError(f"features.{key}.values: at most {MAX_CATEGORY_VALUES} values")
        if "unknown" in values:
            raise FeatureConfigError(f"features.{key}.values: unknown is always allowed; do not list it")
    elif "values" in table:
        raise FeatureConfigError(f"features.{key}: values are for kind category only")
    if kind == "number":
        raw_edges = table.get("edges")
        if isinstance(raw_edges, str):
            raw_edges = [e for e in raw_edges.split(",") if e.strip()]
        try:
            edges = tuple(float(e) for e in raw_edges or ())
        except (TypeError, ValueError):
            raise FeatureConfigError(f"features.{key}.edges takes a list of numbers") from None
        if not edges or not all(math.isfinite(e) for e in edges):
            raise FeatureConfigError(f"features.{key}.edges takes at least one finite number")
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise FeatureConfigError(f"features.{key}.edges must be ascending")
        labels = _tokens(key, "labels", table.get("labels", ()))
        if len(labels) != len(edges) + 1:
            raise FeatureConfigError(f"features.{key}.labels needs {len(edges) + 1} labels for {len(edges)} edges; "
                                     f"got {len(labels)}")
        if "unknown" in labels:
            raise FeatureConfigError(f"features.{key}.labels: unknown is reserved")
    elif "edges" in table or "labels" in table:
        raise FeatureConfigError(f"features.{key}: edges and labels are for kind number only")
    return FeatureSpec(key, values, description, kind=kind, edges=edges, labels=labels, fill=fill,
                       types=types, repos=repos, builtin=False)


BUILTIN_FEATURES = FeatureSet()


def features_from_config(table: Any) -> FeatureSet:
    """`FeatureSet.from_config`, or the built-ins with a warning on stderr when the declaration is bad."""
    try:
        return FeatureSet.from_config(table)
    except FeatureConfigError as exc:
        print(f"warning: config [features]: {exc}; using the built-in features", file=sys.stderr)
        return BUILTIN_FEATURES


def normalize_features(raw: Mapping[str, Any] | None, features: FeatureSet | None = None) -> dict[str, str]:
    """Lowercase, bucket numbers, keep unknown keys under 'extra:<key>' (see FeatureSet.normalize)."""
    return (features or BUILTIN_FEATURES).normalize(raw)


def group_chain(task: Task) -> list[tuple[str, str]]:
    """[("org", o), ("type", t), ("repo", r), ("subtype", s), ("task", id)]; missing levels skipped."""
    chain: list[tuple[str, str]] = []
    if task.org:
        chain.append(("org", task.org))
    chain.append(("type", task.type))
    chain.append(("repo", task.repo))
    if task.subtype:
        chain.append(("subtype", task.subtype))
    chain.append(("task", task.id))
    return chain


def task_types_payload(subtypes: list[str] | None = None, features: FeatureSet | None = None) -> dict:
    """The `loopmath task-types --json` object (schema loopmath.task-types/1).

    Each feature carries its kind, who fills it and its scope (additive fields); `horizon`
    names the run horizon key, which is not a feature level.
    """
    fs = features or BUILTIN_FEATURES
    return {
        "schema": "loopmath.task-types/1",
        "types": [{"id": k, "title": k.replace("_", " "), "description": v} for k, v in TASK_TYPES.items()],
        "features": [f.to_json() for f in fs.specs.values()],
        "horizon": {"key": HORIZON_KEY, "unit": "s", "flag": "--horizon",
                    "description": "The run's wall-clock time budget in seconds; absent when open-ended."},
        "subtypes": list(subtypes or []),
    }


# ---------------------------------------------------------------- labelling hooks (lane 03)
# One definition of a task label, shared by `loopmath onboard` (which labels past
# session groups through the user's own CLI) and by any orchestrator that wants
# to ask a model for a type: the instructions, the JSON Schema of the answer, the
# validator every answer passes through, and a keyword guess for `--labeler none`.

LABEL_VERSION = "label/1"
UNKNOWN_TYPE = "unknown"  # a labeller's way to say "cannot tell"; onboard stores such a group as type unknown
TITLE_MAX = 80


def _feature_values(spec: FeatureSpec) -> list[str]:
    values = spec.label_values()
    return [*values, spec.default] if values else []


def label_schema(subtypes: list[str] | None = None, features: FeatureSet | None = None) -> dict:
    """JSON Schema of one labelling answer: {"labels": [{id, type, subtype, features, confidence, title}]}.

    Every object is closed and every property required, so the same schema works as
    `claude -p --json-schema` and as `codex exec --output-schema` (strict mode). The
    features are the built-ins plus the declared keys the labeller fills.
    """
    fs = features or BUILTIN_FEATURES
    features = {}
    for spec in fs.labeller_specs():
        values = _feature_values(spec)
        features[spec.key] = {"type": "string", "enum": values} if values else {"type": "string"}
    subtype: dict = {"type": ["string", "null"]}
    if subtypes:
        subtype["enum"] = [*subtypes, None]
    else:
        subtype = {"type": "null"}
    item = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string", "enum": [*TASK_TYPES, UNKNOWN_TYPE]},
            "subtype": subtype,
            "features": {"type": "object", "properties": features,
                         "required": list(features), "additionalProperties": False},
            "confidence": {"type": "number"},
            "title": {"type": "string"},
        },
        "required": ["id", "type", "subtype", "features", "confidence", "title"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"labels": {"type": "array", "items": item}},
        "required": ["labels"],
        "additionalProperties": False,
    }


def label_instructions(subtypes: list[str] | None = None, features: FeatureSet | None = None) -> str:
    """The rules a labeller follows, in plain text (the task list follows it in the prompt)."""
    lines = [
        "You label past coding-agent work. Each item below is one session group: a root agent session",
        "plus the sessions it started. Assign each item a task type, an optional subtype, features,",
        "a confidence and a short title. The labels train a cost and success model, so prefer",
        f"'{UNKNOWN_TYPE}' over a guess.",
        "",
        "Task types (pick exactly one):",
    ]
    lines += [f"- {k}: {v}" for k, v in TASK_TYPES.items()]
    lines.append(f"- {UNKNOWN_TYPE}: the summary does not say what the work was, or it is not software work.")
    lines.append("")
    if subtypes:
        lines.append("Subtype: one of " + ", ".join(subtypes) + ", or null when none fits.")
    else:
        lines.append("Subtype: always null.")
    lines.append("")
    lines.append("Features (use 'unknown' when the summary does not show it):")
    for spec in (features or BUILTIN_FEATURES).labeller_specs():
        values = "|".join(_feature_values(spec)) or ("lowercase language name, or unknown" if spec.key == "lang"
                                                    else "a short lowercase word, or unknown")
        scope = f" Only for {spec.scope_text()}; else unknown." if spec.types or spec.repos else ""
        lines.append(f"- {spec.key} ({values}): {spec.description}{scope}")
    lines += [
        "",
        "confidence: a number from 0 to 1, how sure you are of the type.",
        f"title: at most {TITLE_MAX} characters, a neutral description of the task. No names, paths,",
        "secrets or quotes from the prompt.",
        "",
        "Do not use any tool. Answer with exactly one JSON object and nothing else:",
        '{"labels": [{"id": ..., "type": ..., "subtype": ..., "features": {...}, "confidence": ..., "title": ...}]}',
        "with one entry for every item id, in any order.",
    ]
    return "\n".join(lines)


def validate_label(raw: object, subtypes: list[str] | None = None, features: FeatureSet | None = None,
                   repo: str | None = None) -> tuple[dict | None, list[str]]:
    """Check one labeller answer item. Returns (label, problems).

    `label` is {type, subtype, features, confidence, title}, or None when the item has
    no usable type (missing, invalid, or `unknown`). Features go through the FeatureSet;
    values the labeller left unknown, keys it does not fill and keys outside their type
    or repo scope are dropped, so a stored task carries only what was actually labelled.
    A subtype outside `subtypes` is dropped with a problem noted; the type is kept.
    """
    problems: list[str] = []
    if not isinstance(raw, dict):
        return None, ["not an object"]
    kind = str(raw.get("type") or "").strip().lower()
    if kind == UNKNOWN_TYPE:
        return None, ["type unknown"]
    if kind not in TASK_TYPES:
        return None, [f"invalid type {kind[:40]!r}"]
    subtype = raw.get("subtype")
    if subtype is not None:
        subtype = str(subtype).strip()
        if not subtype or subtype not in (subtypes or []):
            problems.append(f"subtype {subtype[:40]!r} not in config subtypes")
            subtype = None
    features_set = features or BUILTIN_FEATURES
    labelled: dict[str, str] = {}
    raw_features = raw.get("features")
    if isinstance(raw_features, dict):
        fs = features_set
        given = {str(k): str(v) for k, v in raw_features.items()}
        kept = fs.store(given)  # a declared number keeps the labeller's number, not its bucket
        for key, value in fs.normalize(given).items():
            spec = fs.specs.get(key)
            if (spec is not None and spec.fill == "labeller" and value and value != spec.default
                    and spec.applies(kind, repo)):
                labelled[key] = kept[key]
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
        problems.append("confidence missing")
    if confidence != confidence:  # NaN
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    title = " ".join(str(raw.get("title") or "").split())[:TITLE_MAX]
    return {"type": kind, "subtype": subtype, "features": labelled,
            "confidence": round(confidence, 3), "title": title}, problems


# Keyword guess for `onboard --labeler none`: the first matching rule wins, in this
# order (a "fix the failing test" prompt is a bug fix before it is tests work).
_GUESS_RULES: tuple[tuple[str, str], ...] = (
    ("bug_fix", r"\b(fix(es|ed|ing)?|bugs?|broken|crash(es|ing)?|regressions?|traceback|exception|doesn'?t work|not working)\b"),
    ("tests", r"\b(unit tests?|test coverage|(add|write|extend)\s+(more\s+)?tests?)\b"),
    ("docs", r"\b(readme|documentation|docstrings?|changelog|docs)\b"),
    ("refactor", r"\b(refactor(ing)?|restructure|clean ?up|simplify|rename)\b"),
    ("infra", r"\b(ci|deploy(ment)?|docker(file)?|kubernetes|terraform|launchd|github actions|makefile|packaging|pyproject)\b"),
    ("data", r"\b(dataset|csv|sql|etl|notebook|dataframe|pandas|schema migration)\b"),
    ("research", r"\b(investigate|research|why does|how does|compare|evaluate|find out|look into|explain)\b"),
    ("feature", r"\b(add|implement|build|create|support|introduce)\b"),
)
GUESS_CONFIDENCE = 0.4


def guess_type(text: str | None) -> tuple[str | None, float]:
    """(type, confidence) from keywords in a task text, or (None, 0.0) when nothing matches."""
    if not text:
        return None, 0.0
    low = text.lower()
    for kind, pattern in _GUESS_RULES:
        if re.search(pattern, low):
            return kind, GUESS_CONFIDENCE
    return None, 0.0
