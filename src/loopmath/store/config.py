"""config.toml: read, write, defaults (design/0.1/02-commands.md, section 3).

Keys are dotted paths into the TOML tables: `org`, `acceptance_rule`,
`rules.<name>`, `goal`, `rescue.kind`, `rescue.person_usd_per_hour`,
`rescue.hours`, `rescue.decay`, `rescue.max_attempts`, `rescue.min_chance`, `models.allowed`, `harnesses`, `subtypes`, `labeler`,
`benchmark_prior_weight` (unset by default: each benchmark's own `weight` in
`benchmarks.toml`; a number is one weight for every benchmark, 0 none), `explore.default_pick`, `explore.auto_payback_runs`,
`referee.model`, `budget.usd`, `budget.period`, `outcome.q.<tier>`,
`onboard.labeler` (D39: `claude:<model>`, `codex:<model>`,
`command:<cmd>` or `none`, chosen by the user, never defaulted), and the
usual workflows `usual.<type>."<repo>"` / `usual.<type>."*"` with
`usual_meta.<type>`, `research.<name>` strings kept
verbatim (`research.sweep_dir`, `research.e0_corpus`), and the
declared task features `features.<key>` with `features.min_tasks`
(taskmodel.FeatureSet; spec 04 section 1), and `belief.success_head` (unset or
`eb`, 0.3.1's success head; `sampled` opts in to the success head by NUTS over
its effects and scales, belief.integrated and belief.sampled).

Reading uses `tomllib` (Python 3.11+), else a reader for the
subset this module writes. The writer emits plain tables and JSON-compatible
values only, so that subset reader can load anything the store wrote.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from ..taskmodel import FeatureConfigError, FeatureSet, features_from_config
from ..types import DEFAULT_RULE, AcceptanceRule, ScoreTarget
from .lock import atomic_write_text

_MISSING = object()

DEFAULTS: dict[str, Any] = {
    "acceptance_rule": "tests",
    "goal": "default",
    "rescue": {"kind": "retry", "hours": 1, "decay": 0.5, "max_attempts": 3, "min_chance": 0.7},
    "models": {"allowed": []},
    "harnesses": ["claude-code", "codex"],
    "subtypes": [],
    "explore": {"default_pick": "best_value"},
    "budget": {"period": "month"},
    "outcome": {"q": {"verified": 0.98, "reported": 0.95, "heuristic": 0.8, "asserted": 0.7}},
    "harness": {"record": "none"},  # `run record` and `run finish` without --harness (`spec/harness.md`)
}

CHOICES: dict[str, tuple[str, ...]] = {
    "goal": ("default", "p50", "p70", "p80", "p90", "p95", "p99"),
    "rescue.kind": ("retry", "redo_usual", "person", "none"),
    "explore.default_pick": ("best_value", "max_gain"),
    "budget.period": ("week", "month", "none"),
    "harness.record": ("none", "auto"),
    # unset or `eb`: 0.3.1's success head; `sampled` opts in (belief.integrated, belief.sampled)
    "belief.success_head": ("eb", "sampled"),
}
LABELER_KINDS = ("claude", "codex", "command")
LABELER_KEYS = ("onboard.labeler", "labeler")  # `labeler` is the spec 02 name; D39 stores `onboard.labeler`
LIST_KEYS = ("models.allowed", "harnesses", "subtypes")
NUMBER_KEYS = ("rescue.person_usd_per_hour", "rescue.hours", "benchmark_prior_weight",
               "explore.auto_payback_runs", "budget.usd", "plan.time_budget_s")
POSITIVE_KEYS = ("plan.time_budget_s",)
NONNEGATIVE_KEYS = ("benchmark_prior_weight",)  # 0 turns the benchmark factors off
COUNT_KEYS = ("plan.exact_picks",)  # whole numbers >= 0; the recommender owns the defaults of the plan keys
RESCUE_KEYS = ("rescue.decay", "rescue.max_attempts", "rescue.min_chance")  # `check_rescue`
KNOWN_ROOTS = ("org", "acceptance_rule", "rules", "goal", "rescue", "models", "harnesses", "subtypes", "belief",
               "labeler", "benchmark_prior_weight", "explore", "referee", "budget", "outcome", "usual",
               "usual_meta", "research", "onboard", "plan", "efforts", "features", "harness")
STRING_ROOTS = ("research",)  # values stored as given, never parsed as JSON
# Every key loopmath reads; a `<...>` part is a name of your choice. `config get` lists the unset ones and
# `config set` refuses a key that is none of these nor a table above one (such as `rescue`).
KEYS = ("org", "acceptance_rule", "rules.<name>", "goal", "rescue.kind", "rescue.person_usd_per_hour",
        "rescue.hours", "rescue.decay", "rescue.max_attempts", "rescue.min_chance", "models.allowed", "harnesses",
        "subtypes", "benchmark_prior_weight", "belief.success_head", "explore.default_pick",
        "explore.auto_payback_runs", "referee.model",
        "budget.usd", "budget.period", "outcome.q.<tier>", "onboard.labeler", "onboard.skip",
        "usual.<type>.<repo>", "usual_meta.<type>.<repo>", "research.sweep_dir", "research.e0_corpus",
        "features.min_tasks", "features.<key>.<field>", "plan.time_budget_s", "plan.exact_picks",
        "efforts.<model>", "harness.record")
UNLISTED_KEYS = ("labeler",)  # the spec 02 name of onboard.labeler: `config set` still takes it


class ConfigError(ValueError):
    """A bad key or value (exit code 1)."""


# What `Config.load` raises for a config.toml that cannot be parsed or read (a directory, no permission, not
# UTF-8). A reader that goes on with the built-in values catches these; `read_error` says why in one line.
READ_ERRORS = (ConfigError, OSError, UnicodeDecodeError)


def read_error(path: Path, exc: Exception) -> str:
    """`path: why` for one of READ_ERRORS (a ConfigError names its file already)."""
    return str(exc) if isinstance(exc, ConfigError) else f"{path}: {getattr(exc, 'strerror', None) or exc}"


# ---------------------------------------------------------------- keys
def split_key(key: str) -> list[str]:
    """`a.b."c.d"` to ["a", "b", "c.d"]. Under `usual.<type>.` and `usual_meta.<type>.` the rest is one repo key."""
    parts: list[str] = []
    buf = ""
    quoted = False
    i = 0
    while i < len(key):
        ch = key[i]
        if ch == '"':
            quoted = not quoted
        elif ch == "." and not quoted:
            parts.append(buf)
            buf = ""
            if len(parts) == 2 and parts[0] in ("usual", "usual_meta"):
                rest = key[i + 1:]
                if len(rest) >= 2 and rest[0] == '"' and rest[-1] == '"':
                    rest = rest[1:-1]
                parts.append(rest)
                return [p for p in parts]
        else:
            buf += ch
        i += 1
    if quoted:
        raise ConfigError(f"unbalanced quote in key {key!r}")
    parts.append(buf)
    if any(p == "" for p in parts):
        raise ConfigError(f"empty part in key {key!r}")
    return parts


def _matches(parts: list[str], pattern: str) -> bool:
    pat = pattern.split(".")
    return len(parts) <= len(pat) and all(p.startswith("<") or p == q for p, q in zip(pat, parts))


def known_key(key: str) -> bool:
    """Whether loopmath reads `key`, or `key` is a table of keys it reads (KEYS)."""
    parts = split_key(key)
    return ".".join(parts) in UNLISTED_KEYS or any(_matches(parts, k) for k in KEYS)


def unset_keys(data: dict[str, Any]) -> list[str]:
    """The KEYS with no value in `data` (a merged config). A `<...>` key is set when its table holds a name
    that is not one of the fixed keys beside it (`features.<key>` beside `features.min_tasks`)."""
    out = []
    for key in KEYS:
        pat = key.split(".")
        fixed = [p for p in pat if not p.startswith("<")] if "<" not in key else pat[:pat.index(next(
            p for p in pat if p.startswith("<")))]
        node: Any = data
        for p in fixed:
            node = node.get(p, _MISSING) if isinstance(node, dict) else _MISSING
        if len(fixed) < len(pat):
            beside = {k.split(".")[len(fixed)] for k in KEYS
                      if k.split(".")[:len(fixed)] == fixed and len(k.split(".")) > len(fixed)} - {pat[len(fixed)]}
            is_set = isinstance(node, dict) and any(name not in beside for name in node)
        else:
            is_set = node is not _MISSING and node is not None
        if not is_set:
            out.append(key)
    return out


# ---------------------------------------------------------------- rules
_SCORE_RE = re.compile(r"^\s*([A-Za-z_][\w:.-]*)\s*(>=|<=)\s*(-?[0-9.eE+-]+)\s*$")


def parse_rule(text: str, *, name: str | None = None) -> AcceptanceRule:
    """A rule from its short form.

    `tests`, `tests+referee` (verdicts that must pass), `merged` (the `merge`
    verdict), `heldout_perf>=2400` or `runtime_s<=200` (a score target), and
    mixes such as `tests+runtime_s<=200`.
    """
    text = (text or "").strip()
    if not text:
        raise ConfigError("empty acceptance rule")
    requires: list[str] = []
    score: ScoreTarget | None = None
    for part in text.split("+"):
        part = part.strip()
        if not part:
            raise ConfigError(f"bad acceptance rule {text!r}")
        m = _SCORE_RE.match(part)
        if m:
            if score is not None:
                raise ConfigError(f"acceptance rule {text!r} has more than one score target")
            try:
                target = float(m.group(3))
            except ValueError:
                raise ConfigError(f"bad score target in {part!r}") from None
            score = ScoreTarget(name=m.group(1), target=target, better="higher" if m.group(2) == ">=" else "lower")
        elif re.fullmatch(r"[A-Za-z_][\w:.-]*", part):
            requires.append("merge" if part == "merged" else part)
        else:
            raise ConfigError(f"bad acceptance rule part {part!r} in {text!r}")
    if text == DEFAULT_RULE.name:
        return DEFAULT_RULE if name in (None, DEFAULT_RULE.name) else AcceptanceRule(
            name=name, definition=DEFAULT_RULE.definition, requires=DEFAULT_RULE.requires)
    return AcceptanceRule(name=name or text, definition=_definition(requires, score),
                          requires=tuple(requires), score=score)


def _definition(requires: list[str], score: ScoreTarget | None) -> str:
    bits = []
    if requires:
        bits.append(" and ".join(requires) + (" passes" if len(requires) == 1 else " pass"))
    if score is not None:
        word = "at least" if score.better == "higher" else "at most"
        bits.append(f"{score.name} {word} {score.target:g}")
    return ", and ".join(bits)


def rule_from_table(name: str, table: dict[str, Any]) -> AcceptanceRule:
    data = dict(table)
    data.setdefault("name", name)
    if "definition" not in data:
        data["definition"] = _definition(list(data.get("requires", ["tests"])), None)
    try:
        return AcceptanceRule.from_dict(data)
    except TypeError as exc:
        raise ConfigError(f"bad rule rules.{name}: {exc}") from None


def rule_to_table(rule: AcceptanceRule) -> dict[str, Any]:
    out = rule.to_dict()
    if out.get("score") is None:
        out.pop("score", None)
    return out


# ---------------------------------------------------------------- values
def check_labeler(value: str) -> str:
    """`none`, or `claude:<model>`, `codex:<model>`, `command:<cmd>`; kept verbatim."""
    text = value.strip()
    kind, sep, rest = text.partition(":")
    if text == "none" or (sep and kind in LABELER_KINDS and rest.strip()):
        return text
    raise ConfigError(f"a labeler is claude:<model>, codex:<model>, command:<cmd> or none; got {value!r}")


_FEATURE_LISTS = ("values", "labels", "types", "repos")
_FEATURE_WORDS = ("kind", "fill")


def _coerce_feature(parts: list[str], value: str) -> Any:
    """`features.min_tasks`, a whole `features.<key>` table as JSON, or one field of it."""
    dotted = ".".join(parts)
    text = value.strip()
    if text == "null":
        return None
    if parts == ["features", "min_tasks"]:
        try:
            n = int(text)
        except ValueError:
            n = 0
        if n < 1:
            raise ConfigError(f"features.min_tasks takes a whole number of 1 or more, got {value!r}")
        return n
    if len(parts) == 2:
        try:
            table = json.loads(text)
        except ValueError:
            table = None
        if not isinstance(table, dict):
            raise ConfigError(f"{dotted} takes a JSON table, for example "
                              "'{\"kind\": \"category\", \"values\": [\"web\", \"cli\"]}'")
        try:
            FeatureSet.from_config({parts[1]: table})
        except FeatureConfigError as exc:
            raise ConfigError(str(exc)) from None
        return table
    field = parts[2] if len(parts) == 3 else ""
    if field in _FEATURE_LISTS or field == "edges":
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = text
        if isinstance(parsed, str):
            parsed = [v.strip() for v in parsed.split(",") if v.strip()]
        if field == "edges":
            try:
                parsed = [float(v) for v in parsed]
            except (TypeError, ValueError):
                raise ConfigError(f"{dotted} takes a list of numbers, got {value!r}") from None
        return parsed
    if field in _FEATURE_WORDS:
        return text.lower()
    if field == "description":
        return value
    raise ConfigError(f"{dotted}: a feature field is one of kind, values, edges, labels, fill, types, repos, "
                      "description")


def check_rescue(key: str, value: Any) -> Any:
    """A `retry` rescue key's value, or ConfigError: `rescue.decay` in (0, 1], `rescue.max_attempts` a whole
    number of at least 1 (1: no retries), `rescue.min_chance` in [0, 1]."""
    number = not isinstance(value, bool) and isinstance(value, (int, float))
    if key == "rescue.decay" and not (number and 0 < value <= 1):
        raise ConfigError(f"rescue.decay takes a number above 0 and at most 1, got {value!r}")
    if key == "rescue.max_attempts" and not (number and float(value).is_integer() and value >= 1):
        raise ConfigError(f"rescue.max_attempts takes a whole number of 1 or more, got {value!r}")
    if key == "rescue.min_chance" and not (number and 0 <= value <= 1):
        raise ConfigError(f"rescue.min_chance takes a number from 0 to 1, got {value!r}")
    return int(value) if key == "rescue.max_attempts" else value


def coerce(key: str, value: str) -> Any:
    """A `config set` value: JSON when it parses, comma lists for list keys, numbers for number keys."""
    parts = split_key(key)
    dotted = ".".join(parts)
    if parts[0] == "features" and len(parts) > 1:
        return _coerce_feature(parts, value)
    if parts[0] == "rules" and len(parts) == 2:
        text = value.strip()
        if text.startswith("{"):
            try:
                table = json.loads(text)
            except ValueError as exc:
                raise ConfigError(f"rules.{parts[1]}: not JSON: {exc}") from None
            return rule_to_table(rule_from_table(parts[1], table))
        return rule_to_table(parse_rule(text, name=parts[1]))
    if dotted == "acceptance_rule":
        return value.strip()
    if parts[0] in STRING_ROOTS:
        return value
    if dotted in LABELER_KEYS:
        return check_labeler(value)
    try:
        parsed = json.loads(value)
    except ValueError:
        parsed = value
    if (dotted in LIST_KEYS or (parts[0] == "efforts" and len(parts) == 2)) and isinstance(parsed, str):
        parsed = [v.strip() for v in parsed.split(",") if v.strip()]
    if dotted in NUMBER_KEYS and parsed is not None:
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            raise ConfigError(f"{dotted} takes a number, got {value!r}")
    if dotted in POSITIVE_KEYS and parsed is not None and parsed <= 0:
        raise ConfigError(f"{dotted} takes a number above 0, got {value!r}")
    if dotted in NONNEGATIVE_KEYS and parsed is not None and parsed < 0:
        raise ConfigError(f"{dotted} takes a number of 0 or more, got {value!r}")
    if dotted in COUNT_KEYS and parsed is not None:
        if isinstance(parsed, bool) or not isinstance(parsed, int) or parsed < 0:
            raise ConfigError(f"{dotted} takes a whole number of 0 or more, got {value!r}")
    if dotted.startswith("outcome.q.") and parsed is not None:
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)) or not 0.5 <= parsed <= 1:
            raise ConfigError(f"{dotted} takes a number in [0.5, 1], got {value!r}")
    if dotted in CHOICES and parsed is not None and parsed not in CHOICES[dotted]:
        raise ConfigError(f"{dotted} is one of {', '.join(CHOICES[dotted])}; got {value!r}")
    if dotted in RESCUE_KEYS and parsed is not None:
        parsed = check_rescue(dotted, parsed)
    return parsed


# ---------------------------------------------------------------- Config
class Config:
    """config.toml with defaults. `get` falls back to DEFAULTS; `set` then `save` writes atomically."""

    def __init__(self, path: Path, data: dict[str, Any] | None = None, text: str | None = None):
        self.path = Path(path)
        self.data: dict[str, Any] = data if data is not None else {}
        self.text = text  # the file as read, so `save` keeps the lines it does not change

    @classmethod
    def load(cls, path: Path) -> "Config":
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls(path, {})
        try:
            data = loads(text)
        except ValueError as exc:
            raise ConfigError(f"{path}: {exc}") from None
        return cls(path, data, text)

    # reading
    def get(self, key: str, default: Any = _MISSING) -> Any:
        parts = split_key(key)
        for source in (self.data, DEFAULTS):
            node: Any = source
            for p in parts:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    node = _MISSING
                    break
            if node is not _MISSING:
                return copy.deepcopy(node)
        return None if default is _MISSING else default

    def merged(self) -> dict[str, Any]:
        """Defaults overlaid with the file, for `config get` with no key."""
        return _deep_merge(copy.deepcopy(DEFAULTS), self.data)

    def rules(self) -> dict[str, AcceptanceRule]:
        return {name: rule_from_table(name, t) for name, t in (self.data.get("rules") or {}).items()
                if isinstance(t, dict)}

    def rule(self, name: str | None = None) -> AcceptanceRule:
        """The named rule from `rules.<name>`, else the short form parsed; None means `acceptance_rule`."""
        if name is None:
            name = self.get("acceptance_rule") or DEFAULT_RULE.name
        named = self.rules()
        if name in named:
            return named[name]
        return parse_rule(name)

    def usual(self, task_type: str, repo: str | None = None) -> str | None:
        """`usual.<type>."<repo>"`, then `usual.<type>."*"`."""
        table = (self.data.get("usual") or {}).get(task_type)
        if isinstance(table, str):
            return table
        if not isinstance(table, dict):
            return None
        if repo is not None and isinstance(table.get(repo), str):
            return table[repo]
        star = table.get("*")
        return star if isinstance(star, str) else None

    def q_for_tier(self, tier: str) -> float:
        value = self.get(f"outcome.q.{tier}")
        return float(value) if isinstance(value, (int, float)) else 0.7

    def features(self) -> FeatureSet:
        """The built-in features plus `[features]`. Raises ConfigError on a bad declaration."""
        try:
            return FeatureSet.from_config(self.data.get("features"))
        except FeatureConfigError as exc:
            raise ConfigError(str(exc)) from None

    def features_or_builtin(self) -> FeatureSet:
        """`features()`, or the built-ins with a warning on stderr when the declaration is bad."""
        return features_from_config(self.data.get("features"))

    # writing
    def set(self, key: str, value: Any) -> None:
        """Set a dotted key; None removes it. A `features` change that leaves a bad declaration is undone."""
        parts = split_key(key)
        if parts[0] == "features":
            before = copy.deepcopy(self.data)
            self._set(parts, value)
            try:
                self.features()
            except ConfigError as exc:
                self.data.clear()
                self.data.update(before)
                hint = ""
                if len(parts) > 2 and ("edges" in str(exc) or "labels" in str(exc)):
                    hint = (f"; set the whole table at once, for example loopmath config set features.{parts[1]} "
                            "'{\"kind\": \"number\", \"edges\": [100, 1000], \"labels\": [\"low\", \"mid\", \"high\"]}'")
                raise ConfigError(f"{exc}{hint}") from None
            return
        self._set(parts, value)

    def _set(self, parts: list[str], value: Any) -> None:
        node = self.data
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                if value is None:
                    return
                nxt = node[p] = {}
            node = nxt
        if value is None:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = value

    def save(self) -> Path:
        """Write config.toml. A file that was read keeps every line whose key did not change, comments too
        (0.2.7); when its edited text would not read back as the new values, the whole file is written anew."""
        text = edit_text(self.text, self.data) if self.text else None
        return atomic_write_text(self.path, text if text is not None else dumps(self.data))


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


# ---------------------------------------------------------------- TOML
_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k: str) -> str:
    return k if _BARE.match(k) else json.dumps(k, ensure_ascii=False)


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            raise ConfigError("config values must be finite numbers")
        return json.dumps(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        if any(isinstance(x, (dict, list, tuple)) for x in v):
            raise ConfigError("config lists hold plain values only")
        return "[" + ", ".join(_value(x) for x in v if x is not None) + "]"
    raise ConfigError(f"cannot store {type(v).__name__} in config.toml")


def dumps(data: dict[str, Any]) -> str:
    """Plain tables and JSON-compatible values; None is skipped."""
    lines: list[str] = []

    def table(prefix: list[str], node: dict[str, Any]) -> None:
        scalars = [(k, v) for k, v in node.items() if not isinstance(v, dict) and v is not None]
        subs = [(k, v) for k, v in node.items() if isinstance(v, dict)]
        if prefix and (scalars or not subs):
            if lines:
                lines.append("")
            lines.append("[" + ".".join(_key(p) for p in prefix) + "]")
        for k, v in scalars:
            lines.append(f"{_key(k)} = {_value(v)}")
        for k, v in subs:
            table(prefix + [k], v)

    table([], data)
    return "\n".join(lines) + ("\n" if lines else "")


def _leaves(node: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    out: dict[tuple[str, ...], Any] = {}
    for k, v in node.items():
        if isinstance(v, dict):
            out.update(_leaves(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def _line_key(line: str) -> list[str] | None:
    """The dotted key of a one-line `key = value`, or None for anything else (comments, headers, blanks)."""
    s = line.strip()
    if not s or s.startswith(("#", "[")) or "=" not in s:
        return None
    try:
        parts = _split_header(s.split("=", 1)[0].strip()) if not s.startswith('"') else None
    except ValueError:
        return None
    return parts if parts and all(parts) else None


def _header_path(line: str) -> tuple[str, ...] | None:
    """The path of a `[table]` header line, a trailing comment allowed (`[outcome.q] # tiers`); None for any other
    line, an array of tables included."""
    s = line.strip()
    if not s.startswith("[") or s.startswith("[["):
        return None
    i, quoted = 1, False
    while i < len(s):
        c = s[i]
        if quoted:
            if c == "\\":
                i += 1
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == "]":
            break
        i += 1
    rest = s[i + 1:].strip()
    if i >= len(s) or (rest and not rest.startswith("#")):
        return None
    try:
        return tuple(_split_header(s[1:i].strip()))
    except ValueError:
        return None


def edit_text(text: str, data: dict[str, Any]) -> str | None:
    """`text` with each changed key's line replaced, removed or added in its table, when that reads back as
    `data`; else None."""
    try:
        old = loads(text)
    except ValueError:
        return None
    before, after = _leaves(old), _leaves(data)
    lines = text.splitlines()
    table: tuple[str, ...] = ()
    where: dict[tuple[str, ...], int] = {}  # a key's full path: its line
    ends: dict[tuple[str, ...], int] = {}  # a table header's path: the last line of its section that is not blank
    for i, line in enumerate(lines):
        s = line.strip()
        header = _header_path(line)
        if header is not None:
            table = header
            ends[table] = i
            continue
        key = _line_key(line)
        if key is not None:
            where[table + tuple(key)] = i
        if s:
            ends[table] = i
    drop: set[int] = set()
    put: dict[int, str] = {}
    added: list[tuple[tuple[str, ...], Any]] = []
    for path in sorted(set(before) | set(after)):
        if before.get(path, _MISSING) == after.get(path, _MISSING):
            continue
        if path in where:
            if path in after:
                put[where[path]] = f"{lines[where[path]].split('=', 1)[0].rstrip()} = {_value(after[path])}"
            else:
                drop.add(where[path])
        elif path in after:
            added.append((path, after[path]))
        else:
            return None
    inserts: dict[int, list[str]] = {}
    tail: list[str] = []
    for path, value in added:
        line = f"{_key(path[-1])} = {_value(value)}"
        if path[:-1] in ends or not path[:-1]:
            inserts.setdefault(ends.get(path[:-1], -1), []).append(line)
        else:
            tail += ["", "[" + ".".join(_key(p) for p in path[:-1]) + "]", line]
    out = inserts.get(-1, [])[:]
    for i, line in enumerate(lines):
        if i not in drop:
            out.append(put.get(i, line))
        out += inserts.get(i, [])
    new = "\n".join(out + tail).strip("\n") + "\n"
    try:
        return new if loads(new) == data else None
    except ValueError:
        return None


def loads(text: str) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # Python 3.10: no third-party TOML reader is declared
        return _loads_subset(text)
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(str(exc)) from None


def _split_header(text: str) -> list[str]:
    parts: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == '"':
            j = i + 1
            while j < len(text) and not (text[j] == '"' and text[j - 1] != "\\"):
                j += 1
            parts.append(json.loads(text[i:j + 1]))
            i = j + 1
        else:
            j = text.find(".", i)
            j = len(text) if j < 0 else j
            parts.append(text[i:j].strip())
            i = j
        while i < len(text) and text[i] in " .":
            i += 1
    return parts


def _loads_subset(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    node = data
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            node = data
            for p in _split_header(line[1:-1].strip()):
                node = node.setdefault(p, {})
            continue
        if line.startswith('"'):
            end = line.find('"', 1)
            while end > 0 and line[end - 1] == "\\":
                end = line.find('"', end + 1)
            key = json.loads(line[:end + 1])
            rest = line[end + 1:].lstrip()
        else:
            key, _, rest = line.partition("=")
            key = key.strip()
            rest = "=" + rest
        if not rest.startswith("="):
            raise ValueError(f"line {n}: expected key = value")
        try:
            node[key] = json.loads(rest[1:].strip())
        except ValueError:
            raise ValueError(f"line {n}: value not readable without tomllib: {raw!r}") from None
    return data
