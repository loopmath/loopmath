"""Task types, features and the grouping chain (design/0.1/03-interfaces.md, section 3; Q6).

Types are fixed. Subtypes are the user's own (config `subtypes`). Features are a
short fixed list the orchestrator fills; unknown is always allowed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    values: tuple[str, ...]
    description: str
    default: str = "unknown"


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


def normalize_features(raw: dict[str, str]) -> dict[str, str]:
    """Lowercase, bucket numbers, keep unknown keys under 'extra:<key>'."""
    out: dict[str, str] = {}
    for key, value in (raw or {}).items():
        k = str(key).strip().lower()
        v = str(value).strip().lower()
        if k == "size" and v.isdigit():
            v = size_bucket(int(v))
        elif k == "touches" and v.isdigit():
            v = touches_bucket(int(v))
        if k in FEATURES:
            spec = FEATURES[k]
            if spec.values and v not in spec.values:
                v = spec.default
            out[k] = v
        else:
            out[f"extra:{k}"] = v
    return out


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


def task_types_payload(subtypes: list[str] | None = None) -> dict:
    """The `loopmath task-types --json` object (schema loopmath.task-types/1)."""
    return {
        "schema": "loopmath.task-types/1",
        "types": [{"id": k, "title": k.replace("_", " "), "description": v} for k, v in TASK_TYPES.items()],
        "features": [{"key": f.key, "values": list(f.values), "description": f.description} for f in FEATURES.values()],
        "subtypes": list(subtypes or []),
    }


# ---------------------------------------------------------------- labelling hooks (lane 03)
# One definition of a task label, shared by `loopmath onboard` (which labels past
# session groups through the user's own CLI) and by any orchestrator that wants
# to ask a model for a type: the instructions, the JSON Schema of the answer, the
# validator every answer passes through, and a keyword guess for `--labeler none`.

LABEL_VERSION = "label/1"
UNKNOWN_TYPE = "unknown"  # a labeller's way to say "cannot tell"; never stored as a task type
TITLE_MAX = 80


def _feature_values(spec: FeatureSpec) -> list[str]:
    return [*spec.values, spec.default] if spec.values else []


def label_schema(subtypes: list[str] | None = None) -> dict:
    """JSON Schema of one labelling answer: {"labels": [{id, type, subtype, features, confidence, title}]}.

    Every object is closed and every property required, so the same schema works as
    `claude -p --json-schema` and as `codex exec --output-schema` (strict mode).
    """
    features = {}
    for spec in FEATURES.values():
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


def label_instructions(subtypes: list[str] | None = None) -> str:
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
    for spec in FEATURES.values():
        values = "|".join(_feature_values(spec)) or "lowercase language name, or unknown"
        lines.append(f"- {spec.key} ({values}): {spec.description}")
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


def validate_label(raw: object, subtypes: list[str] | None = None) -> tuple[dict | None, list[str]]:
    """Check one labeller answer item. Returns (label, problems).

    `label` is {type, subtype, features, confidence, title}, or None when the item has
    no usable type (missing, invalid, or `unknown`). Features go through
    `normalize_features`; values the labeller left unknown and keys outside FEATURES
    are dropped, so a stored task carries only what was actually labelled. A subtype
    outside `subtypes` is dropped with a problem noted; the type is kept.
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
    features: dict[str, str] = {}
    raw_features = raw.get("features")
    if isinstance(raw_features, dict):
        for key, value in normalize_features({str(k): str(v) for k, v in raw_features.items()}).items():
            if key in FEATURES and value != FEATURES[key].default and value:
                features[key] = value
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
        problems.append("confidence missing")
    if confidence != confidence:  # NaN
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    title = " ".join(str(raw.get("title") or "").split())[:TITLE_MAX]
    return {"type": kind, "subtype": subtype, "features": features,
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
