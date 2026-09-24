"""Model, harness and effort helpers for workflows and candidates.

Small and deliberately local: the belief model (lane 05) owns the provider,
family and version forest; these helpers only answer what candidate generation
needs (which harness runs a model, whether two models share a family, which
efforts a harness offers) and parse `PIECE=HARNESS:MODEL:EFFORT`.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

from ..types import Setting

# Efforts each harness offers, lowest first. Config `efforts.<harness>` overrides.
DEFAULT_EFFORTS: dict[str, tuple[str, ...]] = {
    "claude-code": ("low", "medium", "high", "xhigh", "max"),
    "codex": ("low", "medium", "high", "xhigh"),
}

# Ordinal scale used to find the nearest offered effort.
EFFORT_ORDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

_ANTHROPIC_FAMILIES = ("opus", "sonnet", "fable", "haiku")

# Role words: the types.py vocabulary is canonical (D32). OCP's recommended words
# and common synonyms fold onto it for readers that group by role (lane 05's role
# node); a superset of lane 05's local table, so both give the same word.
ROLE_ALIASES: dict[str, str] = {
    **dict.fromkeys(("plan", "planner", "planning", "architect"), "planner"),
    **dict.fromkeys(("implement", "implementer", "dev", "developer", "coder", "impl", "build", "builder",
                     "fix", "integrate", "integrator"), "implementer"),
    **dict.fromkeys(("review", "reviewer", "critic", "audit"), "reviewer"),
    **dict.fromkeys(("test", "tester", "tests", "qa"), "tester"),
    **dict.fromkeys(("select", "selector", "referee", "judge"), "referee"),
    **dict.fromkeys(("worker", "lead", "research", "docs", "ops"), "worker"),
}


def normalize_role(role: str | None) -> str | None:
    """Fold a role word onto the types.py vocabulary (D32); unknown words pass through lowercased."""
    if role is None:
        return None
    key = str(role).strip().lower()
    return ROLE_ALIASES.get(key, key)


def provider_of(model: str) -> str:
    m = str(model).strip().lower()
    if m.startswith("claude") or m.split("-")[0] in _ANTHROPIC_FAMILIES:
        return "anthropic"
    if m.startswith("gpt") or re.match(r"^o\d", m) or "codex" in m:
        return "openai"
    if m.startswith("gemini"):
        return "google"
    return "other"


def family_of(model: str) -> str:
    """Model family: `claude-opus-5-5` -> opus, `gpt-6-astra` -> astra, `gpt-5.4` -> gpt."""
    m = str(model).strip().lower()
    tokens = [t for t in re.split(r"[-_./:]", m) if t]
    for t in tokens:
        if t in _ANTHROPIC_FAMILIES:
            return t
    if provider_of(m) == "openai":
        words = [t for t in tokens[1:] if t.isalpha()]
        return words[0] if words else "gpt"
    words = [t for t in tokens if t.isalpha()]
    return words[0] if words else m


def harness_for(model: str, overrides: Mapping[str, str] | None = None) -> str | None:
    """The harness that runs a model: Claude models in Claude Code, OpenAI models in Codex."""
    if overrides and model in overrides:
        return overrides[model]
    provider = provider_of(model)
    if provider == "anthropic":
        return "claude-code"
    if provider == "openai":
        return "codex"
    return None


def efforts_for(harness: str, overrides: Mapping[str, Sequence[str]] | None = None) -> tuple[str, ...]:
    if overrides and harness in overrides:
        return tuple(overrides[harness])
    return DEFAULT_EFFORTS.get(harness, ("default",))


def nearest_effort(effort: str, offered: Sequence[str]) -> str:
    """`effort` when offered, else the offered effort closest on EFFORT_ORDER (ties go lower)."""
    if not offered:
        return effort
    if effort in offered:
        return effort
    if effort not in EFFORT_ORDER:
        return "high" if "high" in offered else offered[len(offered) // 2]
    want = EFFORT_ORDER.index(effort)
    ranked = [e for e in offered if e in EFFORT_ORDER]
    if not ranked:
        return offered[0]
    return min(ranked, key=lambda e: (abs(EFFORT_ORDER.index(e) - want), EFFORT_ORDER.index(e)))


def parse_set(text: str) -> tuple[str, Setting]:
    """Parse `PIECE=HARNESS:MODEL[:EFFORT]` (the `run start --set` form)."""
    piece, sep, rest = str(text).partition("=")
    piece = piece.strip()
    if not sep or not piece or not rest:
        raise ValueError(f"expected PIECE=HARNESS:MODEL:EFFORT, got {text!r}")
    harness, sep, tail = rest.partition(":")
    if not sep or not harness or not tail:
        raise ValueError(f"expected PIECE=HARNESS:MODEL:EFFORT, got {text!r}")
    if ":" in tail:
        model, effort = tail.rsplit(":", 1)
    else:
        model, effort = tail, "default"
    if not model or not effort:
        raise ValueError(f"expected PIECE=HARNESS:MODEL:EFFORT, got {text!r}")
    return piece, Setting(harness=harness.strip(), model=model.strip(), effort=effort.strip())


def parse_sets(values: Iterable[str]) -> dict[str, Setting]:
    out: dict[str, Setting] = {}
    for value in values:
        piece, setting = parse_set(value)
        if piece in out:
            raise ValueError(f"piece {piece!r} is set twice")
        out[piece] = setting
    return out


def setting_label(setting: Setting | None) -> str:
    if setting is None:
        return "unset"
    return f"{setting.model}/{setting.effort}"
