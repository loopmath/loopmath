"""Validation for the modeled ``dev.loopmath.artifact`` OCP extension (``dev.dagr.artifact`` before 0.3)."""

from __future__ import annotations

import copy
import re

from .ocp_common import OCPError


# The graph writer's key, then the one it wrote before 0.3 (read through v0.4, W182).
_ARTIFACT_EXT_KEYS = ("dev.loopmath.artifact", "dev.dagr.artifact")
_ARTIFACT_TIERS = frozenset({"verified", "reported", "heuristic"})
_ARTIFACT_FATES = frozenset({"kept", "edited", "reverted", "deleted", "unknown"})
_ARTIFACT_LANGUAGE_RE = re.compile(r"^[a-z0-9][a-z0-9_+-]*$")
_ARTIFACT_COUNT_FIELDS = ("bytes", "lines_added", "lines_removed", "tests_touched")
_ARTIFACT_TIERED_FIELDS = (*_ARTIFACT_COUNT_FIELDS, "language")
_ARTIFACT_MISSING = object()
_ARTIFACT_EXT_FIELDS = frozenset(
    {
        *_ARTIFACT_TIERED_FIELDS,
        *(f"{field}_tier" for field in _ARTIFACT_TIERED_FIELDS),
        "fate",
        "fate_tier",
        "meta",
    }
)


def _artifact_extension(record: dict, artifact_id: str) -> dict:
    """Validate and copy one ``dev.loopmath.artifact`` (or ``dev.dagr.artifact``) extension.

    Absence is the pre-Q7/default shape. Once the namespace is present, values
    and evidence tiers must be paired so a missing measurement cannot carry a
    confidence claim and a measured value cannot silently lack one.
    """
    extensions = record.get("ext")
    if not isinstance(extensions, dict):
        return {}
    key = next((k for k in _ARTIFACT_EXT_KEYS if k in extensions), None)
    if key is None:
        return {}
    raw = extensions[key]
    if not isinstance(raw, dict):
        raise OCPError(
            f"OCP artifact {artifact_id!r} extension {key!r} must be an object"
        )
    unknown = sorted(set(raw) - _ARTIFACT_EXT_FIELDS, key=str)
    if unknown:
        raise OCPError(
            f"OCP artifact {artifact_id!r} extension {key!r} has unknown fields {unknown!r}; put producer-specific data in meta"
        )

    values: dict = {}
    for field in _ARTIFACT_COUNT_FIELDS:
        member = raw[field] if field in raw else _ARTIFACT_MISSING
        value = None if member is _ARTIFACT_MISSING else member
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise OCPError(
                f"OCP artifact {artifact_id!r} extension field {field!r} must be a nonnegative integer or null"
            )
        values[field] = value

    language_member = raw["language"] if "language" in raw else _ARTIFACT_MISSING
    language = None if language_member is _ARTIFACT_MISSING else language_member
    if language is not None and (
        not isinstance(language, str)
        or _ARTIFACT_LANGUAGE_RE.fullmatch(language) is None
    ):
        raise OCPError(
            f"OCP artifact {artifact_id!r} extension field 'language' must be a lowercase extension identifier or null"
        )
    values["language"] = language

    for field in _ARTIFACT_TIERED_FIELDS:
        tier_field = f"{field}_tier"
        tier_member = raw[tier_field] if tier_field in raw else _ARTIFACT_MISSING
        tier = None if tier_member is _ARTIFACT_MISSING else tier_member
        if tier is not None and (
            not isinstance(tier, str) or tier not in _ARTIFACT_TIERS
        ):
            raise OCPError(
                f"OCP artifact {artifact_id!r} extension field {tier_field!r} must be verified, reported, heuristic, or null"
            )
        if (values[field] is None) != (tier is None):
            raise OCPError(
                f"OCP artifact {artifact_id!r} extension fields {field!r} and {tier_field!r} must be null together"
            )
        values[tier_field] = tier

    fate_member = raw["fate"] if "fate" in raw else _ARTIFACT_MISSING
    fate = "unknown" if fate_member is _ARTIFACT_MISSING else fate_member
    if not isinstance(fate, str) or fate not in _ARTIFACT_FATES:
        raise OCPError(
            f"OCP artifact {artifact_id!r} extension field 'fate' must be kept, edited, reverted, deleted, or unknown"
        )
    fate_tier_member = raw["fate_tier"] if "fate_tier" in raw else _ARTIFACT_MISSING
    fate_tier = None if fate_tier_member is _ARTIFACT_MISSING else fate_tier_member
    if fate_tier is not None and (
        not isinstance(fate_tier, str) or fate_tier not in _ARTIFACT_TIERS
    ):
        raise OCPError(
            f"OCP artifact {artifact_id!r} extension field 'fate_tier' must be verified, reported, heuristic, or null"
        )
    if (fate == "unknown") != (fate_tier is None):
        raise OCPError(
            f"OCP artifact {artifact_id!r} extension fields 'fate' and 'fate_tier' must use unknown and null together"
        )

    meta_member = raw["meta"] if "meta" in raw else _ARTIFACT_MISSING
    meta = {} if meta_member is _ARTIFACT_MISSING else meta_member
    if not isinstance(meta, dict):
        raise OCPError(
            f"OCP artifact {artifact_id!r} extension field 'meta' must be an object"
        )
    missing_fields = sorted(_ARTIFACT_EXT_FIELDS - set(raw))
    if missing_fields:
        raise OCPError(
            f"OCP artifact {artifact_id!r} extension {key!r} is missing required fields {missing_fields!r}"
        )
    return {
        **values,
        "fate": fate,
        "fate_tier": fate_tier,
        "meta": copy.deepcopy(meta),
    }
