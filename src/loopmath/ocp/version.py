"""OCP version strings: '0.x' compared numerically (spec 01 section 3).

Readers gate v0.2 behaviour on "version at least 0.2", so a v0.3 document
(a strict superset) takes every v0.2 rule and read path.
"""

from __future__ import annotations

import re
from typing import Any

_VERSION = re.compile(r"^(\d+)\.(\d+)$")


def parse_version(value: Any) -> tuple[int, int] | None:
    """(major, minor) for an OCP version string such as '0.3', else None."""
    if not isinstance(value, str):
        return None
    match = _VERSION.match(value)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def version_at_least(doc_or_version: Any, minimum: str) -> bool:
    """True when the document's `ocp` field (or the version string) is at least `minimum`.

    An unparseable or missing version is never "at least" anything.
    """
    value = doc_or_version.get("ocp") if isinstance(doc_or_version, dict) else doc_or_version
    have, want = parse_version(value), parse_version(minimum)
    if have is None or want is None:
        return False
    return have >= want
