"""OCP v0.3: schema, conformance checker (moved from spec/), migration, emit helpers (spec 01).

Owner: lane 01. Spec: spec/OCP.md (section 8 for v0.3).

- `conformance`: the checker (`validate_doc`, `validate_file`, `validate_many`).
- `canonical`: the canonical JSON of a configuration and its `cfg_` id.
- `emit`: OCP v0.3 records built from `loopmath.types` values, for the store.
- `migrate`: v0.1 and v0.2 documents (and contract v3 run files) to v0.3.
"""

from __future__ import annotations

from .version import parse_version, version_at_least

CURRENT_VERSION = "0.3"

__all__ = ["CURRENT_VERSION", "parse_version", "version_at_least"]
