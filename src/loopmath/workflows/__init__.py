"""Workflow catalog, TOML format, configuration ids, candidates, diff and inference (spec 03, spec 05 section 1).

Names below load their module on first use, so `import loopmath.workflows`
stays cheap and cannot form an import cycle with `loopmath.ocp`. `candidates`,
`diff` and `infer` are submodules (lane 01's migrate imports `infer` as a
module), so their functions are imported from them:
`from loopmath.workflows.diff import diff`.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "catalog": "format",
    "load_workflow": "format",
    "load_workflow_file": "format",
    "dump_workflow": "format",
    "validate_workflow": "format",
    "validate_configuration": "format",
    "user_workflows": "format",
    "WorkflowFormatError": "format",
    "config_id": "ids",
    "make_config": "ids",
    "canonical": "ids",
    "infer_detail": "infer",
    "infer_workflow": "infer",
    "normalize_role": "models",
    "family_of": "models",
    "harness_for": "models",
    "workflow_to_ocp": "ocp",
    "configuration_to_ocp": "ocp",
    "configuration_from_any": "ocp",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
