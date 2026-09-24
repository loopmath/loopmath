"""Canonical JSON and configuration ids: `cfg_` + the first 12 hex of SHA-256 (OCP 2.2).

D2: the id has one definition, in OCP terms, owned by lane 01 as
`loopmath.ocp.canonical`. The functions here keep their public names over
`types` objects: they map through `workflow_to_ocp` and `settings_to_ocp` and
call lane 01's `config_id`, `canonical_json` and `canonical_workflow`.

`tests/workflows/test_workflows_ids.py` pins ids, so a change to the definition
shows up there.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..ocp import canonical as _canonical
from ..types import Configuration, Setting, Workflow
from .ocp import settings_to_ocp, workflow_to_ocp


def config_id_ocp(workflow_ocp: Mapping[str, Any], settings_ocp: Mapping[str, Any]) -> str:
    """The id from OCP objects (lane 01's definition, D2)."""
    return _canonical.config_id(workflow_ocp, settings_ocp)


def canonical(workflow: Workflow, settings: Mapping[str, Setting]) -> str:
    """Canonical JSON of `{workflow, settings}` in OCP terms."""
    return _canonical.canonical_json(workflow_to_ocp(workflow), settings_to_ocp(settings))


def config_id(workflow: Workflow, settings: Mapping[str, Setting]) -> str:
    """`cfg_` + 12 hex over the canonical `{workflow, settings}`: same shape and settings, same id."""
    return config_id_ocp(workflow_to_ocp(workflow), settings_to_ocp(settings))


def structure_key(workflow: Workflow) -> str:
    """The canonical workflow alone, without settings: equal keys mean the same shape."""
    return json.dumps(_canonical.canonical_workflow(workflow_to_ocp(workflow)), sort_keys=True, separators=(",", ":"))


def make_config(workflow: Workflow, settings: Mapping[str, Setting], extra: dict[str, Any] | None = None) -> Configuration:
    """A Configuration with its id; settings are ordered by piece."""
    order = [p.id for p in workflow.pieces]
    ordered = {k: settings[k] for k in order if k in settings}
    ordered.update({k: v for k, v in settings.items() if k not in ordered})
    return Configuration(id=config_id(workflow, ordered), workflow=workflow, settings=ordered, extra=dict(extra or {}))
