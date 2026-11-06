"""The flow folder: `flow.ocp.json` (or its TOML view `flow.toml`), the record, and `plan.json` beside it.

`load` reads a folder holding `flow.ocp.json` or `flow.toml` (the JSON when both are there), or a flow document
of any name, read as TOML when the name ends in `.toml`. The document must validate with no error. Writes go
to `flow.ocp.json` in the folder. A plan made for another flow digest is stale: `load` still reads it, so
`flow plan` can rewrite it, but its arms are refused until then.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ocp import conformance, finite
from ..ocp.canonical import config_id, flow_digest
from .records import Flow, Plan
from .tomlview import flow_from_toml

FLOW_FILE = "flow.ocp.json"
TOML_FILE = "flow.toml"
PLAN_FILE = "plan.json"
ACCEPTED = f"a folder holding {FLOW_FILE} or {TOML_FILE}, or a flow document"


@dataclass(frozen=True)
class FlowFolder:
    """A loaded flow: the document, its `Flow`, the computed digest and the plan, if any."""

    path: Path  # the flow document read: flow.ocp.json, flow.toml or the file given
    dir: Path
    doc: dict[str, Any]
    flow: Flow
    digest: str  # flow_digest of the document's flow
    plan: Plan | None

    @property
    def plan_stale(self) -> bool:
        """True when a plan is present and was made for another flow digest."""
        return self.plan is not None and self.plan.flow.digest != self.digest

    def configuration(self, arm: str | None = None) -> tuple[str, dict[str, Any]]:
        """The arm id and a copy of its OCP configuration; None or `base` gives the flow's configuration."""
        if arm is None or arm == "base":
            return "base", copy.deepcopy(self.flow.configuration)
        plan = self.plan
        if plan is None:
            raise ValueError(f"no arm {arm!r}: there is no {PLAN_FILE} in {self.dir}, so the only arm is base")
        if self.plan_stale:
            raise ValueError(f"{self.dir / PLAN_FILE} is stale: it was made for flow digest {plan.flow.digest}, "
                             f"the flow's digest is {self.digest}; run loopmath flow plan")
        found = [a for a in plan.arms if a.id == arm]
        if not found:
            arms = ", ".join(["base"] + [a.id for a in plan.arms if a.id != "base"])
            raise ValueError(f"no arm {arm!r} in {self.dir / PLAN_FILE}; the arms are {arms}")
        configuration = copy.deepcopy(found[0].configuration)
        workflow, settings = configuration.get("workflow"), configuration.get("settings")
        try:  # a workflow reference resolves through the catalog, as validation does
            computed = config_id(workflow, settings, resolve=conformance.catalog_resolver) \
                if isinstance(workflow, dict) and isinstance(settings, dict) else None
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ValueError(f"arm {arm!r}: the configuration id cannot be computed ({exc})") from exc
        if computed is None or configuration.get("id") != computed:
            raise ValueError(f"arm {arm!r}: the configuration id {configuration.get('id')!r} does not match its "
                             f"workflow and settings ({computed})")
        return arm, configuration


def load(path: str | os.PathLike[str]) -> FlowFolder:
    """Read a flow folder or a flow document, and `plan.json` beside it when present."""
    given = Path(path)
    if given.is_dir():
        doc_path = given / FLOW_FILE
        if not doc_path.is_file():
            doc_path = given / TOML_FILE
        if not doc_path.is_file():
            raise ValueError(f"{given}: there is no {FLOW_FILE} here, and no {TOML_FILE}; give {ACCEPTED}")
    elif given.is_file():
        doc_path = given
    else:
        raise ValueError(f"{given}: no such file or folder; give {ACCEPTED}")
    if doc_path.suffix == ".toml":
        try:
            doc = flow_from_toml(doc_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"{doc_path}: not a flow TOML view ({exc}); give {ACCEPTED}") from exc
    else:
        try:
            doc = finite.loads(doc_path.read_text(encoding="utf-8"))
        except finite.NotFinite as exc:
            raise ValueError(f"{doc_path}: {exc}") from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{doc_path}: not a JSON document ({exc}); give {ACCEPTED}") from exc
        except ValueError as exc:  # the interpreter's limit on reading an integer; the limit is never lifted
            raise ValueError(f"{doc_path}: a number with more than {sys.get_int_max_str_digits()} digits cannot be "
                             f"read; give {FLOW_FILE} with shorter numbers") from exc
    if not isinstance(doc, dict) or doc.get("kind") != "flow":
        raise ValueError(f"{doc_path}: not a flow document (kind 'flow'); give {ACCEPTED}")
    errors = [f for f in conformance.validate_doc(doc) if f.level == "error"]
    if errors:
        raise ValueError(f"{doc_path}: the flow document does not validate: "
                         + "; ".join(f"{f.code} at {f.path}" for f in errors))
    flow = Flow.from_document(doc)
    plan = None
    plan_path = doc_path.parent / PLAN_FILE
    if plan_path.is_file():
        try:
            plan = Plan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{plan_path}: {exc}") from exc
    return FlowFolder(path=doc_path, dir=doc_path.parent, doc=doc, flow=flow, digest=flow_digest(doc["flow"]),
                      plan=plan)


def save_plan(directory: str | os.PathLike[str], plan: Plan) -> Path:
    """Write `plan.json` in `directory` atomically: indent 2, UTF-8 as written, a final newline."""
    path = Path(directory) / PLAN_FILE
    text = json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
