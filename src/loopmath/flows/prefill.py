"""What logged runs say about a flow: `loopmath flow init --from-ocp FILE... --from-rows FILE`.

The runs are read in a fixed order, the OCP files as given and then the rows in file order, and every one is
checked before an answer is read or a file is written. One refusal names every refused file or row:

- an OCP file is read as UTF-8 JSON and must be an OCP run document (`ocp` and a `run` object), not a flow
  document, with no error finding;
- a rows file is read as `adapt harness-runs` reads it (`harness_runs.rows.read_numbered`), each row converted
  with no flow (`harness_runs.ocp.to_ocp`). A row naming a flow is refused: a flow's own rows go to
  `loopmath flow record`. A rows file with no row is refused, so every file given gives a run.

A run declares its configuration when its `run.configuration` is inline (a workflow with pieces, and settings)
and either the run names a flow (`run.flow`) or the configuration's `source` is not `habit`. The `habit`
configuration the row converter writes for a logged row records why it ran, not a declaration, so prefill infers
from it. Runs that all declare one configuration give it exactly (`how: declared`, confidence 1.0). Runs that
declare none are inferred, each by `workflows.infer.infer_detail`. A mix, declared runs of two configuration ids,
a declared workflow given by catalog reference and one run id twice are refused by name.

Inferred runs give the steps in first-appearance order. Each step takes the most common whole piece (role and
width) and the most common whole setting among the runs that have it, a tie going to the one read first, so no
step gets a setting no run used. A setting read from a run's configuration keeps its option values as the
configuration gives them. The proposed budget is the most rounds seen (an attempt with no round counts
1). Checks, the repair map and artifacts are never taken from inferred runs. The confidence is a deterministic
heuristic, not a probability: the mean of `infer_detail`'s confidences times the agreement, the smallest share,
over the steps, of all runs whose piece and setting for that step are the chosen ones.

`Prefill.ext` is `flow.ext["dev.loopmath.prefill"]`: `{"from": [run ids], "how", "confidence"}`.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

from ..harness_runs.ocp import to_ocp
from ..harness_runs.rows import names_flow, read_numbered
from ..ocp import conformance, finite
from ..output import counted
from ..workflows.infer import infer_detail
from ..workflows.ocp import settings_from_ocp
from . import walk
from .tomlview import _first_difference, _one_line, _same

DECLARED, INFERRED = "declared", "inferred"
ROWS_FLOW = ("names a flow; prefill reads logged runs without one, and a flow's own rows go to loopmath flow record "
             "FLOW --rows FILE")
ONE_CONFIGURATION = "give the runs of one configuration, or runs that declare none"


class Refused(ValueError):
    """Runs refused: `problems` holds one line each, a file or row named; nothing was written."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclasses.dataclass(frozen=True)
class Prefill:
    """What the runs say. `how` is "declared" or "inferred"; `runs` holds the run ids in read order. Declared runs
    give `configuration`, the first declaring run's `run.configuration` as given. Inferred runs give `steps`, each
    `{id, role, width, harness, model, effort, context_policy, options}` in the runs' order, and `rounds`, the
    most rounds seen. `notes` names each later declaring run whose configuration has other members."""

    how: str
    runs: tuple[str, ...]
    confidence: float
    configuration: dict[str, Any] | None = None
    steps: tuple[dict[str, Any], ...] = ()
    rounds: int = 1
    notes: tuple[str, ...] = ()

    @property
    def ext(self) -> dict[str, Any]:
        return {"from": list(self.runs), "how": self.how, "confidence": self.confidence}

    def summary(self) -> list[str]:
        """The lines that show what the runs give: the runs, then each step with its role and setting."""
        lines = [f"prefilled from {counted(len(self.runs), 'run')} (how {self.how}, confidence {self.confidence}): "
                 f"{', '.join(self.runs)}"]
        for step in _shown_steps(self):
            width = f", width {step['width']}" if step["width"] != 1 else ""
            lines.append(f"  {step['id']}: {step['role']}{width}, {step['model']} on {step['harness']}, "
                         f"effort {step['effort']}")
        if self.how == DECLARED:
            lines.append(f"  configuration {self.configuration.get('id')} as declared, with its checks and budget")
        else:
            lines.append(f"  the runs saw at most {counted(self.rounds, 'round')}; budget_rounds {self.rounds} is "
                         "proposed")
        return [_one_line(line) for line in lines]


def _shown_steps(p: Prefill) -> list[dict[str, Any]]:
    if p.how != DECLARED:
        return list(p.steps)
    workflow, settings = p.configuration["workflow"], settings_from_ocp(p.configuration["settings"])
    out = []
    for piece in workflow.get("pieces") or ():
        s = settings.get(piece.get("id")) if isinstance(piece, dict) else None
        if s is not None:
            out.append({"id": piece["id"], "role": piece.get("role"), "width": piece.get("width", 1),
                        "harness": s.harness, "model": s.model, "effort": s.effort})
    return out


# ------------------------------------------------------------------ reading the runs

def _findings(doc: dict[str, Any]) -> str | None:
    errors = [f for f in conformance.validate_doc(doc) if f.level == "error"]
    if not errors:
        return None
    return "; ".join(f"{f.code} at {f.path}: {f.message}" for f in errors[:3]) + (
        f" and {len(errors) - 3} more" if len(errors) > 3 else "")


def _ocp_file(name: str) -> tuple[dict[str, Any] | None, str | None]:
    """(the run document, None), or (None, why it is refused); read as `flow record` reads a run document."""
    try:
        doc = finite.loads(Path(name).expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "no such file"
    except finite.NotFinite as exc:
        return None, str(exc)
    except (OSError, ValueError) as exc:  # not UTF-8, not JSON, or an integer of 4300+ digits
        return None, f"cannot be read as JSON: {exc}"
    if not isinstance(doc, dict) or "ocp" not in doc:
        return None, "not an OCP document (it has no ocp member); --from-ocp takes run documents"
    if doc.get("kind") == "flow":
        return None, "a flow document; --from-ocp takes run documents of the workflow"
    if not isinstance(doc.get("run"), dict):
        return None, "not an OCP run document (it has no run object); --from-ocp takes run documents"
    why = _findings(doc)
    return (None, why) if why else (doc, None)


def _rows_file(name: str) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    """(where and document per row, refusals): the rows read and converted as `adapt harness-runs` does, no flow."""
    from ..harness_runs.commands import _mtime

    path = Path(name).expanduser()
    try:
        numbered = read_numbered(path)
    except ValueError as exc:  # names the file, or the file and line
        return [], [str(exc)]
    if not numbered:
        return [], [f"{name}: no rows; each file given must give a run"]
    default = _mtime(path)
    docs, problems = [], []
    for n, row in numbered:
        where = f"{name}:{n}"
        if names_flow(row):
            problems.append(f"{where}: row {row.get('id')} {ROWS_FLOW}")
            continue
        try:
            doc = to_ocp(row, base_dir=path.parent, observed_default=default)
        except (OSError, ValueError) as exc:
            problems.append(f"{where}: {exc}")
            continue
        why = _findings(doc)
        if why:
            problems.append(f"{where}: {why}")
            continue
        docs.append((where, doc))
    return docs, problems


def read(ocp_files: list[str] | None, rows_file: str | None) -> list[tuple[str, dict[str, Any]]]:
    """(where, run document) per run in read order: the OCP files as given, then the rows in file order. Raises
    Refused naming every refused file or row, and each run id given twice with both places."""
    runs, problems = [], []
    for name in ocp_files or ():
        doc, why = _ocp_file(name)
        if why is not None:
            problems.append(f"{name}: {why}")
        else:
            runs.append((name, doc))
    if rows_file is not None:
        docs, refused = _rows_file(rows_file)
        runs += docs
        problems += refused
    if problems:
        raise Refused(problems)
    first: dict[str, str] = {}
    for where, doc in runs:
        rid = doc["run"]["id"]
        if rid in first:
            problems.append(f"{where}: run {rid} is also given by {first[rid]}; give each run once")
        first.setdefault(rid, where)
    if problems:
        raise Refused(problems)
    return runs


# ------------------------------------------------------------------ declared or inferred

def declaration(run: dict[str, Any]) -> str | None:
    """"declared" when the run declares its configuration inline, "reference" when it declares one whose workflow
    is a catalog reference, None when it declares none (no inline configuration, or the `habit` one of a logged
    row with no flow)."""
    cfg = run.get("configuration")
    if not (isinstance(cfg, dict) and isinstance(cfg.get("workflow"), dict) and isinstance(cfg.get("settings"), dict)):
        return None
    if not isinstance(run.get("flow"), dict) and cfg.get("source") == "habit":
        return None
    return "declared" if "pieces" in cfg["workflow"] else "reference"


def _named(where: str, doc: dict[str, Any]) -> str:
    return f"{doc['run']['id']} ({where})"


def _declared(declaring: list[tuple[str, dict[str, Any]]]) -> Prefill:
    by_id: dict[Any, list[str]] = {}
    for where, doc in declaring:
        by_id.setdefault(doc["run"]["configuration"].get("id"), []).append(_named(where, doc))
    if len(by_id) > 1:
        raise Refused([f"the runs declare {len(by_id)} configurations; {ONE_CONFIGURATION}"]
                      + [f"{cid}: declared by {', '.join(names)}" for cid, names in by_id.items()])
    (first_where, first), cfg_id = declaring[0], next(iter(by_id))
    configuration = copy.deepcopy(first["run"]["configuration"])
    notes = []
    for where, doc in declaring[1:]:
        other = doc["run"]["configuration"]
        if not _same(other, configuration):
            at = ".".join(str(k) for k in _first_difference(configuration, other, [])) or "the whole object"
            notes.append(f"run {_named(where, doc)} declares {cfg_id} as run {_named(first_where, first)} does, "
                         f"with other members (first at {at}); the first run's configuration is taken")
    return Prefill(DECLARED, tuple(doc["run"]["id"] for _, doc in declaring), 1.0, configuration=configuration,
                   notes=tuple(notes))


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _mode(values: list[Any]) -> Any:
    """The most common value, compared whole; a tie goes to the one seen first."""
    counts: dict[str, list[Any]] = {}
    for value in values:
        counts.setdefault(_canon(value), [0, value])[0] += 1
    return max(counts.values(), key=lambda c: c[0])[1]  # max keeps the first of equal counts


def _whole(value: Any) -> int | None:
    """`value` as an integer the way JSON Schema reads one (2 and 2.0 alike), else None; a boolean is not one. The
    rule of `harness_runs.rows._whole`."""
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _round(attempt: Any) -> int:
    """An attempt's `round` (2.0 counts 2), 1 when it has none."""
    value = _whole(attempt.get("round")) if isinstance(attempt, dict) else None
    return value if value is not None and value >= 1 else 1


def _rounds(doc: dict[str, Any]) -> int:
    """The most rounds a run saw: its attempts' highest `round`, an attempt with none counting 1, a run with no
    attempts 1."""
    return max((_round(a) for a in doc.get("attempts") or ()), default=1)


def _inferred(runs: list[tuple[str, dict[str, Any]]]) -> Prefill:
    seen: list[dict[str, tuple[dict[str, Any], dict[str, Any]]]] = []  # per run: step id to (piece, setting)
    confidences, problems = [], []
    for where, doc in runs:
        try:
            inference = infer_detail(doc)
        except ValueError as exc:  # InferError too: nothing in the run says what ran
            problems.append(f"{where}: what ran cannot be read from run {doc['run']['id']} ({exc})")
            continue
        config, steps = inference.configuration, {}
        given = doc["run"]["configuration"]["settings"] if inference.inferred_from == "configuration" else None
        for piece in config.workflow.pieces:
            s = config.settings.get(piece.id) or piece.setting
            if s is None:
                problems.append(f"{where}: run {doc['run']['id']} gives no setting for its step {piece.id}")
                continue
            steps[piece.id] = ({"role": piece.role, "width": piece.width},
                               {"harness": s.harness, "model": s.model, "effort": s.effort,
                                "context_policy": s.context_policy,
                                "options": dict(s.options) if given is None
                                else copy.deepcopy((given.get(piece.id) or {}).get("options") or {})})
        seen.append(steps)
        confidences.append(inference.confidence)
    if problems:
        raise Refused(problems)
    order: list[str] = []
    for steps in seen:
        order += [sid for sid in steps if sid not in order]
    chosen, agreement = [], 1.0
    for sid in order:
        having = [steps[sid] for steps in seen if sid in steps]
        piece, setting = _mode([p for p, _ in having]), _mode([s for _, s in having])
        agreeing = sum(1 for p, s in having if _same(p, piece) and _same(s, setting))
        agreement = min(agreement, agreeing / len(seen))
        chosen.append({"id": sid, **copy.deepcopy(piece), **copy.deepcopy(setting)})
    confidence = round(sum(confidences) / len(confidences) * agreement, 3)
    return Prefill(INFERRED, tuple(doc["run"]["id"] for _, doc in runs), confidence, steps=tuple(chosen),
                   rounds=max(_rounds(doc) for _, doc in runs))


def prefill(runs: list[tuple[str, dict[str, Any]]]) -> Prefill:
    """What the runs `read` gave say: declared when they all declare one configuration, inferred when none does.
    Raises Refused for a declared catalog workflow reference, a mix, and declared runs of two configurations."""
    kinds = [(where, doc, declaration(doc["run"])) for where, doc in runs]
    references = [f"{where}: run {doc['run']['id']} declares its workflow by catalog reference "
                  f"({_one_line(json.dumps(doc['run']['configuration']['workflow'], ensure_ascii=False))}); prefill "
                  "takes a configuration with its workflow inline" for where, doc, kind in kinds if kind == "reference"]
    if references:
        raise Refused(references)
    declaring = [(where, doc) for where, doc, kind in kinds if kind == "declared"]
    if len(declaring) == len(runs):
        return _declared(declaring)
    if declaring:
        named: dict[Any, list[str]] = {}
        for where, doc in declaring:
            named.setdefault(doc["run"]["configuration"].get("id"), []).append(_named(where, doc))
        raise Refused([f"some runs declare their configuration and some do not; {ONE_CONFIGURATION}"]
                      + [f"{cid}: declared by {', '.join(names)}" for cid, names in named.items()]
                      + ["no configuration declared by " + ", ".join(_named(where, doc) for where, doc, kind in kinds
                                                                     if kind is None)])
    return _inferred(runs)


def missing_text(p: Prefill) -> str:
    """The keys the answers must give after prefill, in question order, each the runs leave open described."""
    described = {"steps": "steps (what each step reads and writes; the runs give each step's role and setting)",
                 "checks": f"checks (the gates, what each failure reruns, the rescue, and budget_rounds; the runs saw "
                           f"at most {counted(p.rounds, 'round')})"} if p.how == INFERRED else {}
    return ", ".join(described.get(k, k) for k in walk.keys(p))
