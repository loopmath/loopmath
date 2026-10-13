"""`loopmath outcome --rule review RUN...` (0.2.7): read runs by the review rule, or undo that.

The review rule (`belief.rules`) reads a run by its latest review or accepted verdict. The command moves a run to
`review` and saves the rule it had. `loopmath outcome --undo-rule RUN...` always restores the saved rule, as it was;
naming that rule (`tests` when it had none) does the same when its name is not `review`, since a run's own rule
can be named `review` too. It refuses a run with a `tests` verdict by name, since its tests decide it, and it
writes nothing unless every named run can move. Each move or undo adds a `note` event to the run. A finished run's
receipts are re-scored at once, as after a late signal; the caller refits once.
"""

from __future__ import annotations

import copy
from typing import Any

from ..belief import rules
from . import runs as R
from .home import NotFound, Store, StoreError

CHANGE_EXT = "dev.loopmath.rule_change"  # {"previous": the stored rule before the move, or None}
UNDO_COMMAND = "loopmath outcome --undo-rule"


def rule_name(rule: Any) -> str:
    """A stored rule's name as every reader reads it: a rule with no name is the default, `tests`."""
    return str(rule["name"]) if isinstance(rule, dict) and rule.get("name") else "tests"


def _shown(rule: Any) -> str:
    """A rule's name in a note or a line: a run's own rule named review is told apart from the review rule."""
    name = rule_name(rule)
    return "its own rule named review" if name == "review" and not rules.is_review_rule(rule) else name


def _saved(doc: dict[str, Any]) -> dict[str, Any] | None:
    """The move's record on a run the command moved and has not moved back, else None."""
    run = doc.get("run") or {}
    ext = run.get("ext") if isinstance(run.get("ext"), dict) else {}
    change = ext.get(CHANGE_EXT)
    return change if rules.is_review_rule(run.get("acceptance_rule")) and isinstance(change, dict) else None


def offered(doc: dict[str, Any]) -> list[str]:
    """The rule names the command offers for this run: `review`, and the saved previous rule's to undo the move
    when that name is not `review` (`--undo-rule` undoes it either way)."""
    saved = _saved(doc)
    previous = rule_name(saved.get("previous")) if saved is not None else "review"
    return ["review"] + ([previous] if previous != "review" else [])


def _plan(doc: dict[str, Any], name: str | None) -> tuple[str | None, str | None]:
    """(action, refusal) for one run: action `move`, `undo` or `unchanged`. `name` None is `--undo-rule`."""
    run_id = (doc.get("run") or {}).get("id")
    if name is None:
        if _saved(doc) is None:
            return None, f"run {run_id} has no rule to restore: `{rules.REVIEW_COMMAND}` has not moved it"
        return "undo", None
    if name == "review":
        if "tests" in {s.get("name") for s in rules._verdicts(doc)}:
            return None, f"run {run_id} has a tests verdict, so its tests decide it; it stays under its rule"
        return ("unchanged" if rules.is_review_rule((doc.get("run") or {}).get("acceptance_rule")) else "move"), None
    names = offered(doc)
    if name in names[1:]:
        return "undo", None
    return None, f"rule {name} is not offered for run {run_id} (offered: {', '.join(names)})"


def _write(doc: dict[str, Any], action: str) -> tuple[Any, str]:
    """Apply the move or the undo to the run file's document; returns the stored rule it replaced and the note's
    detail."""
    run = doc["run"]
    previous = copy.deepcopy(run.get("acceptance_rule"))
    if action == "move":
        run["acceptance_rule"] = rules.review_rule_dict()
        run.setdefault("ext", {})[CHANGE_EXT] = {"previous": previous}
        detail = f"acceptance rule review (was {_shown(previous)})"
    else:
        saved = run["ext"].pop(CHANGE_EXT)["previous"]
        if saved is None:
            run.pop("acceptance_rule", None)
        else:
            run["acceptance_rule"] = saved
        if not run["ext"]:
            del run["ext"]
        shown = _shown(saved)
        detail = f"acceptance rule {'' if shown == rule_name(saved) else 'back to '}{shown} (was review)"
    R.add_event(doc, "note", detail=detail)
    return previous, detail


def change_rule(store: Store, name: str | None, run_ids: list[str]) -> list[dict[str, Any]]:
    """Move every named run to `name`, or restore each one's saved rule when `name` is None (`--undo-rule`), all or
    nothing, under the store lock. Returns one entry per run: {run, action, rule, previous, restorable, finished,
    rescored, z, detail}, `rule` the name of the rule the run has now, `restorable` whether `--undo-rule` restores a
    saved rule and `detail` the note added (None when unchanged). Raises NotFound for an unknown run and StoreError
    for a refusal, naming every run that cannot move."""
    from .finish import rescore_run

    if not run_ids:
        raise StoreError(f"--undo-rule takes the runs to restore: {UNDO_COMMAND} RUN..." if name is None else
                         f"--rule {name} takes the runs to move: {rules.REVIEW_COMMAND} RUN...")
    with store.lock():
        missing, refused, plans = [], [], []
        for run_id in dict.fromkeys(run_ids):  # a run named twice moves once
            try:
                doc = store.run_doc(run_id)
            except NotFound:
                missing.append(run_id)
                continue
            action, refusal = _plan(doc, name)
            if refusal:
                refused.append(refusal)
            else:
                plans.append((run_id, action))
        if missing:
            raise NotFound("; ".join([f"no run {', '.join(missing)} in {store.home}"] + refused) + "; nothing written")
        if refused:
            raise StoreError("; ".join(refused) + "; nothing written")
        out = []
        for run_id, action in plans:
            entry: dict[str, Any] = {"run": run_id, "action": action, "rule": "review", "previous": None,
                                     "restorable": False, "finished": False, "rescored": [], "z": None, "detail": None}
            if action == "unchanged":
                doc = store.run_doc(run_id)
                entry.update(previous=doc["run"].get("acceptance_rule"), restorable=_saved(doc) is not None)
                out.append(entry)
                continue
            doc = store._read(run_id)
            entry["previous"], entry["detail"] = _write(doc, action)
            entry.update(rule=rule_name(doc["run"].get("acceptance_rule")), restorable=_saved(doc) is not None)
            store._write(doc)
            if R.is_finished(doc):
                store.flag_rescore(run_id)
                late = rescore_run(store, run_id)
                entry.update(finished=True, rescored=late["rescored"], z=late["z"])
            out.append(entry)
    return out
