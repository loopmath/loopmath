"""Observe, holdout, and reveal mask grammar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fit_pricing import EFFORT_ORDER

__all__ = [
    "MODEL_SHORT", "_FULL_MODEL_NAMES", "E1A_OBSERVE", "E1B_REVEAL", "Mask",
    "_resolve_model_name", "_parse_cell_groups", "_parse_reveal_groups",
    "parse_mask", "apply_mask",
]


MODEL_SHORT = {
    "luna": "gpt-5.6-luna", "sol": "gpt-5.6-sol", "terra": "gpt-5.6-terra",
    "opus": "claude-opus-5", "sonnet": "claude-sonnet-5", "fable": "claude-fable-5",
}
_FULL_MODEL_NAMES = set(MODEL_SHORT.values())

E1A_OBSERVE = "luna:low,medium,xhigh;sol:medium;terra:medium"
E1B_REVEAL = "t7:opus,sonnet,fable"


@dataclass
class Mask:
    observed_cells: set[tuple[str, str]]     # (model, effort)
    holdout: str                             # "rest" or an explicit spec
    task_reveal: dict[str, set[str]]         # task_group -> set of model names additionally observed
    spec: str                                # the original strings, for the record


def _resolve_model_name(raw: str) -> str:
    name = raw.strip()
    if not name:
        raise ValueError("empty model name in mask spec")
    if name in MODEL_SHORT:
        return MODEL_SHORT[name]
    if name in _FULL_MODEL_NAMES:
        return name
    known = sorted(MODEL_SHORT) + sorted(_FULL_MODEL_NAMES)
    raise ValueError(f"unknown model '{raw}' in mask spec (known names: {known})")


def _parse_cell_groups(spec_str: str) -> set[tuple[str, str]]:
    """Parse `model:effort,effort,...;model:effort,...` into (model, effort) cells."""
    cells: set[tuple[str, str]] = set()
    spec_str = (spec_str or "").strip()
    if not spec_str:
        return cells
    for group in spec_str.split(";"):
        group = group.strip()
        if not group:
            continue
        if ":" not in group:
            raise ValueError(
                f"malformed mask group '{group}': expected 'model:effort,effort,...'"
            )
        model_part, effort_part = group.split(":", 1)
        model = _resolve_model_name(model_part)
        effort_part = effort_part.strip()
        if effort_part == "*":
            efforts = list(EFFORT_ORDER)
        else:
            efforts = [e.strip() for e in effort_part.split(",") if e.strip()]
            if not efforts:
                raise ValueError(f"malformed mask group '{group}': no efforts listed")
            for e in efforts:
                if e not in EFFORT_ORDER:
                    raise ValueError(
                        f"unknown effort '{e}' in mask group '{group}' "
                        f"(known efforts: {list(EFFORT_ORDER)})"
                    )
        for e in efforts:
            cells.add((model, e))
    return cells


def _parse_reveal_groups(spec_str: str) -> dict[str, set[str]]:
    """Parse `taskgroup:model,model,...;taskgroup:model,...` into a reveal map."""
    reveal: dict[str, set[str]] = {}
    spec_str = (spec_str or "").strip()
    if not spec_str:
        return reveal
    for group in spec_str.split(";"):
        group = group.strip()
        if not group:
            continue
        if ":" not in group:
            raise ValueError(
                f"malformed reveal group '{group}': expected 'taskgroup:model,model,...'"
            )
        task_group_part, model_part = group.split(":", 1)
        task_group = task_group_part.strip()
        if not task_group:
            raise ValueError(f"malformed reveal group '{group}': missing task group")
        models = [_resolve_model_name(m) for m in model_part.split(",") if m.strip()]
        if not models:
            raise ValueError(f"malformed reveal group '{group}': no models listed")
        reveal.setdefault(task_group, set()).update(models)
    return reveal


def parse_mask(observe: str, holdout: str = "rest", reveal: str | None = None) -> Mask:
    """Parse the `--observe` / `--holdout` / `--reveal` grammar into a `Mask`.

    `observe` and an explicit `holdout` share the same grammar:
    semicolon-separated `model:effort,effort,...` groups, short (`MODEL_SHORT`)
    or full model names, `*` meaning every effort. `holdout="rest"` (the
    default) means every cell not named by `observe` is held out; anything
    else is parsed with the same grammar and treated as an explicit holdout
    set. `reveal` is `taskgroup:model,model,...` groups (see `E1B_REVEAL`):
    rows matched by a reveal are observed even when their (model, effort)
    cell would otherwise be held out. Raises `ValueError` with a clear
    message on an unknown model name or a malformed group.
    """
    observed_cells = _parse_cell_groups(observe)
    task_reveal = _parse_reveal_groups(reveal) if reveal else {}
    spec = f"observe={observe!r} holdout={holdout!r} reveal={reveal!r}"
    return Mask(observed_cells=observed_cells, holdout=holdout, task_reveal=task_reveal, spec=spec)


def apply_mask(df: pd.DataFrame, mask: Mask) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (observed, heldout). Every row lands in exactly one; assert that."""
    if df.empty:
        empty = df.copy()
        return empty, empty.copy()

    cells = list(zip(df["model"], df["effort"]))

    if mask.holdout == "rest":
        base_observed = np.array([c in mask.observed_cells for c in cells])
    else:
        holdout_cells = _parse_cell_groups(mask.holdout)
        in_observe = np.array([c in mask.observed_cells for c in cells])
        in_holdout = np.array([c in holdout_cells for c in cells])
        # Explicit holdout: anything not named by either spec defaults to
        # held out (the observe list is read as a whitelist of what to
        # reveal, consistent with the "rest" default).
        base_observed = in_observe & ~in_holdout

    if mask.task_reveal:
        revealed = np.array([
            row_model in mask.task_reveal.get(row_task_group, ())
            for row_model, row_task_group in zip(df["model"], df["task_group"])
        ])
    else:
        revealed = np.zeros(len(df), dtype=bool)

    is_observed = base_observed | revealed
    observed = df[is_observed]
    heldout = df[~is_observed]

    assert len(observed) + len(heldout) == len(df)
    assert set(observed.index).isdisjoint(set(heldout.index))
    return observed, heldout
