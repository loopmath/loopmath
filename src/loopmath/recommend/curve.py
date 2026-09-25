"""Success-cost curve rows, rescue cost and the default pick (spec 05, section 2).

`ell` is the expected dollars to an accepted result: `E[C_run] + (1 - g) C_rescue`.
The recommender sets `C_rescue` from config `rescue.kind` and always hands it to
the belief as `rescue_usd=`, so the belief's `ell` is the one
source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..types import CurveRow, Prediction
from .stats import p_at_least

LEVELS: tuple[int, ...] = (50, 70, 80, 90, 95, 99)
UNCERTAIN_BELOW = 0.8  # a row is uncertain when fewer than 80 percent of draws reach its level
RESCUE_KINDS = ("redo_usual", "person", "none")


@dataclass(frozen=True)
class Rescue:
    """C_rescue in dollars and tokens, and how it was set."""

    kind: str
    usd: float
    tokens: float
    basis: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "usd": self.usd, "tokens": self.tokens, "basis": self.basis}


class NoFiniteRescue(ValueError):
    """`redo_usual` when the usual workflow's chance is zero: repeating it never gets an accepted result."""


def rescue_cost(kind: str | None, usual: Prediction | None, *, person_usd_per_hour: float | None = None,
                hours: float | None = None) -> Rescue:
    """C_rescue for `rescue.kind`.

    - `redo_usual` (default): `E[C_run(usual)] / g(usual)`, the expected cost of an
      accepted result by repeating the usual workflow, for every positive `g`;
      `NoFiniteRescue` (a ValueError, exit 1) when `g` is zero;
    - `person`: `person_usd_per_hour` times `hours` (default 1); no tokens;
    - `none`: zero, so success is shown but not priced.
    """
    kind = kind or "redo_usual"
    if kind == "none":
        return Rescue("none", 0.0, 0.0, "not priced")
    if kind == "person":
        if person_usd_per_hour is None:
            raise ValueError("rescue.kind is person but rescue.person_usd_per_hour is not set")
        h = 1.0 if hours is None else float(hours)
        rate = float(person_usd_per_hour)
        return Rescue("person", rate * h, 0.0, f"{h:g} h at ${rate:g}/h")
    if kind != "redo_usual":
        raise ValueError(f"unknown rescue.kind {kind!r}; expected one of {', '.join(RESCUE_KINDS)}")
    if usual is None:
        raise ValueError("rescue.kind redo_usual needs the usual workflow's prediction")
    g = usual.p_success.mean
    if not g > 0:
        raise NoFiniteRescue("rescue.kind redo_usual has no finite cost here: the usual workflow has no chance of "
                             "an accepted result under this rule; set rescue.kind to person or none")
    return Rescue("redo_usual", usual.cost.usd.mean / g, usual.cost.tokens.mean / g,
                  "usual workflow repeated until accepted")


def ell_key(pred: Prediction) -> tuple:
    return (pred.ell.usd.mean, pred.cost.usd.mean, pred.config)


def default_pick(preds: Sequence[Prediction]) -> Prediction:
    """`argmin ell`: the lowest expected dollars to an accepted result."""
    if not preds:
        raise ValueError("no candidates to rank")
    return min(preds, key=ell_key)


DrawShare = Callable[[Prediction, float], float]


def curve(preds: Sequence[Prediction], *, share: DrawShare | None = None,
          levels: Sequence[int] = LEVELS) -> list[CurveRow]:
    """One row per level X: `argmin E[C_run]` subject to mean `g >= X/100`.

    `share(pred, x)` is the share of draws with `g >= x`; without it the share is
    read from the 80 percent interval. A row is `uncertain` when the share is
    under 0.8 and `reached: false` when no candidate has mean `g >= x`. Adjacent
    rows with the same configuration merge; a merged row is uncertain when any of
    its levels is (that is, when its highest level is).
    """
    share = share or (lambda p, x: p_at_least(p.p_success, x))
    rows: list[CurveRow] = []
    for level in levels:
        x = level / 100.0
        ok = [p for p in preds if p.p_success.mean >= x]
        if not ok:
            rows.append(CurveRow((level,), None, False, True, None))
            continue
        best = min(ok, key=lambda p: (p.cost.usd.mean, p.ell.usd.mean, p.config))
        uncertain = share(best, x) < UNCERTAIN_BELOW
        last = rows[-1] if rows else None
        if last is not None and last.config == best.config:
            rows[-1] = CurveRow(last.levels + (level,), best.config, True, last.uncertain or uncertain, best)
        else:
            rows.append(CurveRow((level,), best.config, True, uncertain, best))
    return rows


def goal_row(rows: Sequence[CurveRow], level: int) -> tuple[CurveRow | None, str | None]:
    """The row holding `level`; when it is not reached, the highest reached row below it."""
    for row in rows:
        if level in row.levels and row.reached:
            return row, None
    below = [r for r in rows if r.reached and max(r.levels) < level]
    if below:
        row = below[-1]
        return row, f"no workflow reaches {level}%, so the goal is the {max(row.levels)}% row"
    return None, f"no workflow reaches {level}%, so the goal is the default pick"


def parse_goal(choice: str | None) -> int | None:
    """`p80` to 80; `default` or None to None."""
    if choice in (None, "", "default"):
        return None
    text = str(choice).strip().lower()
    if text.startswith("p") and text[1:].isdigit() and int(text[1:]) in LEVELS:
        return int(text[1:])
    raise ValueError(f"goal must be default or one of {', '.join('p%d' % x for x in LEVELS)}; got {choice!r}")
