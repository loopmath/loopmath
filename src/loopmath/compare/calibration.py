"""The adopt threshold, calibrated so that false adopts stay at or under a stated rate.

A posterior chance of 0.95 checked after every pair does not keep false adopts at 5%: each look is another
chance for noise to cross the line. So the comparison adopts at `p_better >= threshold`, where the threshold
was chosen by simulation (`calibrate.py`) for this look schedule:

- **Looks** after every pair, from the first, up to a cap fixed before the first look (`max_pairs`; a check's
  cap is its number of pairs). Any subset of those looks, and any stop that keeps the base (reject, the next
  pair's value, the budget), can only lower the false adopt rate, so the bound covers them too.
- **Null families**: the simulated cases where the candidate is no better, named in the table entry. The bound
  is claimed only for those families, at the default floor or above.
- **Tuning and checking**: thresholds are tuned on one seed as the least favourable family's
  (1 - margin * alpha) quantile of the largest P(better) seen up to the cap (the margin is in the entry); the
  rates reported beside each bound come from a held-out seed, with exact binomial 95% intervals.

A cap past the table, a floor under the default, or a confidence stricter than the table's alphas gets the
strictest threshold the table has and no stated bound.

The switched cost model (`model.COST_MODEL = "sw"`, an experimental setting, not the default) has its own table,
`_thresholds_sw.py`, calibrated for it; the default reads `_thresholds.py`.
"""

from __future__ import annotations

from typing import Any

from ..output import counted
from . import model as M

DEFAULT_CAP = 24  # the cap for calls that give none
DEFAULT_FLOORS = {"cpar": M.FLOOR_CPAR, "gain": M.FLOOR_GAIN, "score": M.FLOOR_SCORE}
LOOKS = "after every pair, from the first, up to the cap"

try:
    from ._thresholds import TABLE
except ImportError:  # pragma: no cover - the generated table ships with the package
    TABLE: dict[str, Any] = {}
try:
    from ._thresholds_sw import TABLE as TABLE_SW
except ImportError:  # pragma: no cover - the generated table ships with the package
    TABLE_SW: dict[str, Any] = {}


def table_in_use() -> dict[str, Any]:
    """The table calibrated for the cost model in use."""
    return TABLE_SW if M.COST_MODEL == "sw" else TABLE


def threshold(mode: str, *, confidence: float, floor: float, cap: int | None, n_pairs: int) -> dict[str, Any]:
    """{threshold, cap, bound, why} for `mode` (cpar, gain or score). The threshold depends only on the mode,
    the alpha and the cap, never on the pairs seen so far, which is what keeps the bound."""
    alpha = 1.0 - confidence
    cap = int(cap) if cap is not None else DEFAULT_CAP
    entry = table_in_use().get(mode)
    if not entry:
        return {"threshold": confidence, "cap": cap, "bound": None,
                "why": f"no false adopt bound is stated: there is no calibration for {mode}"}
    table = entry["thresholds"]
    use = max((a for a in table if float(a) <= alpha + 1e-9), key=float, default=None)
    caps = sorted(int(c) for c in table[use]) if use is not None else []
    tcap = min((c for c in caps if c >= cap), default=None)
    strictest = max(v for by_cap in table.values() for v in by_cap.values())
    if use is None or tcap is None:
        why = (f"confidence {confidence} is stricter than the table's" if use is None
               else f"the cap of {cap} pairs is past the table's {max(caps)}")
        return {"threshold": max(strictest, confidence), "cap": cap, "bound": None,
                "why": f"no false adopt bound is stated: {why}; adopting needs P(better) at or above "
                       f"{max(strictest, confidence):.3f}, the strictest calibrated threshold"}
    thr = max(float(table[use][str(tcap)]), confidence)
    held = entry["held_out"][use][str(tcap)]
    if floor < DEFAULT_FLOORS[mode] - 1e-12:
        return {"threshold": thr, "cap": cap, "bound": None,
                "why": f"no false adopt bound is stated: the floor {floor:g} is under the calibrated "
                       f"{DEFAULT_FLOORS[mode]:g}"}
    if n_pairs > cap:
        return {"threshold": thr, "cap": cap, "bound": None,
                "why": f"no false adopt bound is stated: {n_pairs} pairs is past the cap of {cap}"}
    if not held.get("verified"):
        return {"threshold": thr, "cap": cap, "bound": None,
                "why": f"no false adopt bound is stated: the held-out rate {held['rate']:.3f} for {held['family']} "
                       f"is over {float(use):g} at a cap of {tcap}"}
    bound = {"alpha": float(use), "cap": tcap, "looks": LOOKS, "families": list(entry["families"]),
             "held_out": dict(held), "tune_seed": entry["tune_seed"], "check_seed": entry["check_seed"],
             "sims": entry["sims"]}
    span = (f"{LOOKS} of {tcap} pairs" if cap == tcap else  # fewer looks than the table's cap keep its bound
            f"after every pair, from the first, up to this check's cap of {counted(cap, 'pair')} (the threshold "
            f"calibrated for {tcap} pairs, which covers fewer)")
    return {"threshold": thr, "cap": cap, "bound": bound,
            "why": f"adopt threshold {thr:.3f} keeps false adopts at or under {float(use):g} for looks {span}, "
                   f"for the {len(entry['families'])} simulated null families in bound.families; held out: "
                   f"{held['rate']:.3f} at worst (95% {held['lo']:.3f} to {held['hi']:.3f}, {held['sims']} checks)"}
