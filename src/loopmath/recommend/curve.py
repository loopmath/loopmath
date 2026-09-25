"""Success-cost curve rows, rescue cost and the default pick (spec 05, section 2).

`ell` is the expected dollars to an accepted result: `E[C_run] + (1 - g) C_rescue`.
The recommender sets `C_rescue` from config `rescue.kind` and always hands it to
the belief as `rescue_usd=`, so the belief's `ell` is the one
source of truth.

`retry` (the default, 0.2.1, F7 and T2): a miss is fixed by retrying with the rescue workflow, each
further attempt on the same task with `rescue.decay` times the chance of the one before, up to
`rescue.max_attempts` attempts in all counting the first run (`retry_rescue`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..types import CurveRow, Prediction
from .stats import p_at_least

LEVELS: tuple[int, ...] = (50, 70, 80, 90, 95, 99)
UNCERTAIN_BELOW = 0.8  # a row is uncertain when fewer than 80 percent of draws reach its level
RESCUE_KINDS = ("retry", "redo_usual", "person", "none")
RETRY_DECAY = 0.5  # rescue.decay
RETRY_MAX_ATTEMPTS = 3  # rescue.max_attempts, counting the first run
RETRY_MIN_CHANCE = 0.70  # rescue.min_chance


@dataclass(frozen=True)
class Rescue:
    """C_rescue in dollars and tokens, and how it was set."""

    kind: str
    usd: float
    tokens: float
    basis: str = ""
    # `retry` only: the rescue workflow and the retry model (spec 05 section 2)
    config: str | None = None
    chance: dict[str, float] | None = None  # the rescue workflow's chance in one run: mean, lo, hi
    run_cost_usd: float | None = None
    decay: float | None = None
    max_attempts: int | None = None
    min_chance: float | None = None
    p_accepted: float | None = None  # mean over draws of the chance the retries fix a miss

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"kind": self.kind, "usd": self.usd, "tokens": self.tokens, "basis": self.basis}
        if self.kind == "retry":
            out.update({"config": self.config, "chance": self.chance, "run_cost_usd": self.run_cost_usd,
                        "decay": self.decay, "max_attempts": self.max_attempts, "min_chance": self.min_chance,
                        "p_accepted": self.p_accepted})
        return out

    @property
    def fixes(self) -> float:
        """The chance that the rescue fixes a miss within the attempts it models: `p_accepted` for `retry`,
        0 for the other kinds (they model no bounded retries, so `p_accepted_within` is the one-run chance)."""
        return float(self.p_accepted or 0.0) if self.kind == "retry" else 0.0

    @property
    def attempts(self) -> int:
        return int(self.max_attempts or 1) if self.kind == "retry" else 1


class NoFiniteRescue(ValueError):
    """`redo_usual` when the usual workflow's chance is zero: repeating it never gets an accepted result."""


def retry_rescue(q: Sequence[float], c: Sequence[float], t: Sequence[float] | None = None, *,
                 decay: float = RETRY_DECAY, max_attempts: int = RETRY_MAX_ATTEMPTS) -> tuple[float, float, float]:
    """`(C_rescue usd, C_rescue tokens, p_accepted)` for `retry` from the rescue workflow's draws: chance `q_s`,
    run cost `c_s` and tokens `t_s`. With K = `max_attempts` and d = `decay`, retry j = 1 .. K-1 has chance
    `q_s d^j` and runs only when the retries before it missed:

        spend_s = sum_{j=1..K-1} c_s prod_{i<j} (1 - q_s d^i)
        succ_s  = 1 - prod_{j=1..K-1} (1 - q_s d^j)
        C_rescue = mean_s(spend_s) / mean_s(succ_s)

    dollars per accepted rescue across tasks, a ratio of means, so draws near zero chance do not blow it up.
    `p_accepted` is `mean_s(succ_s)`. K = 1 means no retries: all three are zero. `NoFiniteRescue` when
    no draw can fix a miss."""
    k = int(max_attempts)
    d = float(decay)
    if not 0 < d <= 1:
        raise ValueError(f"rescue.decay must be above 0 and at most 1; got {decay!r}")
    if k < 1:
        raise ValueError(f"rescue.max_attempts must be a whole number of at least 1; got {max_attempts!r}")
    n = len(q)
    if n == 0 or len(c) != n or (t is not None and len(t) != n):
        raise ValueError("retry_rescue needs one run cost (and token count) per chance draw")
    if k == 1:
        return 0.0, 0.0, 0.0
    spend = spend_t = succ = 0.0
    for s in range(n):
        qs = min(1.0, max(0.0, float(q[s])))
        miss = 1.0
        paid = 0.0
        for j in range(1, k):
            paid += miss
            miss *= 1.0 - qs * d ** j
        spend += float(c[s]) * paid
        spend_t += (float(t[s]) if t is not None else 0.0) * paid
        succ += 1.0 - miss
    if not succ > 0:
        raise NoFiniteRescue("rescue.kind retry has no finite cost here: the rescue workflow has no chance of "
                             "an accepted result under this rule; set rescue.kind to person or none")
    return spend / succ, spend_t / succ, succ / n


def accepted_within(g: float, rescue: "Rescue") -> float:
    """`p_accepted_within`: `g + (1 - g) p_fix`, the chance of an accepted result within the rescue's attempts.
    Affine and increasing in `g`, so a draw interval of `g` maps to its interval end by end."""
    p = rescue.fixes
    return g + (1.0 - g) * p


def rescue_cost(kind: str | None, usual: Prediction | None, *, person_usd_per_hour: float | None = None,
                hours: float | None = None) -> Rescue:
    """C_rescue for `rescue.kind` other than `retry` (the recommender prices that one from the rescue
    workflow's draws, `retry_rescue`).

    - `redo_usual`: `E[C_run(usual)] / g(usual)`, the expected cost of an
      accepted result by repeating the usual workflow, for every positive `g`;
      `NoFiniteRescue` (a ValueError, exit 1) when `g` is zero;
    - `person`: `person_usd_per_hour` times `hours` (default 1); no tokens;
    - `none`: zero, so success is shown but not priced.
    """
    kind = kind or "redo_usual"
    if kind == "retry":
        raise ValueError("rescue.kind retry is priced by the recommender (`retry_rescue`), not `rescue_cost`")
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
