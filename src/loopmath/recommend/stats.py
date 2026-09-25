"""Small numeric helpers on 80 percent intervals (spec 03 `Interval`).

`Prediction` carries each quantity as a mean and its 10th and 90th percentiles,
not the belief's 400 draws. When the belief does not hand draws over,
the recommender reads a distribution back from the interval: logit-normal for
chances, log-normal for a receipt's cost surprise. These are stated
approximations.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from ..types import Interval, Money

_N = NormalDist()
Z80 = _N.inv_cdf(0.9)  # half-width of an 80 percent interval in standard deviations
EPS = 1e-4


def logit(p: float) -> float:
    p = min(1.0 - EPS, max(EPS, p))
    return math.log(p / (1.0 - p))


def p_at_least(iv: Interval, x: float) -> float:
    """P(p >= x) for a chance whose 80 percent interval is `iv` (logit-normal)."""
    if x <= 0.0:
        return 1.0
    if x >= 1.0:
        return 0.0
    lo, hi = logit(iv.lo), logit(iv.hi)
    if hi - lo < 1e-9:
        return 1.0 if iv.mean >= x else 0.0
    center = (lo + hi) / 2.0
    sigma = (hi - lo) / (2.0 * Z80)
    return _N.cdf((center - logit(x)) / sigma)


def log_residual(value: float, iv: Interval) -> float | None:
    """Standardized residual of a positive value on the log scale of a positive interval."""
    if value <= 0 or iv.lo <= 0 or iv.hi <= 0:
        return None
    lo, hi = math.log(iv.lo), math.log(iv.hi)
    center = (lo + hi) / 2.0
    sigma = (hi - lo) / (2.0 * Z80)
    if sigma < 1e-12:
        return None
    return (math.log(value) - center) / sigma


def money_dict(m: Money) -> dict:
    return {"usd": m.usd.to_dict(), "tokens": m.tokens.to_dict()}
