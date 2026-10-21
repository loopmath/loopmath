"""The paired comparison's statistics: a base arm against a candidate arm on the same tasks.

Every model is conjugate or sampled directly, so a comparison costs milliseconds. Each one is proper at any
number of pairs, and a summary's mean is reported only when the posterior mean exists. The models, with their
priors:

- **A binary outcome** (accepted under the rule). The pairs both of whose runs were scored fall in four
  cells: both accepted, the candidate only, the base only, neither. The cell chances get a Dirichlet(1/2)
  prior, so each arm's chance of acceptance starts uniform on 0 to 1 (Beta(1, 1)). The gain is the
  candidate's chance minus the base's: the candidate-only cell minus the base-only cell. A scored run whose
  partner is missing or unscored still tells about its own arm: Dirichlet draws are reweighted by its
  Bernoulli likelihood and resampled (`OVERSAMPLE` times as many draws; the effective size is kept).
- **A score** (a score rule). The paired differences d, signed so that positive is better, are normal with
  mean mu and sd sigma. mu has a flat prior; sigma^2 a scaled inverse chi-square prior with `SCORE_NU` = 1
  degree of freedom and scale s0, the spread of all the scores in the scored pairs (both arms pooled). The
  prior is worth one pair, so it is proper for one pair and for identical differences, and it is scale
  aware. mu's posterior is Student t with 1 + n - 1 = n degrees of freedom: its mean exists from two pairs.
  P(better) uses the exact t distribution function.
- **Cost per run.** Log dollars of the two runs of a pair are bivariate normal: flat prior on the means, an
  inverse Wishart prior on the covariance with `COST_NU` = 3 degrees of freedom and scale `COST_SD`^2 times
  the identity (proper; its correlation is uniform on -1 to 1). The posterior from the pairs with both costs
  known is sampled exactly (Bartlett), truncated to arm sds within `SIGMA_RANGE` by rejection, which keeps
  every mean cost exp(mu + sigma^2 / 2) finite. A costed run without a costed partner reweights the draws by
  its normal likelihood. With no costed pair yet, each arm is sampled on its own from the matching marginal
  (inverse gamma for sigma^2), with the correlation drawn from its prior. When an arm's costed runs mix runs
  recorded at $0 (a tiny positive amount in log dollars) with priced runs, no cost range is stated
  (`zero_beside_priced`): that one log far below the others spreads the arm's costs, and its mean cost then
  grows with the pairs.
- **A switched cost model: an experimental setting, not the default.** `COST_MODEL = "sw"` keeps the
  lognormal above unless the pairs' mean log costs favour two normals over one (a BIC margin over `MIX_TAU`),
  every pair orders the arms one way and the lognormal's means the other way, or the lognormal states no
  range. Then each arm's mean cost comes from a Dirichlet process over the observed runs (`COST_PART`: its
  base measure, alpha and tau), whose draws keep every pair's order. That suits arms that mix cheap and dear
  models. Its ranges cover a true cost ratio of 1 poorly where it answers at few pairs: when every pair
  orders the arms one way and there are no other runs, a range that keeps that order cannot hold a ratio
  of 1. The default stays the lognormal above (`COST_MODEL = "ln"`); "avg", averaging the two, is kept for
  comparison only.
- **Cost per accepted result** (cpar), loopmath's default goal: the dollars to an accepted result when a run
  that is not accepted is run again with the same arm, mean cost over chance (Wald's identity). Its
  posterior mean exists only when the arm has at least one accepted run: with none, E[1/p] diverges, so the
  mean is null while the median and the ranges stay.

**The decision.** The objective is cpar (default for a binary rule; the improvement is 1 - cpar_cand /
cpar_base) or the gain (a score rule, or `objective="gain"`). `p_better` is the chance that the improvement
is above the floor. Adopt at `p_better >= threshold`, where the threshold is calibrated by simulation for
the look schedule and the cap (`calibration.py`). Reject at `p_better <= 1 - confidence`. Otherwise run more
pairs while the next pair is worth its price and the budget allows; when it is not, keep the base (reject).
Every stop other than adopt keeps the base, so it can only lower the false adopt rate.

**The value of the next pairs** is `compare/value.py`'s (`pairs_value`): the expected dollars over
`horizon_runs` future runs of going on as the procedure does (check-change's look after every pair, or the
gate's looks and guards) against stopping now, on futures drawn from the same posterior that gave the decision
(including the reweighting by lone runs), each action scored against the world its future was drawn from.
`draw_future` and `extend` below are its parts. A mean cost exp(mu + sigma^2 / 2) is heavy-tailed in sigma, so
a few futures carry much of the value; its Monte Carlo error is reported, and a stop needs it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import expit, stdtr

DRAWS = 4000
CONFIDENCE = 0.95
FLOOR_CPAR = 0.05  # the candidate must cut dollars per accepted result by 5% to count as better
FLOOR_GAIN = 0.05  # or raise the chance of acceptance by 5 points
FLOOR_SCORE = 0.0  # a score's floor, in its units
HORIZON_RUNS = 50  # future similar runs the decision applies to, for the next pair's value
MIN_PAIRS_VALUE = 4  # with the value stop on, scored pairs before the next pair's value may stop a check: on
#                      fewer, the value stop ended checks of a real cut after one or two pairs (0.3.1)
SIGMA_RANGE = (0.05, 3.0)  # the log-cost sds' prior support
COST_NU = 3.0  # inverse Wishart degrees of freedom: the weakest proper prior in two dimensions
COST_SD = 0.5  # its scale: a log-cost sd of 0.5 (dollars within a factor of 1.6)
COST_MODEL = "ln"  # the default, the lognormal alone; "sw", switched to the DP part, and "avg", averaged with
#                    it, are experimental settings, not the default
COST_PART = ("extra", 2.0, 1.0)  # "sw" and "avg" only: the DP part's base measure ("level", "paired" or
#                                  "extra"), alpha and tau
MIX_TAU = 6.0  # sw: the BIC margin for two normals over one on the pairs' mean log costs
MIX_ITERS = 60  # EM steps from each start of the mixture test
MIX_MIN = 6  # pairs before the mixture test is made
DP_MIN_WEIGHT = 1e-3  # avg: below this weight the DP part is not drawn
DP_ATOMS = 24  # the fewest sticks of a DP draw
DOM_MIN_PAIRS = 1  # pairs from which unanimous pairs overrule a contradicting mean
SCORE_NU = 1.0  # the score differences' variance prior is worth one pair
OVERSAMPLE = 8  # draws per kept draw when lone runs reweight them
LOOK_STEPS = (1, 2, 4, 8, 16)
LOOK_SIMS = 256  # simulated futures, each read at every step count
LOOK_DRAWS = 800  # posterior draws per simulated future
DIRICHLET_PRIOR = 0.5
CELLS = ("both", "cand_only", "base_only", "neither")
ARMS = ("base", "cand")


# ---------------------------------------------------------------- summaries
def interval(draws: np.ndarray | None, *, mean: bool = True, value: float | None = None,
             nd: int = 6) -> dict[str, Any] | None:
    """{mean, median, lo, hi, bands {"80", "95"}}: lo and hi are the 80% range, as loopmath's intervals.
    `mean=False` when the posterior mean does not exist (it is then None, whatever the finite draws say);
    `value` gives an exact mean instead of the draws' average."""
    if draws is None:
        return None
    a = np.asarray(draws, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    a = np.where(np.isnan(a), np.inf, a)
    q = np.percentile(a, [2.5, 10, 50, 90, 97.5], method="nearest")  # no interpolation across an infinite draw
    m = None
    if mean and finite.size == a.size:
        m = float(value) if value is not None else float(np.mean(a))

    def r(v: float | None) -> float | None:
        return round(float(v), nd) if v is not None and math.isfinite(v) else None

    return {"mean": r(m), "median": r(q[2]), "lo": r(q[1]), "hi": r(q[3]),
            "bands": {"80": [r(q[1]), r(q[3])], "95": [r(q[0]), r(q[4])]}}


def score_scale(values: list[float]) -> float:
    """The score prior's scale s0: the sd of all the scores in the scored pairs, both arms pooled; for
    identical scores, a thousandth of their size (at least 1e-3)."""
    v = np.asarray(values, dtype=float)
    if v.size >= 2:
        s = float(v.std(ddof=1))
        if s > 0:
            return s
    return 1e-3 * max(float(np.abs(v).mean()) if v.size else 0.0, 1.0)


# ---------------------------------------------------------------- data in sufficient statistics
@dataclass
class Data:
    """What the models read: cell counts of fully scored pairs, the unpaired scored runs, paired score
    differences with the score prior's scale, and the log costs, in pairs and alone."""

    kind: str = "binary"  # binary | score
    cells: np.ndarray = field(default_factory=lambda: np.zeros(4))  # both, cand_only, base_only, neither
    lone: dict[str, tuple[int, int]] = field(default_factory=lambda: {"base": (0, 0), "cand": (0, 0)})  # (acc, not)
    diffs: list[float] = field(default_factory=list)
    scale: float = 1.0  # score_scale of the scored pairs' scores
    cost_pairs: list[tuple[float, float]] = field(default_factory=list)  # (log base, log cand)
    cost_lone: dict[str, list[float]] = field(default_factory=lambda: {"base": [], "cand": []})
    # (runs recorded at $0, runs with a cost) per arm, as compare counted them; a drawn future keeps these
    cost_zero: dict[str, tuple[int, int]] = field(default_factory=lambda: {"base": (0, 0), "cand": (0, 0)})

    def copy(self) -> "Data":
        return Data(self.kind, self.cells.copy(), dict(self.lone), list(self.diffs), self.scale,
                    list(self.cost_pairs), {k: list(v) for k, v in self.cost_lone.items()}, dict(self.cost_zero))

    def successes(self, arm: str) -> int:
        """Accepted runs of the arm, paired and lone: the cpar mean exists only from one."""
        c = self.cells
        paired = c[0] + (c[2] if arm == "base" else c[1])
        return int(paired) + int(self.lone[arm][0])

    def costed(self, arm: str) -> bool:
        return bool(self.cost_pairs) or bool(self.cost_lone[arm])


# ---------------------------------------------------------------- posteriors
@dataclass
class Posterior:
    """Joint draws of every parameter, row i of each array from one draw."""

    n: int
    theta: np.ndarray | None = None  # (n, 4) cell chances
    mu: np.ndarray | None = None  # (n,) score: mean difference
    sd: np.ndarray | None = None  # (n,) score: sd of a difference
    t: tuple[float, float, float] | None = None  # score: mu's t marginal (df, location, scale)
    cmu: np.ndarray | None = None  # (n, 2) log-cost means (base, cand); NaN for an arm without costs
    ccov: np.ndarray | None = None  # (n, 2, 2)
    cost: "CostDraws | None" = None  # COST_MODEL "sw" or "avg": the mean costs, and how each draw's runs are drawn
    ess: dict[str, float] = field(default_factory=dict)  # effective sample sizes of reweighted draws
    cost_refused: str | None = None  # why there are no cost draws although there are costs

    def chance(self, arm: str) -> np.ndarray | None:
        if self.theta is None:
            return None
        return self.theta[:, 0] + self.theta[:, 2 if arm == "base" else 1]

    def mean_cost(self, arm: str) -> np.ndarray | None:
        a = ARMS.index(arm)
        if self.cost is not None:
            m = self.cost.mean[:, a]
            return m if np.isfinite(m[0]) else None
        if self.cmu is None:
            return None
        if not np.isfinite(self.cmu[0, a]):
            return None
        return np.exp(self.cmu[:, a] + self.ccov[:, a, a] / 2)


def _resample(logw: np.ndarray, rng: np.random.Generator, n: int) -> tuple[np.ndarray, float]:
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return rng.choice(w.size, size=n, replace=True, p=w), float(1.0 / np.sum(w ** 2))


def binary_posterior(data: Data, rng: np.random.Generator, n: int) -> tuple[np.ndarray, float | None]:
    """Cell chance draws: Dirichlet(1/2 + cells), reweighted by the lone runs' Bernoulli likelihoods."""
    (ab, rb), (ac, rc) = data.lone["base"], data.lone["cand"]
    extra = ab + rb + ac + rc
    theta = rng.dirichlet(DIRICHLET_PRIOR + np.asarray(data.cells, dtype=float), size=n * OVERSAMPLE if extra else n)
    if not extra:
        return theta, None
    pb = np.clip(theta[:, 0] + theta[:, 2], 1e-12, 1 - 1e-12)
    pc = np.clip(theta[:, 0] + theta[:, 1], 1e-12, 1 - 1e-12)
    logw = ab * np.log(pb) + rb * np.log1p(-pb) + ac * np.log(pc) + rc * np.log1p(-pc)
    idx, ess = _resample(logw, rng, n)
    return theta[idx], ess


def score_t(data: Data) -> tuple[float, float, float] | None:
    """mu's posterior t marginal (df, location, scale), or None with no scored pair."""
    k = len(data.diffs)
    if k == 0:
        return None
    d = np.asarray(data.diffs, dtype=float)
    ss = float(((d - d.mean()) ** 2).sum())
    nu = SCORE_NU + k - 1
    s2 = (SCORE_NU * data.scale ** 2 + ss) / nu
    return nu, float(d.mean()), math.sqrt(s2 / k)


def score_posterior(data: Data, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Joint draws of (mu, sigma): sigma^2 from its scaled inverse chi-square posterior, then mu given sigma."""
    t = score_t(data)
    if t is None:
        return None
    nu, loc, scale = t
    k = len(data.diffs)
    sig2 = nu * (scale ** 2 * k) / rng.chisquare(nu, size=n)
    return loc + np.sqrt(sig2 / k) * rng.standard_normal(n), np.sqrt(sig2)


def _inv_wishart2(nu: float, psi: np.ndarray, rng: np.random.Generator, m: int) -> np.ndarray:
    """m draws of a 2 x 2 inverse Wishart(nu, psi), by Bartlett's decomposition of its inverse."""
    chol = np.linalg.cholesky(np.linalg.inv(psi))
    a = np.zeros((m, 2, 2))
    a[:, 0, 0] = np.sqrt(rng.chisquare(nu, size=m))
    a[:, 1, 1] = np.sqrt(rng.chisquare(nu - 1, size=m))
    a[:, 1, 0] = rng.standard_normal(m)
    la = np.matmul(chol, a)
    w = np.matmul(la, la.transpose(0, 2, 1))
    det = w[:, 0, 0] * w[:, 1, 1] - w[:, 0, 1] ** 2
    out = np.empty_like(w)
    out[:, 0, 0], out[:, 1, 1] = w[:, 1, 1] / det, w[:, 0, 0] / det
    out[:, 0, 1] = out[:, 1, 0] = -w[:, 0, 1] / det
    return out


class CostRange(ValueError):
    """No draw of the truncated cost model was found: the costs lie far outside the sds it allows."""


COST_RANGE_WHY = (f"the log costs spread far outside the cost model's range (a log-cost sd of {SIGMA_RANGE[0]:g} "
                  f"to {SIGMA_RANGE[1]:g}), so no cost range is stated")


def zero_beside_priced(data: Data) -> str | None:
    """Why the lognormal states no cost range when an arm's costs mix runs recorded at $0 with priced runs, or
    None. A $0 run enters the log costs at a tiny positive amount, far below a priced run's log, and the spread it
    makes grows the arm's mean cost with the pairs."""
    mixed = [(word, z, n) for arm, word in (("base", "base"), ("cand", "candidate"))
             for z, n in [data.cost_zero.get(arm, (0, 0))] if 0 < z < n]
    if not mixed:
        return None
    who = " and ".join(f"the {word}'s" for word, _, _ in mixed)
    at = " and ".join(f"{z} of {n}" for _, z, n in mixed)
    return (f"{who} recorded costs mix $0 and priced runs ({at} at $0); a $0 run among priced ones spreads the "
            f"log costs far outside what the cost model reads, so no cost range is stated")


def _in_range(cov: np.ndarray) -> np.ndarray:
    """The prior's support: both sds in SIGMA_RANGE and the matrix positive definite."""
    lo, hi = SIGMA_RANGE[0] ** 2, SIGMA_RANGE[1] ** 2
    v0, v1 = cov[:, 0, 0], cov[:, 1, 1]
    return (v0 >= lo) & (v0 <= hi) & (v1 >= lo) & (v1 <= hi) & (v0 * v1 - cov[:, 0, 1] * cov[:, 1, 0] > 0)


def _normal_loglik(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Summed normal log likelihood of the values x under each draw (mu, var)."""
    if x.size == 0:
        return np.zeros(mu.size)
    s1, s2 = float(x.sum()), float((x ** 2).sum())
    return -0.5 * x.size * np.log(var) - (s2 - 2 * mu * s1 + x.size * mu ** 2) / (2 * var)


def _truncated_inv_gamma(shape: float, rate: float, rng: np.random.Generator, n: int) -> tuple[np.ndarray, int]:
    """n draws and the number of distinct ones: exact draws by rejection, resampled when fewer than n were
    kept, and CostRange when none was."""
    lo, hi = SIGMA_RANGE[0] ** 2, SIGMA_RANGE[1] ** 2
    kept = np.empty(0)
    for _ in range(20):
        v = rate / rng.gamma(shape, size=2 * n)
        kept = np.concatenate([kept, v[(v >= lo) & (v <= hi)]])
        if kept.size >= n:
            return kept[:n], n
    if kept.size == 0:
        raise CostRange(COST_RANGE_WHY)
    return rng.choice(kept, size=n), kept.size


def cost_posterior(data: Data, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, float | None] | None:
    """Joint draws of the log-cost means (n, 2) and covariances (n, 2, 2); NaN for an arm with no costs."""
    xy = np.asarray(data.cost_pairs, dtype=float).reshape(-1, 2)
    lone = [np.asarray(data.cost_lone[a], dtype=float) for a in ARMS]
    k = xy.shape[0]
    if k == 0 and not any(v.size for v in lone):
        return None
    if k == 0:  # each arm on its own, from the prior's marginal; the correlation from its prior
        mu = np.full((n, 2), np.nan)
        cov = np.zeros((n, 2, 2))
        distinct = n
        for a, v in enumerate(lone):
            if v.size == 0:
                cov[:, a, a] = np.nan
                continue
            ss = float(((v - v.mean()) ** 2).sum())
            var, got = _truncated_inv_gamma((COST_NU - 1) / 2 + (v.size - 1) / 2, COST_SD ** 2 / 2 + ss / 2, rng, n)
            distinct = min(distinct, got)
            mu[:, a] = v.mean() + np.sqrt(var / v.size) * rng.standard_normal(n)
            cov[:, a, a] = var
        rho = rng.uniform(-1.0, 1.0, size=n)
        cov[:, 0, 1] = cov[:, 1, 0] = rho * np.sqrt(np.abs(cov[:, 0, 0] * cov[:, 1, 1]))
        return mu, cov, (None if distinct >= n else float(distinct))
    zbar = xy.mean(axis=0)
    dev = xy - zbar
    psi = COST_SD ** 2 * np.eye(2) + dev.T @ dev
    extra = sum(v.size for v in lone)
    m = n * (OVERSAMPLE if extra else 2)
    cov = np.empty((0, 2, 2))
    for _ in range(20):
        c = _inv_wishart2(COST_NU + k - 1, psi, rng, m)
        cov = np.concatenate([cov, c[_in_range(c)]])
        if cov.shape[0] >= (m if extra else n):
            break
    if cov.shape[0] == 0:  # no draw of the truncated model: refuse, rather than return draws of another model
        raise CostRange(COST_RANGE_WHY)
    # mu | Sigma ~ N(zbar, Sigma / k)
    l00 = np.sqrt(cov[:, 0, 0] / k)
    l10 = cov[:, 1, 0] / k / l00
    l11 = np.sqrt(np.maximum(cov[:, 1, 1] / k - l10 ** 2, 0.0))
    z = rng.standard_normal((cov.shape[0], 2))
    mu = np.stack([zbar[0] + l00 * z[:, 0], zbar[1] + l10 * z[:, 0] + l11 * z[:, 1]], axis=1)
    ess = None
    if extra:
        logw = _normal_loglik(lone[0], mu[:, 0], cov[:, 0, 0]) + _normal_loglik(lone[1], mu[:, 1], cov[:, 1, 1])
        idx, ess = _resample(logw, rng, n)
    else:
        idx = np.arange(n) if cov.shape[0] >= n else rng.choice(cov.shape[0], size=n)  # iid draws: the first n
        ess = None if cov.shape[0] >= n else float(cov.shape[0])  # fewer kept: resampled, this many distinct
    return mu[idx], cov[idx], ess


@dataclass
class CostDraws:
    """Draws of each arm's mean dollars per run (NaN for an arm without costs), and what `draw_future` needs
    to draw future runs from each draw: a lognormal draw's parameters, or a DP draw's random measure."""

    mean: np.ndarray  # (n, 2)
    dp: np.ndarray  # (n,) bool: the draw is the DP part's
    row: np.ndarray  # (n,) the draw's row in its part's arrays
    mu: np.ndarray | None = None  # the lognormal's draws: (n0, 2) log-cost means
    cov: np.ndarray | None = None  # (n0, 2, 2)
    v: np.ndarray | None = None  # the DP part's draws: (n1,) the mass on new runs
    w0: np.ndarray | None = None  # (n1, A) the new runs' weights
    atoms: np.ndarray | None = None  # (2, n1, A) the new runs' dollars by arm
    bw: np.ndarray | None = None  # (n1, U) the observed units' weights
    units: np.ndarray | None = None  # (U, 2) the observed log costs: the pairs, the lone base runs, the lone
    #                                   candidate runs, NaN for a missing cost
    weight: float = 0.0  # the DP part's weight: 0 or 1 for "sw"
    ess: float | None = None  # the lognormal's, when lone runs reweight its draws


def _mix_bic(y: np.ndarray) -> float:
    """BIC(one normal) - BIC(two normals with one shared variance) on y, by MIX_ITERS EM steps from splits at
    its quartiles (the larger likelihood of the starts); -inf under MIX_MIN values or with no spread. Each EM
    step reads the data through its weighted sums of 1, y and y^2, so the starts' parameters are scalars."""
    n = y.size
    if n < MIX_MIN or y.std() == 0:
        return -math.inf
    ll1 = n * (-0.5 * math.log(2 * math.pi * y.var()) - 0.5)
    cuts = np.percentile(y, [25, 50, 75])
    r = (y[None, :] > cuts[:, None]).astype(float)
    k = r.sum(axis=1)
    r = r[(k >= 1) & (k <= n - 1)]
    if r.shape[0] == 0:
        return -math.inf
    yc = y - y.mean()
    Y = np.stack([np.ones(n), yc, yc * yc], axis=1)
    t0, t1, t2 = (float(x) for x in Y.sum(axis=0))
    X = Y[:, :2].T
    ab = np.empty((r.shape[0], 2))

    def m_step(s0: float, s1: float, s2: float) -> tuple[float, float, float, float]:
        u0, u1, u2 = t0 - s0, t1 - s1, t2 - s2
        m1, m2 = u1 / u0, s1 / s0
        return min(max(s0 / n, 1e-3), 1 - 1e-3), m1, m2, max((u2 - u1 * m1 + s2 - s1 * m2) / n, 1e-4)

    for _ in range(MIX_ITERS):
        s = (r @ Y).tolist()
        for j, sj in enumerate(s):
            w, m1, m2, v = m_step(*sj)
            b = (m2 - m1) / v
            ab[j] = math.log(w / (1 - w)) - b * (m1 + m2) / 2, b
        r = expit(ab @ X)  # the second component's responsibilities
    best = -math.inf
    for sj in s:  # the likelihood at the last step's parameters
        w, m1, m2, v = m_step(*sj)
        lp = np.logaddexp(math.log(1 - w) - 0.5 * (yc - m1) ** 2 / v, math.log(w) - 0.5 * (yc - m2) ** 2 / v)
        best = max(best, float(lp.sum()) - 0.5 * n * math.log(2 * math.pi * v))
    return 2 * (best - ll1) - 2 * math.log(n)  # two more parameters: a second mean and a weight


def _sticks(rng: np.random.Generator, n: int, alpha: float) -> np.ndarray:
    """n stick-breaking weight vectors of a DP(alpha), renormalized: enough sticks that the mass left over,
    (alpha / (1 + alpha))^sticks, is about 1e-4 or less."""
    atoms = max(DP_ATOMS, math.ceil(9.5 * (alpha + 0.5)))
    b = 1.0 - rng.random((n, atoms)) ** (1.0 / alpha)  # Beta(1, alpha) by its inverse cdf
    w = b * np.concatenate([np.ones((n, 1)), np.cumprod(1 - b[:, :-1], axis=1)], axis=1)
    return w / w.sum(axis=1, keepdims=True)


def _level(g: np.ndarray, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(mu, sd) draws of a normal on log levels g: flat on mu, and the lognormal's marginal prior on the
    variance (inverse gamma from COST_NU and COST_SD, sds within SIGMA_RANGE)."""
    ss = float(((g - g.mean()) ** 2).sum())
    var, _ = _truncated_inv_gamma((COST_NU - 1) / 2 + (g.size - 1) / 2, COST_SD ** 2 / 2 + ss / 2, rng, n)
    return g.mean() + np.sqrt(var / g.size) * rng.standard_normal(n), np.sqrt(var)


def _dp_part(units: np.ndarray, k: int, rng: np.random.Generator, n: int) -> CostDraws:
    """n draws of the DP posterior DP(alpha G0 + the units). The units are k pairs and the lone runs; the mass
    on G0 is V ~ Beta(alpha, units), on G0 a DP(alpha, G0) draw, and on the units Dirichlet(1, ..., 1)
    weights, each arm's renormalized over the units with its cost. G0 keeps each pair's dominance:
    - "level": a new pair is exp(L) times an observed pair's shape (b / g, c / g), g = sqrt(b c) its level,
      with L normal and its (mu, sd) drawn from their posterior on the observed log levels. With no pair, each
      arm's runs are exp(L) with (mu, sd) from its lone runs, the arms sharing their standard normals.
    - "paired": an observed unit's costs (one uniform for both arms, the pairs first) times one lognormal
      factor of mean 1 and log sd tau.
    - "extra": "paired", with a second such factor t on a pair's extra cost: the cheaper arm min(b, c) s, the
      dearer arm min(b, c) s + |c - b| s t. Every atom is positive and keeps its pair's order, each arm's mean
      given the unit is its observed cost, and one pair's cost ratio is not a point unless its costs are equal.
      A lone run is scaled by s alone."""
    kind, alpha, tau = COST_PART
    U = np.exp(units)
    w0 = _sticks(rng, n, alpha)
    atoms = np.full((2,) + w0.shape, np.nan)
    if kind == "level":
        z = rng.standard_normal(w0.shape)
        if k:
            lg = units[:k].mean(axis=1)
            mu, sd = _level(lg, rng, n)
            shape = np.exp(units[:k] - lg[:, None])
            j = rng.integers(0, k, size=w0.shape)
            lev = np.exp(mu[:, None] + sd[:, None] * z)
            for a in range(2):
                atoms[a] = lev * shape[j, a]
        else:
            for a in range(2):
                x = units[np.isfinite(units[:, a]), a]
                if x.size:
                    mu, sd = _level(x, rng, n)
                    atoms[a] = np.exp(mu[:, None] + sd[:, None] * z)
    elif kind == "extra":
        u = rng.random(w0.shape)
        f = np.exp(tau * rng.standard_normal(w0.shape) - tau ** 2 / 2)
        t = np.exp(tau * rng.standard_normal(w0.shape) - tau ** 2 / 2)
        lo, ext = np.fmin(U[:, 0], U[:, 1]), np.abs(U[:, 1] - U[:, 0])  # a lone run: its own cost, no extra
        for a in range(2):
            idx = np.flatnonzero(np.isfinite(U[:, a]))
            if idx.size:
                j = idx[np.minimum((u * idx.size).astype(int), idx.size - 1)]
                dear = np.isfinite(ext[j]) & (U[j, a] > U[j, 1 - a])
                atoms[a] = f * (lo[j] + np.where(dear, ext[j] * t, 0.0))
    else:
        u = rng.random(w0.shape)
        f = np.exp(tau * rng.standard_normal(w0.shape) - tau ** 2 / 2)
        for a in range(2):
            x = U[np.isfinite(U[:, a]), a]
            if x.size:
                atoms[a] = f * x[np.minimum((u * x.size).astype(int), x.size - 1)]
    v = rng.beta(alpha, len(U), size=n)
    bw = rng.dirichlet(np.ones(len(U)), size=n)
    mean = np.full((n, 2), np.nan)
    for a in range(2):
        has = np.isfinite(U[:, a])
        if has.any():
            wa = bw[:, has]
            mean[:, a] = v * (w0 * atoms[a]).sum(axis=1) + (1 - v) * (wa @ U[has, a]) / wa.sum(axis=1)
    return CostDraws(mean, np.ones(n, bool), np.arange(n), v=v, w0=w0, atoms=atoms, bw=bw, units=units, weight=1.0)


def _contradicted(xy: np.ndarray, mean: np.ndarray) -> bool:
    """At DOM_MIN_PAIRS pairs or more, every pair orders the arms one way and the draws' means order them the
    other way."""
    if xy.shape[0] < DOM_MIN_PAIRS:
        return False
    up, down = bool(np.all(xy[:, 1] >= xy[:, 0])), bool(np.all(xy[:, 1] <= xy[:, 0]))
    m = mean.mean(axis=0)
    return up != down and ((up and m[1] < m[0]) or (down and m[1] > m[0]))


def cost_draws(data: Data, rng: np.random.Generator, n: int) -> CostDraws | None:
    """The cost model of COST_MODEL "sw" or "avg": the lognormal (`cost_posterior`) and the DP part on the
    same data. Both read the mixture test, the BIC margin delta of two normals over one on the pairs' mean log
    costs (`_mix_bic`).
    - "sw": the DP part when delta > MIX_TAU, or when every pair orders the arms one way and the lognormal's
      means the other way; else the lognormal.
    - "avg": each draw is the DP part's with its BIC weight w = 1 / (1 + exp(-delta / 2)), else the
      lognormal's; the DP part alone when the average's means contradict unanimous pairs.
    Either is the DP part where the lognormal states no range."""
    xy = np.asarray(data.cost_pairs, dtype=float).reshape(-1, 2)
    lone = [np.asarray(data.cost_lone[a], dtype=float) for a in ARMS]
    k = xy.shape[0]
    if k == 0 and not any(v.size for v in lone):
        return None
    units = np.vstack([xy, np.c_[lone[0], np.full(lone[0].size, np.nan)], np.c_[np.full(lone[1].size, np.nan), lone[1]]])
    delta = _mix_bic(xy.mean(axis=1)) if k else -math.inf
    if COST_MODEL == "sw" and delta > MIX_TAU:
        return _dp_part(units, k, rng, n)
    try:
        mu, cov, ess = cost_posterior(data, rng, n)
    except CostRange:
        return _dp_part(units, k, rng, n)
    ln = np.exp(mu + np.stack([cov[:, 0, 0], cov[:, 1, 1]], axis=1) / 2)
    out = CostDraws(ln, np.zeros(n, bool), np.arange(n), mu=mu, cov=cov, ess=ess)
    if COST_MODEL == "avg":
        w = 0.5 * (1 + math.tanh(delta / 4)) if math.isfinite(delta) else 0.0
        out.weight = w
        if w >= DP_MIN_WEIGHT:
            dp = rng.random(n) < w
            n1 = int(dp.sum())
            try:
                part = _dp_part(units, k, rng, n1) if n1 else None
            except CostRange:  # the DP part states no range: the lognormal alone
                part, out.weight = None, 0.0
            if part is not None:
                out.mean = out.mean.copy()
                out.mean[dp] = part.mean
                out.dp, out.row = dp, np.where(dp, np.cumsum(dp) - 1, np.cumsum(~dp) - 1)
                out.v, out.w0, out.atoms, out.bw, out.units = part.v, part.w0, part.atoms, part.bw, units
                out.mu, out.cov = mu[~dp], cov[~dp]
    if _contradicted(xy, out.mean):
        return _dp_part(units, k, rng, n)
    return out


def posterior(data: Data, rng: np.random.Generator, n: int = DRAWS) -> Posterior:
    """Outcome and cost draws come from two child streams of `rng`, so the cost ranges do not move when only
    outcomes change. The outcome and the cost parameters are independent a priori and in the likelihood."""
    post = Posterior(n=n)
    r_out, r_cost = rng.spawn(2)
    if data.kind == "score":
        s = score_posterior(data, r_out, n)
        if s is not None:
            post.mu, post.sd = s
            post.t = score_t(data)
    else:
        post.theta, ess = binary_posterior(data, r_out, n)
        if ess is not None:
            post.ess["outcome"] = round(ess, 1)
    try:
        if COST_MODEL == "ln" and (why := zero_beside_priced(data)):
            raise CostRange(why)
        c = cost_posterior(data, r_cost, n) if COST_MODEL == "ln" else cost_draws(data, r_cost, n)
    except CostRange as e:
        c, post.cost_refused = None, str(e)
    if isinstance(c, CostDraws):
        post.cost, ess = c, c.ess
    elif c is not None:
        post.cmu, post.ccov, ess = c
    if c is not None and ess is not None:
        post.ess["cost"] = round(ess, 1)
    return post


# ---------------------------------------------------------------- the comparison's draws
@dataclass
class Draws:
    post: Posterior
    gain: np.ndarray | None
    cost_base: np.ndarray | None
    cost_cand: np.ndarray | None
    cpar_base: np.ndarray | None
    cpar_cand: np.ndarray | None
    improvement: np.ndarray | None  # the objective's improvement, positive is better
    mean_ok: dict[str, bool] = field(default_factory=dict)  # does each summary's posterior mean exist
    gain_mean: float | None = None  # exact, for a score


def draws(data: Data, objective: str, rng: np.random.Generator, n: int = DRAWS) -> Draws:
    post = posterior(data, rng, n)
    cb, cc = post.mean_cost("base"), post.mean_cost("cand")
    ok = {"cost_base": cb is not None, "cost_cand": cc is not None}
    if data.kind == "score":
        g = post.mu
        exists = post.t is not None and post.t[0] > 1  # the t mean exists from 2 degrees of freedom
        ok.update(gain=exists, improvement=exists)
        return Draws(post, g, cb, cc, None, None, g, ok, post.t[1] if exists else None)
    pb, pc = post.chance("base"), post.chance("cand")
    g = pc - pb
    with np.errstate(divide="ignore", invalid="ignore"):
        kb = cb / pb if cb is not None else None
        kc = cc / pc if cc is not None else None
    sb, sc = data.successes("base"), data.successes("cand")
    ok.update(gain=True, cpar_base=sb >= 1, cpar_cand=sc >= 1)
    if objective == "gain":
        imp, ok["improvement"] = g, True
    elif kb is not None and kc is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = 1.0 - kc / kb
        ok["improvement"] = sc >= 1  # 1 - (C_c / C_b)(p_b / p_c): E[1 / p_c] needs an accepted candidate run
    else:
        imp = None
    return Draws(post, g, cb, cc, kb, kc, imp, ok)


def p_better(data: Data, d: Draws, floor: float) -> float | None:
    """P(improvement > floor): exact for a score (the t marginal), from the draws otherwise."""
    if data.kind == "score":
        t = d.post.t
        return None if t is None else float(stdtr(t[0], (t[1] - floor) / t[2]))
    return None if d.improvement is None else float(np.mean(d.improvement > floor))


# ---------------------------------------------------------------- the decision; the next pairs' parts
@dataclass
class Settings:
    objective: str = "cpar"
    floor: float = FLOOR_CPAR
    confidence: float = CONFIDENCE
    threshold: float = CONFIDENCE  # adopt at p_better >= threshold (calibration.threshold)
    horizon_runs: float = HORIZON_RUNS
    value_usd: float | None = None  # gain objective: dollars per point of gain (default: the base's cpar)


def decide(pb_: float | None, st: Settings) -> str | None:
    if pb_ is None:
        return None
    if pb_ >= st.threshold:
        return "adopt"
    if pb_ <= 1.0 - st.confidence:
        return "reject"
    return None


def draw_future(data: Data, post: Posterior, i: int, m: int, rng: np.random.Generator) -> dict[str, Any]:
    """m more pairs from posterior draw i (the posterior predictive, one parameter draw per future): the cells
    or the score differences, and the log costs, paired or by arm. `extend` takes any prefix of them."""
    out: dict[str, Any] = {}
    if data.kind == "binary":
        th = post.theta[i] / post.theta[i].sum()
        out["cells"] = rng.choice(4, size=m, p=th)
    elif post.mu is not None:
        out["diffs"] = post.mu[i] + post.sd[i] * rng.standard_normal(m)
    c = post.cost
    if c is not None and c.dp[i]:
        x = _dp_future(c, int(c.row[i]), m, rng)
        if np.all(np.isfinite(x[0])):
            out["cost_pairs"] = x
        else:
            out["cost_lone"] = {arm: x[:, a] for a, arm in enumerate(ARMS) if np.isfinite(x[0, a])}
    elif c is not None or post.cmu is not None:
        mu, cov = (c.mu[c.row[i]], c.cov[c.row[i]]) if c is not None else (post.cmu[i], post.ccov[i])
        if np.all(np.isfinite(mu)):
            l00 = math.sqrt(cov[0, 0])
            l10 = cov[1, 0] / l00
            l11 = math.sqrt(max(cov[1, 1] - l10 ** 2, 0.0))
            z = rng.standard_normal((m, 2))
            out["cost_pairs"] = np.stack([mu[0] + l00 * z[:, 0], mu[1] + l10 * z[:, 0] + l11 * z[:, 1]], axis=1)
        else:
            out["cost_lone"] = {arm: mu[a] + math.sqrt(cov[a, a]) * rng.standard_normal(m)
                                for a, arm in enumerate(ARMS) if np.isfinite(mu[a])}
    return out


def _dp_future(c: CostDraws, k: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """m runs' log costs (m, 2) from row k of a DP part's random measure: each is a new run with chance v (one
    stick for both arms), else an observed unit, drawn by each arm's weights over the units with its cost
    through one shared uniform, the pairs first. Their mean over many is the draw's mean cost."""
    new = rng.random(m) < c.v[k]
    j = rng.choice(c.w0.shape[1], size=m, p=c.w0[k])
    u = rng.random(m)
    out = np.full((m, 2), np.nan)
    for a in range(2):
        has = np.isfinite(c.units[:, a])
        if has.any():
            cw = np.cumsum(np.where(has, c.bw[k], 0.0))
            pick = np.minimum(np.searchsorted(cw / cw[-1], u, side="right"), len(cw) - 1)
            out[:, a] = np.where(new, np.log(c.atoms[a, k, j]), c.units[pick, a])
    return out


def extend(data: Data, fut: dict[str, Any], m: int) -> Data:
    """The data after the first m pairs of a drawn future."""
    out = data.copy()
    if "cells" in fut:
        out.cells = out.cells + np.bincount(fut["cells"][:m], minlength=4)
    if "diffs" in fut:
        out.diffs = out.diffs + list(fut["diffs"][:m])
    if "cost_pairs" in fut:
        out.cost_pairs = out.cost_pairs + [tuple(x) for x in fut["cost_pairs"][:m]]
    for arm, xs in (fut.get("cost_lone") or {}).items():
        out.cost_lone[arm] = out.cost_lone[arm] + list(xs[:m])
    return out


def future(data: Data, post: Posterior, i: int, m: int, rng: np.random.Generator) -> Data:
    """The data after m more pairs drawn from posterior draw i."""
    return extend(data, draw_future(data, post, i, m, rng), m)
