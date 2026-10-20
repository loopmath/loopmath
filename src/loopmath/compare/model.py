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
  (inverse gamma for sigma^2), with the correlation drawn from its prior.
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

**The value of the next pair**, in dollars over `horizon_runs` future runs, is the knowledge gradient of
`belief/lookahead.py` (spec 04 section 6) on this posterior: each simulated future takes one joint draw of
the parameters from the same posterior that gave the decision (including the reweighting by lone runs), then
draws its m pairs from that draw. The value is the mean over futures of the plug-in saving of the decision
after them; the decision now is undecided, so it is worth 0. One pair rarely moves a leaning decision, so the
value is the best per-pair value over 1, 2, 4, 8 and 16 more pairs (KG(*), Frazier and Powell 2010,
"Paradoxes in learning and the marginal value of information", Decision Analysis 7(4)), never past the cap.
Each of the `LOOK_SIMS` futures is drawn once for the largest step count and read at every step count, so the
step counts are compared on the same futures. A mean cost exp(mu + sigma^2 / 2) is heavy-tailed in sigma, so
a few futures carry much of the value: at 96 futures one tail draw could flip the stop, which 256 nested
futures do not (`tests/compare/test_compare_model.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import stdtr

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

    def copy(self) -> "Data":
        return Data(self.kind, self.cells.copy(), dict(self.lone), list(self.diffs), self.scale,
                    list(self.cost_pairs), {k: list(v) for k, v in self.cost_lone.items()})

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
    ess: dict[str, float] = field(default_factory=dict)  # effective sample sizes of reweighted draws
    cost_refused: str | None = None  # why there are no cost draws although there are costs

    def chance(self, arm: str) -> np.ndarray | None:
        if self.theta is None:
            return None
        return self.theta[:, 0] + self.theta[:, 2 if arm == "base" else 1]

    def mean_cost(self, arm: str) -> np.ndarray | None:
        if self.cmu is None:
            return None
        a = ARMS.index(arm)
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
        c = cost_posterior(data, r_cost, n)
    except CostRange as e:
        c, post.cost_refused = None, str(e)
    if c is not None:
        post.cmu, post.ccov, ess = c
        if ess is not None:
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


# ---------------------------------------------------------------- the decision and the next pair's value
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


def plug_in_saving(data: Data, st: Settings, d: Draws) -> float | None:
    """Dollars a future run saves by adopting the candidate, at the posterior means (the KG plug-in): mean
    cost over mean chance per arm for cpar; for the gain, the gain's dollar value less the cost change."""
    if d.cost_base is None or d.cost_cand is None or d.gain is None:
        return None
    eb, ec = float(np.mean(d.cost_base)), float(np.mean(d.cost_cand))
    if data.kind == "score":
        return None if st.value_usd is None else st.value_usd * float(np.mean(d.gain)) - (ec - eb)
    pb, pc = float(np.mean(d.post.chance("base"))), float(np.mean(d.post.chance("cand")))
    if st.objective == "gain":
        value = st.value_usd if st.value_usd is not None else eb / pb
        return value * (pc - pb) - (ec - eb)
    return eb / pb - ec / pc


def draw_future(data: Data, post: Posterior, i: int, m: int, rng: np.random.Generator) -> dict[str, Any]:
    """m more pairs from posterior draw i (the posterior predictive, one parameter draw per future): the cells
    or the score differences, and the log costs, paired or by arm. `extend` takes any prefix of them."""
    out: dict[str, Any] = {}
    if data.kind == "binary":
        th = post.theta[i] / post.theta[i].sum()
        out["cells"] = rng.choice(4, size=m, p=th)
    elif post.mu is not None:
        out["diffs"] = post.mu[i] + post.sd[i] * rng.standard_normal(m)
    if post.cmu is not None:
        mu, cov = post.cmu[i], post.ccov[i]
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


def decision_value(data: Data, st: Settings, rng: np.random.Generator) -> float:
    """Dollars per future run of the decision the data would give: the plug-in saving if it adopts, else 0."""
    d = draws(data, st.objective, rng, LOOK_DRAWS)
    if decide(p_better(data, d, st.floor), st) != "adopt":
        return 0.0
    s = plug_in_saving(data, st, d)
    return 0.0 if s is None else s


def next_pair_value(data: Data, st: Settings, post: Posterior, rng: np.random.Generator,
                    max_pairs: int | None = None, sims: int = LOOK_SIMS) -> tuple[float | None, int | None]:
    """(dollars over the horizon that one more pair is worth, the step count that gave it); None when the
    rule's outcome has no dollar value (a score without `value_usd`) or there is no posterior to draw from.
    Called only while undecided, so the decision now is worth 0. Each simulated future is drawn once, for the
    largest step count, and every step count reads its prefix: the steps are compared on the same futures."""
    if data.kind == "score" and (st.value_usd is None or post.mu is None):
        return None, None
    if data.kind == "binary" and post.theta is None:
        return None, None
    steps = [m for m in LOOK_STEPS if max_pairs is None or m <= max_pairs]
    if not steps:
        return 0.0, None
    totals = np.zeros(len(steps))
    for i in rng.integers(post.n, size=sims):
        fut = draw_future(data, post, int(i), steps[-1], rng)
        for j, m in enumerate(steps):
            totals[j] += decision_value(extend(data, fut, m), st, rng)
    per_pair = st.horizon_runs * totals / sims / np.asarray(steps, dtype=float)
    j = int(np.argmax(per_pair))
    return (float(per_pair[j]), steps[j]) if per_pair[j] > 0 else (0.0, None)
