"""Which task to run next, when to stop, and the design-based estimates the replay compares against.

Methods:
- `loss`: the task whose next run is expected to cut Var(S) the most (the expected-loss rule);
- `info`: the most Fisher information about the ability at its current mean (classic adaptive testing);
- `fixed`: one order for every candidate, by information averaged over the ability prior (anchor tasks);
- `random`: a seeded order per candidate;
- `stratified`: strata by the history's predicted rate, Neyman allocation from history-predicted spreads, a
  seeded random order within each stratum. Strata and allocation come from the history alone.

The design-based estimates (`srs_estimate`, `stratified_estimate`, `difference_estimate`) target S, the mean of
the probabilities, and every run is a fresh Bernoulli draw, so their variance estimates keep the run noise and
use no finite population correction: conservative, and never zero at the full suite. They are unbiased at a
fixed n with predictions and allocation fixed from the history; a data-dependent stop voids that.

A sample of one, or one with every outcome equal (all passed, all failed), has no usable sample variance. Each
estimate then floors the variance per run at a Bernoulli bound: p~(1 - p~) with p~ = (sum y + 1) / (n + 2), and
0.25 for one run. The floor is conservative, not a coverage guarantee; the replay measures coverage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from .posterior import CandidatePosterior

METHODS = ("loss", "info", "fixed", "random", "stratified")
STRATA = 4


def _rng(seed: int, key: str) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def random_order(tasks: list[str], seed: int = 0, key: str = "") -> list[str]:
    """A random order of `tasks`, fixed by `seed` and the candidate `key`."""
    return [tasks[i] for i in _rng(seed, key).permutation(len(tasks))]


def strata_of(post: CandidatePosterior, k: int = STRATA) -> np.ndarray:
    """Per task: its stratum, 0 to k - 1, by quantile of the history's predicted rate for a new arm."""
    r = post.prior_rates()
    k = max(1, min(k, len(r)))
    ranks = np.argsort(np.argsort(r, kind="stable"), kind="stable")
    return (ranks * k) // len(r)


def stratified_order(post: CandidatePosterior, seed: int = 0, key: str = "", k: int = STRATA) -> list[int]:
    """The full stratified order (task indices), from the history alone: Neyman allocation by the history's
    predicted spread per stratum (two runs per stratum first, so each has a variance), random within strata."""
    h = strata_of(post, k)
    r = post.prior_rates()
    rng = _rng(seed, "strata:" + key)
    pools = {s: list(rng.permutation(np.flatnonzero(h == s))) for s in np.unique(h)}
    size = {s: len(v) for s, v in pools.items()}
    spread = {s: float(np.sqrt(np.mean(r[h == s] * (1 - r[h == s])) + np.var(r[h == s]))) or 1e-3 for s in pools}
    took = {s: 0 for s in pools}
    order = []
    while any(pools.values()):
        live = [s for s in pools if pools[s]]
        need = [s for s in live if took[s] < 2]
        if need:
            s = min(need, key=lambda z: (took[z], z))
        else:  # the largest drop in sum_h N_h^2 S_h^2 / n_h from one more run
            s = max(live, key=lambda z: (size[z] * spread[z]) ** 2 / (took[z] * (took[z] + 1)))
        order.append(int(pools[s].pop(0)))
        took[s] += 1
    return order


def next_tasks(post: CandidatePosterior, method: str = "loss", n: int = 1, seed: int = 0, key: str = "",
               strata: int = STRATA) -> list[dict]:
    """The next `n` tasks with no run yet, best first: task, predicted rate p and value (the expected cut in
    the SD of S, in score units, from one run on that task alone)."""
    if method not in METHODS:
        raise ValueError(f"method is one of {', '.join(METHODS)}, not {method!r}")
    tasks = post.fit.tasks
    open_ = np.flatnonzero(~post.seen())
    if len(open_) == 0 or n <= 0:
        return []
    mean, var = post.score()["mean"], post.score()["sd"] ** 2
    cut = post.var_cut()
    if method == "loss":
        order = sorted(open_, key=lambda j: (-cut[j], j))
    elif method in ("info", "fixed"):
        info = post.information(at_prior=method == "fixed")
        order = sorted(open_, key=lambda j: (-info[j], j))
    elif method == "random":
        pos = {t: i for i, t in enumerate(random_order(tasks, seed, key))}
        order = sorted(open_, key=lambda j: pos[tasks[j]])
    else:
        pos = {j: i for i, j in enumerate(stratified_order(post, seed, key, strata))}
        order = sorted(open_, key=lambda j: pos[j])
    rates = post.rates()
    sd = np.sqrt(max(var, 0.0))
    return [{"task": tasks[j], "p": float(rates[j]), "value": float(sd - np.sqrt(max(var - cut[j], 0.0)))}
            for j in order[:n]]


@dataclass
class Decision:
    stop: bool
    reason: str
    side: str | None = None  # versus a bar: "above" or "below" when decided
    p_above: float | None = None
    target: float | None = None  # the precision stop's half-width


def decide(post: CandidatePosterior, *, halfwidth: float | None = None, versus=None, level: float = 0.9,
           min_sessions: int = 5, max_sessions: int | None = None) -> Decision:
    """Stop or continue. With `versus` (a fixed score, or another arm's `CandidatePosterior`), stop once
    P(S > bar) reaches `level` or falls to 1 - level; otherwise stop once the 80% range's half-width is within
    `halfwidth` (default: what a full run of the suite would give)."""
    sessions = post.sessions
    p_above = post.p_above(versus) if versus is not None else None
    side = None
    if p_above is not None:
        side = "above" if p_above >= level else "below" if p_above <= 1 - level else None
    est = post.score()
    half = (est["range80"][1] - est["range80"][0]) / 2
    target = post.full_run_halfwidth(0.8) if halfwidth is None else float(halfwidth)
    if bool(post.seen().all()):
        return Decision(True, "every task of the suite has a run", side, p_above, target)
    if max_sessions is not None and sessions >= max_sessions:
        return Decision(True, f"reached the cap of {max_sessions} sessions", side, p_above, target)
    if sessions < min_sessions:
        return Decision(False, f"{sessions} of the first {min_sessions} sessions", side, p_above, target)
    if versus is not None:
        bar = "the arm's score" if isinstance(versus, CandidatePosterior) else f"{float(versus):.3f}"
        if side is not None:
            return Decision(True, f"P(score > {bar}) is {p_above:.3f}: {side} at the {level:.2f} level", side,
                            p_above, target)
        return Decision(False, f"P(score > {bar}) is {p_above:.3f}, between {1 - level:.2f} and {level:.2f}",
                        side, p_above, target)
    if half <= target:
        return Decision(True, f"the 80% range is +-{half:.3f}, within the target +-{target:.3f}", side, p_above,
                        target)
    return Decision(False, f"the 80% range is +-{half:.3f}, wider than the target +-{target:.3f}", side, p_above,
                    target)


# ---------------------------------------------------------------- design-based estimates
def _interval(mean: float, var: float, levels=(0.8, 0.95)) -> dict:
    sd = float(np.sqrt(max(var, 0.0)))
    out = {"mean": float(mean), "sd": sd}
    for lv in levels:
        z = float(ndtri((1 + lv) / 2))
        out[f"range{int(round(lv * 100))}"] = [max(0.0, mean - z * sd), min(1.0, mean + z * sd)]
    return out


def run_variance(y) -> float:
    """Per run: the sample variance of `y`, floored at the Bernoulli bound p~(1 - p~), p~ = (sum y + 1) /
    (n + 2); 0.25 for one run or none."""
    y = np.asarray(y, float)
    n = len(y)
    if n <= 1:
        return 0.25
    pt = (float(y.sum()) + 1) / (n + 2)
    return max(float(np.var(y, ddof=1)), pt * (1 - pt))


def srs_estimate(y) -> dict:
    """The sample mean of one run per randomly drawn task, with s_y^2 / n as its variance: no finite
    population correction, so the run noise stays and the range is conservative for S."""
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return _interval(0.5, 0.25)
    return _interval(float(y.mean()), run_variance(y) / n)


def stratified_mean(samples: list[list[float]], sizes: list[int]) -> float:
    """sum_h W_h y_bar_h over the strata, W_h = N_h / N."""
    N = float(sum(sizes))
    return float(sum(n * np.mean(s) for s, n in zip(samples, sizes)) / N)


def stratified_estimate(samples: list[list[float]], sizes: list[int]) -> dict:
    """The stratified mean with sum_h W_h^2 s_h^2 / n_h (no finite population correction; the Bernoulli bound
    0.25 / n_h for a stratum with one run). A stratum with no run counts at 0.5, with the bound 0.25."""
    N = float(sum(sizes))
    mean = var = 0.0
    for s, n in zip(samples, sizes):
        s = np.asarray(s, float)
        w = n / N
        if len(s) == 0:
            mean += w * 0.5
            var += w * w * 0.25
            continue
        mean += w * s.mean()
        var += w * w * run_variance(s) / len(s)
    return _interval(mean, var)


def difference_estimate(y, m_seen, m_all) -> dict:
    """mean(m_all) + mean(y - m_seen), with m fixed from the history before any candidate run; the variance is
    the residuals' s^2 / n, no finite population correction, floored at the predictions' own Bernoulli
    variance, mean m(1 - m) over the seen tasks (0.25 for one run)."""
    y, ms = np.asarray(y, float), np.asarray(m_seen, float)
    base = float(np.mean(m_all))
    if len(y) == 0:
        return _interval(base, 0.25)
    r = y - ms
    per = 0.25 if len(r) == 1 else max(float(np.var(r, ddof=1)), float(np.mean(ms * (1 - ms))))
    return _interval(base + float(r.mean()), per / len(r))
