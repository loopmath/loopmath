"""`eval labels next`: what a human reviewer should score or compare next.

Reads a fit file, optionally a value file, and a budget (`--n` assignments or `--minutes`). Writes a
labels plan (`loopmath.eval-labels/1`) and, with `--form`, a blank CSV that `eval scores import --labels`
reads back.

Blind by construction: no plan, form or printed output holds an arm, a model name or a judge score. The
options record `--arms` as a count and a digest only. Run ids and refs come from the user's run records
as they are, so the command refuses, and writes nothing, when a candidate's run id or ref spells out an
arm, a model id or a model family. One exception: when the study's run records mark every candidate run
as simulated, the plan is written and a note says it is not blind. One real run keeps the refusal.

Candidates
----------
- A score: a target (run, step) with the components that have a reading by another judge and none by
  this reviewer. A `human_only` component needs no judge reading of its own.
- A pair: two targets of one input from different arms, on the step's `overall` component or the
  components named. Pairs this reviewer already judged are left out when the study folder holds
  `pairs.jsonl`.

Pilot mode
----------
A random sample on the full rubric, with no model quantity. Candidates are stratified by step, arm and
judge score band (thirds of the plain mean of the judges' readings, ranked per component), then drawn by
systematic sampling with a random start over the strata in turn. Every candidate has the same inclusion
probability `n / N`, and every stratum gets its proportional share to within one. With a minutes budget
the candidates are grouped by their minutes and each group gets a fixed number of draws, so every plan
spends the same minutes; the probability is then one per group, and the groups differ by rounding only.
Reads a fit that did not converge.

Active mode
-----------
Every figure comes from the posterior draws by the approximation below, with no refit.

For runs `B` of one input, write `q_B` for their stacked hidden quality. Given the arm means `mu`, the
model says `q_B ~ N(mu_B, T_B)`, where `T_B` holds `Sigma_input` between every two runs and `Sigma_run`
on each run (posterior means of the covariances). From the draws: `P_B`, the posterior covariance of
`q_B` given every reading, and `S`, the posterior covariance of the arm means of all arms. Because the
runs of one input touch the rest of the study only through the arm means,

    Cov(mu, q_B) ~= S[:, B] T_B^-1 C_B

with `C_B` the covariance of `q_B` given the arm means: `P_B` (first brought between 0 and
`T_B + S[B, B]`) less the part that is the arm means' own uncertainty, `C_B + C_B W C_B = P_B` with
`W = T_B^-1 S[B, B] T_B^-1`. It is solved in closed form (`given_means`), which gives a positive
semidefinite `C_B`, never above `P_B`, then kept under `T_B`. With `G = C_B T_B^-1` this is the joint
normal law `q_B = G mu_B + rest`, `Var(rest) = C_B`, so the step below is never larger than the posterior
covariance of the arm means, and the command refuses if a computed step breaks that bound.

A label reads linear parts of `q_B` with the reviewer's own noise: `y = L q_B + e`, `e ~ N(0, N)`, with
`N = tau^2 + sigma^2` of the reviewer per component (for a pair `2 tau^2 + sigma_pair^2`, on the
difference of the two outputs). Reading `y` moves the posterior mean of `mu` by a normal step with
covariance

    M = Cov(mu, y) Var(y)^-1 Cov(y, mu),      Var(y) = L (C_B + G S[B, B] G') L' + N

- With a value file, the arm values `V_a` per draw are computed as `eval value` defines them (closed
  form without gates, a fixed set of new runs with gates). Their slope on the arm's means is a least
  squares fit across the draws, so the step in arm values is normal with covariance `Omega = b M b'`.
  The **fall in decision loss** is the knowledge-gradient step `E[max_a (V_a + step_a)] - max_a V_a`,
  computed in closed form as the sum, over every arm but the one that looks best now, of the expected
  gain from switching to it (exact with two arms, an upper bound with more, no sampling noise), and
  never more than the decision loss there is: the value expected to be lost by picking the arm that
  looks best now. Under `best_under_cost` only arms under the cost cap can be picked; the command
  refuses when no arm is, or when an arm has no cost. The objectives `cheapest_within_margin` and
  `cost_per_pass` are refused in active mode: their loss is not built.
- Without a value file, the fall is the **fall in the total variance of the arm means** on the reported
  scale, `trace(J M J')`, with `J` the slope of reported quality on the mean.

What the approximation leaves out: a label also teaches the judges' biases and family effects, and that
part of its worth is not counted. The uniform share below is the guard.

The four scores of an assignment (variances are in units of the component's spread across runs, the
diagonal of `Sigma_input + Sigma_run`, so 1 means "the judges tell nothing"):

- `judge_uncertainty`: the posterior variance of the hidden quality, summed over its components.
- `decision_weight`: the fall in decision loss from this assignment alone. Null without a value file.
- `component_gain`: the conditional variance of each component given the others in the assignment and
  the readings, `Sigma_kk - Sigma_k,rest Sigma_rest^-1 Sigma_rest,k`, averaged over the components.
- `per_minute`: the fall from the whole offer divided by its minutes. Paired targets (the runs of the
  arms on one input) are one offer, drawn together, and share this figure. An offer starts from every
  arm's target on the input and leaves one out at a time while that raises its fall per minute: arms
  still in the running stay together, and a target left out is asked only by the uniform draw.

A component of a score assignment is dropped when its conditional variance given the other components
is under `DROP_BELOW`: with the others labelled, a human adds almost nothing on it. The smallest goes
first. A `human_only` component is never dropped, and one component always stays.

Sampling. Two independent draws, and no drawn plan is over the budget.

- The uniform draw: a simple random sample of single candidates, blind to the scores, each asked on the
  full rubric. Its size is fixed: the uniform share `u` of the budget, at least one draw when `u > 0`
  and the budget pays for it. Every candidate has the same probability `p_u` under it.
- The draw by the scores: whole offers, without replacement. An offer's weight is the square root of
  `per_minute`, floored at `FLOOR` times the mean weight. Offers are grouped by what they cost (their
  assignments, or their minutes), every group gets a fixed number of draws (`allocate`), and within a
  group the draw is systematic in random order with probabilities in proportion to the weights. Its
  budget is what the uniform draw leaves. With a count budget that depends on how many of the uniform
  picks take two assignments (see below), so the layout is made for every such number and the
  probability an offer has under the scores is summed over them.

A candidate is asked when either draw picks it, and its assignment records exactly that probability:
`p_u + P(not in the uniform draw, and drawn by the scores)`. It is of kind `uniform` when the uniform
draw picked it, else `active`. What only the uniform draw asks (the components dropped from a target, and
the targets left out of an offer) has `p_u` as its probability; dropped components come as an assignment
of their own, so a recorded probability holds for every component its assignment asks. Every assignment
of an active plan also records `p_u` as `ext.uniform_probability`.

Offers are whole, so a plan can come in under the budget: by less than the cost of one offer, and by the
targets both draws picked (asked once). With a minutes budget the uniform draw is sized for its dearest
candidates, so what cheaper picks leave is not spent either.

Both modes: `--repeat-share` of the budget goes to repeats (targets this reviewer scored before, a
simple random sample) and, with `--overlap ID`, to overlaps (the second reviewer on targets drawn in
this plan). Their number is what their share pays for whichever are drawn.

Assignment ids carry a digest of every input (the fit, the draws, the value file, and `pairs.jsonl` and
`items.jsonl` of the study folder when they are read), the options, the seed and the assignments
themselves.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..output import EXIT_OK, emit_json
from . import contracts as C
from . import fitfile as F
from . import quality as Q

REPORT_SCHEMA = "loopmath.eval-labels-next/1"
COMMAND = "labels next"
BANDS = 3            # judge score bands of the pilot strata
FLOOR = 0.1          # an offer's weight is at least this share of the mean weight
DROP_BELOW = 0.05    # a component is dropped when this little of its spread stays open given the others
NEW_RUNS = 512       # new runs per posterior draw when the value has gates
STUDY_ROLES = {"runs": C.RUNS_FILE, "scores": C.SCORES_FILE, "pairs": C.PAIRS_FILE, "items": C.ITEMS_FILE}
SCORE_FORM = ("assignment", "run", "step", "component", "judge", "item", "of", "value", "observed_at")
PAIR_FORM = ("assignment", "a", "b", "step", "component", "judge", "shown_first", "choice")
MINUTES = {"score": "review_minutes", "pair": "compare_minutes"}


def _usage(where: str, problem: str, fix: str, code: str = "usage") -> C.ContractError:
    return C.ContractError(f"eval {COMMAND}", where, problem, fix, code)


# ---------------------------------------------------------------------------------------------------
# Candidates

@dataclass
class Target:
    """A (run, step) of the fit: `read[d]` is the fit's record of dimension `d` on it."""

    run: int
    step: str
    refs: dict
    read: dict[int, dict]


@dataclass
class Offer:
    """One assignment a reviewer can be given: a score of one target, or a comparison of two."""

    ask: str                       # "score" or "pair"
    runs: tuple[int, ...]          # run positions: one for a score, (a, b) for a pair
    step: str
    dims: list[int]                # the dimensions asked
    refs: dict
    scores: dict = field(default_factory=lambda: dict.fromkeys(C.LABEL_SCORES))
    gain: float = 0.0              # the fall from this assignment alone
    dropped: list[int] = field(default_factory=list)


@dataclass
class Unit:
    """Offers drawn together: one offer, or the paired targets of one input.

    `rest` holds the targets of the input that add too little per minute to be asked for by the scores.
    They are asked only when the draw is marked `uniform`."""

    offers: list[Offer]
    minutes: float = 0.0
    rest: list[Offer] = field(default_factory=list)
    stratum: tuple = ()
    per_minute: float = 0.0
    p: float = 0.0                 # inclusion probability
    p_uniform: float = 0.0         # the part of it that is blind to the scores
    rank: int = 0


def minutes_of(fit: F.Fit, ask: str, dims: Sequence[int]) -> float:
    return float(sum(fit.dimensions[d][MINUTES[ask]] for d in dims))


def read_targets(fit: F.Fit) -> list[Target]:
    runs = {run: r for r, run in enumerate(fit.index["runs"])}
    dims = {dim["id"]: d for d, dim in enumerate(fit.dimensions)}
    return [Target(runs[row["run"]], row["step"], dict(row.get("refs") or {}),
                   {dims[one["dimension"]]: one for one in row["dims"]}) for row in fit["targets"]]


def _calls(row: dict, judge: str) -> int:
    return int((row.get("judges") or {}).get(judge, 0))


def _judged(row: dict, reviewer: str) -> bool:
    return any(n > 0 for judge, n in (row.get("judges") or {}).items() if judge != reviewer)


def open_dims(target: Target, reviewer: str, keep: set[int], dims: Sequence[dict]) -> list[int]:
    """The dimensions of a target this reviewer can be asked to score."""
    any_judged = any(_judged(row, reviewer) for row in target.read.values())
    return [d for d, row in sorted(target.read.items())
            if d in keep and not _calls(row, reviewer)
            and (_judged(row, reviewer) or (dims[d]["human_only"] and any_judged))]


def score_offers(fit: F.Fit, targets: Sequence[Target], reviewer: str, keep: set[int], arms: set[int]) -> list[Offer]:
    out = []
    for target in targets:
        if int(fit.run_arm[target.run]) in arms:
            mine = open_dims(target, reviewer, keep, fit.dimensions)
            if mine:
                out.append(Offer("score", (target.run,), target.step, mine, dict(target.refs)))
    return out


def pair_dims(fit: F.Fit, named: Sequence[str] | None) -> dict[str, list[int]]:
    """Per step, the dimensions a pair is compared on: those named, else the step's overall component."""
    out: dict[str, list[int]] = {}
    for d, dim in enumerate(fit.dimensions):
        if (dim["component"] in named) if named else dim["role"] == "overall":
            out.setdefault(dim["step"], []).append(d)
    if not out:
        raise _usage("--ask", "the rubric has no overall component to compare outputs on",
                     "name the components to compare with --components", "no_overall")
    return out


def pair_offers(fit: F.Fit, targets: Sequence[Target], reviewer: str, by_step: dict[str, list[int]],
                arms: set[int], done: set[tuple]) -> list[Offer]:
    """Every two targets of one input from different arms. `done` holds (runs, dimension) already compared."""
    groups: dict[tuple, list[Target]] = {}
    for target in targets:
        if int(fit.run_arm[target.run]) in arms and target.step in by_step:
            groups.setdefault((int(fit.run_input[target.run]), target.step), []).append(target)
    out = []
    for (_, step), group in sorted(groups.items()):
        for n, first in enumerate(group):
            for second in group[n + 1:]:
                if fit.run_arm[first.run] == fit.run_arm[second.run]:
                    continue
                runs = frozenset((first.run, second.run))
                mine = [d for d in by_step[step] if (runs, d) not in done
                        and all(d in t.read and _judged(t.read[d], reviewer) for t in (first, second))]
                if mine:
                    out.append(Offer("pair", (first.run, second.run), step, mine,
                                     {"a": dict(first.refs), "b": dict(second.refs)}))
    return out


def pairs_done(fit: F.Fit, study: Path, reviewer: str) -> set[tuple] | None:
    """The (runs, dimension) this reviewer already compared, from the study's `pairs.jsonl`. None without it."""
    path = study / C.PAIRS_FILE
    if not path.is_file():
        return None
    index = {run: r for r, run in enumerate(fit.index["runs"])}
    overall = {dim["step"]: d for d, dim in enumerate(fit.dimensions) if dim["role"] == "overall"}
    position = {dim["id"]: d for d, dim in enumerate(fit.dimensions)}
    done = set()
    for row in C.live(C.read_lines(path, C.SCHEMA_PAIRS), path.name)[0]:
        d = (overall.get(row["step"]) if row["component"] is None
             else position.get(C.dimension_id(row["step"], row["component"])))
        if row["judge"] == reviewer and row["a"] in index and row["b"] in index and d is not None:
            done.add((frozenset((index[row["a"]], index[row["b"]])), d))
    return done


def set_strata(fit: F.Fit, units: Sequence[Unit], by_run: dict[tuple, Target]) -> None:
    """Pilot strata: step, arm (the two arms of a pair) and the band of the judges' plain mean reading.

    Readings are ranked per dimension first, so components on different scales weigh alike. The plain
    mean is the fit's `unpooled` record of the readings, not a model quantity."""
    values: dict[int, dict[int, float]] = {}
    for n, unit in enumerate(units):
        offer = unit.offers[0]
        for d in offer.dims:
            seen = [by_run[(r, offer.step)].read[d]["unpooled"]["mean"] for r in offer.runs
                    if by_run[(r, offer.step)].read.get(d, {}).get("unpooled")]
            if seen:
                values.setdefault(d, {})[n] = float(np.mean(seen))
    score: dict[int, list[float]] = {}
    for d, row in values.items():
        for position, n in enumerate(sorted(row, key=lambda n: (row[n], n))):
            score.setdefault(n, []).append((position + 0.5) / len(row))
    known = sorted(score, key=lambda n: (float(np.mean(score[n])), n))
    band = {n: min(BANDS - 1, BANDS * position // len(known)) for position, n in enumerate(known)}
    for n, unit in enumerate(units):
        offer = unit.offers[0]
        unit.stratum = (offer.step, offer.ask, tuple(sorted(int(fit.run_arm[r]) for r in offer.runs)),
                        band.get(n, BANDS))


# ---------------------------------------------------------------------------------------------------
# The active scores

def _cov(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    return x.T @ x / max(len(x), 1)


def _bounded(matrix: np.ndarray, bound: np.ndarray) -> np.ndarray:
    """`matrix` made symmetric and brought between 0 and `bound` (positive definite) in the matrix order."""
    chol = np.linalg.cholesky(bound)
    white = np.linalg.solve(chol, np.linalg.solve(chol, (matrix + matrix.T) / 2.0).T)
    w, v = np.linalg.eigh((white + white.T) / 2.0)
    return chol @ ((v * np.clip(w, 0.0, 1.0)) @ v.T) @ chol.T


def given_means(p: np.ndarray, total: np.ndarray, own: np.ndarray) -> np.ndarray:
    """The covariance `C` of runs' quality given the arm means, from `C + C W C = P`, `W = T^-1 S T^-1`.

    Solved in closed form, with no iteration: `C = P^1/2 g(P^1/2 W P^1/2) P^1/2`, `g(z) = 2 / (1 + sqrt(1 + 4z))`
    on the eigenvalues. `g` lies in (0, 1], so `C` is positive semidefinite and never above `P`. It is
    then kept under `T`, the spread of the runs around their arm means before any reading."""
    w, v = np.linalg.eigh((p + p.T) / 2.0)
    root = (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T
    inner = np.linalg.solve(total, root)
    z = inner.T @ own @ inner
    w, v = np.linalg.eigh((z + z.T) / 2.0)
    g = 2.0 / (1.0 + np.sqrt(1.0 + 4.0 * np.clip(w, 0.0, None)))
    return _bounded(root @ ((v * g) @ v.T) @ root, total)


def switch_gain(base: np.ndarray, step: np.ndarray) -> float:
    """The expected gain in value from labels that move the arm values by a normal step of covariance `step`.

    For the arm `b` that looks best now and each other arm `a`, the step in `V_a - V_b` is normal with
    sd `s`, and switching to `a` when it comes out ahead is worth `s pdf(g / s) - g cdf(-g / s)`, with `g`
    the gap now. The sum over the other arms is returned: the knowledge-gradient step exactly with two
    arms, an upper bound of it with more. It has no sampling noise and never falls when labels are added."""
    best = int(np.argmax(base))
    total = 0.0
    for a in range(len(base)):
        if a == best:
            continue
        spread = step[a, a] + step[best, best] - 2.0 * step[a, best]
        if spread > 1e-300:
            s, gap = math.sqrt(spread), float(base[best] - base[a])
            total += s * math.exp(-0.5 * (gap / s) ** 2) / math.sqrt(2.0 * math.pi) \
                - gap * 0.5 * math.erfc(gap / s / math.sqrt(2.0))
    return max(total, 0.0)


class Model:
    """The posterior moments the active scores need, and the fall a label is expected to buy."""

    def __init__(self, fit: F.Fit, spec: Q.ValueSpec | None, reviewer: int, seed: int = 0) -> None:
        # `seed` fixes the normal draws of the scores. It is not the plan's seed: the scores of a fit do
        # not change with `--seed`, only the draw does.
        draws = fit.draws
        mu = draws["mu"]
        self.fit, self.spec = fit, spec
        S, A, _ = mu.shape
        fine = (np.isfinite(mu).all(axis=(0, 1)) & np.isfinite(draws["sd_input"]).all(axis=0)
                & np.isfinite(draws["sd_run"]).all(axis=0))
        self.use = np.flatnonzero(fine)                      # dimensions with an estimate
        self.at = {int(d): u for u, d in enumerate(self.use)}
        U = len(self.use)
        self.A, self.U = A, U
        grid = np.ix_(range(S), self.use, self.use)
        self.t_input = Q.covariance(draws["sd_input"][:, self.use], draws["corr_input"][grid]).mean(axis=0)
        self.t_run = Q.covariance(draws["sd_run"][:, self.use], draws["corr_run"][grid]).mean(axis=0)
        total = self.t_input + self.t_run
        self.scale = np.where(np.diag(total) > 0, np.diag(total), 1.0)
        self.q = draws["q_run"][:, :, self.use]
        self.run_ok = np.isfinite(self.q).all(axis=(0, 2))
        self.run_input = fit.run_input
        self.run_arm = fit.run_arm
        s_mu = self.s_mu = _cov(mu[:, :, self.use].reshape(S, A * U))
        self.noise = self._reviewer_noise(draws, reviewer)
        self.pair_noise = self._reviewer_noise(draws, reviewer, pair=True)
        self.values, self.allowed, self.note = None, np.ones(A, dtype=bool), None
        if spec is None:
            sd = np.sqrt(np.diag(total))
            mean, h = mu[:, :, self.use].mean(axis=0), 1e-4
            slope = np.stack([(Q.arm_reported(mean[:, u] + h, sd[u], fit.kinds[d])
                               - Q.arm_reported(mean[:, u] - h, sd[u], fit.kinds[d])) / (2 * h)
                              for u, d in enumerate(self.use)], axis=1)
            self.left = slope.reshape(A * U)[:, None] * s_mu
            self.prior = self.left * slope.reshape(A * U)[None, :]
        else:
            self.values = arm_values(fit, spec, seed)
            self.base = self.values.mean(axis=0)
            beta = np.zeros((A, A * U))
            for a in range(A):
                x = mu[:, a, :][:, self.use]
                beta[a, a * U:(a + 1) * U] = np.linalg.lstsq(x - x.mean(axis=0), self.values[:, a] - self.base[a],
                                                             rcond=None)[0]
            self.left = beta @ s_mu
            self.prior = self.left @ beta.T
            objective = spec.objective
            if objective["kind"] not in ("best_value", "best_under_cost"):
                raise _usage("--value", f'active mode does not score labels for the objective "{objective["kind"]}"',
                             "use --mode pilot, or run active mode without --value (labels are then ranked by "
                             "what they tell about the arm means)", "objective_not_scored")
            if objective["kind"] == "best_under_cost":
                costs = [row.get("cost_usd") for row in fit["arms"]]
                cap = objective["cost_cap_usd"]
                if any(cost is None for cost in costs):
                    raise _usage("--value", "an arm has no cost to hold against the cost cap",
                                 "record cost_usd on its runs and fit again, or use --mode pilot", "no_cost")
                self.allowed = np.array([cost <= cap for cost in costs])
                if not self.allowed.any():
                    raise _usage("--value", f"every arm costs more than the cost cap of {cap:g} USD, so no arm "
                                 "can be picked", "raise objective.cost_cap_usd in the value file, or use "
                                 "--mode pilot", "cost_cap_infeasible")
            # E[max] is never under the max of the means: anything below 0 is rounding.
            ok = self.allowed
            self.loss = max(float(self.values[:, ok].max(axis=1).mean() - self.base[ok].max()), 0.0)
        self._blocks: dict[tuple, tuple] = {}

    def _reviewer_noise(self, draws: dict, reviewer: int, pair: bool = False) -> np.ndarray:
        """The variance a reading by the reviewer adds per usable dimension: 0 where the fit has none."""
        tau = np.nan_to_num((draws["tau"][:, reviewer, :][:, self.use] ** 2).mean(axis=0))
        sigma = (draws["sigma"][:, reviewer, :][:, self.use] ** 2).mean(axis=0)
        if not pair:
            return tau + np.nan_to_num(sigma)
        call = (draws["sigma_pair"][:, reviewer, :][:, self.use] ** 2).mean(axis=0)
        return 2.0 * tau + np.nan_to_num(np.where(np.isfinite(call), call, sigma))

    def expected_loss(self) -> float | None:
        """The value expected to be lost by picking the arm that looks best now."""
        if self.values is None:
            return None
        return self.loss

    def block(self, runs: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(P, the variance of q_B in the joint law used, left x T^-1 C)` for runs of one input."""
        key = tuple(runs)
        if key not in self._blocks:
            m, U = len(key), self.U
            raw = _cov(self.q[:, list(key), :].reshape(-1, m * U))
            same = (self.run_input[list(key)][:, None] == self.run_input[list(key)][None, :]).astype(float)
            total = np.kron(same, self.t_input) + np.kron(np.eye(m), self.t_run)
            total = total + 1e-9 * (np.trace(total) / len(total) + 1e-12) * np.eye(len(total))
            columns = [int(self.run_arm[r]) * U + u for r in key for u in range(U)]
            own = self.s_mu[np.ix_(columns, columns)]
            # P_B lies between 0 and the spread of the runs plus the arm means' own uncertainty. Given
            # the arm means it is C, and q_B = G mu_B + rest with G = C T^-1 and Var(rest) = C. That is a
            # joint normal law of (mu, q_B), so no label moves the arm means by more than they are open.
            given = given_means(_bounded(raw, total + own), total, own)
            pull = np.linalg.solve(total, given)
            self._blocks[key] = (raw, given + pull.T @ own @ pull, self.left[:, columns] @ pull)
        return self._blocks[key]

    def rows(self, runs: Sequence[int], offer: Offer) -> tuple[np.ndarray, np.ndarray]:
        """The linear parts of the block's quality an offer reads, and the reviewer's noise on each."""
        U, place = self.U, {r: n for n, r in enumerate(runs)}
        L = np.zeros((len(offer.dims), len(runs) * U))
        for k, d in enumerate(offer.dims):
            L[k, place[offer.runs[0]] * U + self.at[d]] = 1.0
            if offer.ask == "pair":
                L[k, place[offer.runs[1]] * U + self.at[d]] = -1.0
        noise = self.pair_noise if offer.ask == "pair" else self.noise
        return L, np.array([noise[self.at[d]] for d in offer.dims])

    def fall(self, runs: Sequence[int], offers: Sequence[Offer]) -> float:
        """The expected fall in decision loss (in the variance of the arm means without a value file)
        from the labels of `offers`, all on `runs` of one input."""
        _, clipped, left = self.block(runs)
        parts = [self.rows(runs, offer) for offer in offers]
        L, noise = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        seen = L @ clipped @ L.T + np.diag(noise)
        hit = left @ L.T
        step = hit @ np.linalg.pinv(seen, hermitian=True) @ hit.T
        size, room = np.diag(step), np.diag(self.prior)
        if not np.isfinite(step).all() or (size < -1e-9 * (room.sum() + 1e-300)).any() \
                or (size > room * (1.0 + 1e-6) + 1e-9 * room.sum()).any():
            raise C.ContractError(self.fit.path.name, "draws",
                                  "the posterior draws give no valid update for a label: its step in the arm "
                                  "means is not between 0 and their posterior spread",
                                  "use --mode pilot, or fit again with more draws", "invalid_update")
        if self.values is None:
            return float(min(max(np.trace(step), 0.0), np.trace(self.prior)))
        return min(switch_gain(self.base[self.allowed], step[np.ix_(self.allowed, self.allowed)]), self.loss)

    def open_variance(self, offer: Offer) -> tuple[np.ndarray, np.ndarray]:
        """Per component of an offer, in units of the component's spread across runs: the posterior
        variance of what the offer reads, and its conditional variance given the offer's other components."""
        raw, _, _ = self.block(offer.runs)
        L, _ = self.rows(offer.runs, offer)
        scale = np.sqrt(np.array([self.scale[self.at[d]] for d in offer.dims]))
        cov = (L @ raw @ L.T) / scale[:, None] / scale[None, :]
        precision = np.diag(np.linalg.pinv(cov, hermitian=True))
        conditional = np.where(precision > 0, 1.0 / np.where(precision > 0, precision, 1.0), 0.0)
        return np.diag(cov), np.minimum(conditional, np.diag(cov))

    def score(self, offer: Offer) -> None:
        """Fill the scores of one offer (all but `per_minute`, which belongs to the unit)."""
        variance, conditional = self.open_variance(offer)
        offer.gain = self.fall(offer.runs, [offer])
        offer.scores = {"judge_uncertainty": float(variance.sum()),
                        "decision_weight": None if self.values is None else offer.gain,
                        "component_gain": float(conditional.mean()), "per_minute": None}

    def narrow(self, offers: Sequence[Offer]) -> tuple[list[Offer], float]:
        """The targets of one input worth asking together, and their joint fall per minute.

        Starts from every target and leaves one out at a time while that raises the joint fall per
        minute. Two contested arms stay together (the input's effect cancels between them); an arm that
        is out of the running is left to the uniform share."""
        def rate(group: Sequence[Offer]) -> float:
            runs = tuple(dict.fromkeys(r for offer in group for r in offer.runs))
            return self.fall(runs, group) / sum(minutes_of(self.fit, offer.ask, offer.dims) for offer in group)

        kept, now = list(offers), rate(offers)
        while len(kept) > 1:
            best, n = max((rate(kept[:n] + kept[n + 1:]), n) for n in range(len(kept)))
            if not best > now * (1 + 1e-9):
                break
            kept.pop(n)
            now = best
        return kept, now

    def usable(self, offer: Offer) -> bool:
        return bool(offer.dims) and all(self.run_ok[r] for r in offer.runs) and all(d in self.at for d in offer.dims)

    def trim(self, offer: Offer) -> None:
        """Drop the components a human adds almost nothing on, given the offer's other components."""
        dims = self.fit.dimensions
        while len(offer.dims) > 1:
            _, conditional = self.open_variance(offer)
            free = [(conditional[k], d) for k, d in enumerate(offer.dims)
                    if not dims[d]["human_only"] and conditional[k] < DROP_BELOW]
            if not free:
                break
            gone = min(free)[1]
            offer.dims = [d for d in offer.dims if d != gone]
            offer.dropped.append(gone)


def arm_values(fit: F.Fit, spec: Q.ValueSpec, seed: int = 0) -> np.ndarray:
    """The expected value of a new run on a new input, per posterior draw and arm: `[S, A]`.

    Without gates it is `w'E[m]` in closed form. With gates it is the mean of `V(q)` over `NEW_RUNS`
    new runs, `q ~ N(mu_arm, Sigma_input + Sigma_run)`, from one fixed set of normal draws shared by
    every arm and every posterior draw."""
    draws = fit.draws
    need = sorted(set(np.flatnonzero(spec.weights).tolist()) | {d for d, _ in spec.gates})
    mu = draws["mu"][:, :, need]
    sd_input, sd_run = draws["sd_input"][:, need], draws["sd_run"][:, need]
    for k, d in enumerate(need):
        if not (np.isfinite(mu[:, :, k]).all() and np.isfinite(sd_input[:, k]).all()
                and np.isfinite(sd_run[:, k]).all()):
            raise C.ContractError(fit.path.name, f"draws: {fit.dimensions[d]['id']}",
                                  "the fit holds no estimate for a dimension the value file weights or gates",
                                  "fit with this component, or drop it from the value file", "no_estimate")
    S, A, K = mu.shape
    kinds = [fit.kinds[d] for d in need]
    if not spec.gates:
        total = np.sqrt(sd_input ** 2 + sd_run ** 2)
        out = np.zeros((S, A))
        for k, d in enumerate(need):
            out += spec.weights[d] * Q.arm_reported(mu[:, :, k], total[:, None, k], kinds[k])
        return out
    grid = np.ix_(range(S), need, need)
    cov = Q.covariance(sd_input, draws["corr_input"][grid]) + Q.covariance(sd_run, draws["corr_run"][grid])
    w, v = np.linalg.eigh(cov)
    roots = v * np.sqrt(np.clip(w, 0.0, None))[:, None, :]
    z = np.random.default_rng([int(seed), 11]).standard_normal((NEW_RUNS // 2, K))
    z = np.concatenate([z, -z])
    at = {d: k for k, d in enumerate(need)}
    out = np.empty((S, A))
    for s in range(S):
        m = Q.reported_all(mu[s][:, None, :] + (z @ roots[s].T)[None, :, :], kinds)
        ok = np.ones(m.shape[:-1], dtype=bool)
        for d, threshold in spec.gates:
            ok &= m[..., at[d]] >= threshold
        weighted = np.zeros(m.shape[:-1])
        for d in np.flatnonzero(spec.weights):
            weighted += spec.weights[d] * m[..., at[int(d)]]
        out[s] = np.where(ok, weighted, spec.gated_value).mean(axis=1)
    return out


# ---------------------------------------------------------------------------------------------------
# Sampling

def inclusion(p: np.ndarray, k: int) -> np.ndarray:
    """Inclusion probabilities proportional to `p` that sum to `k`, none above 1."""
    p = np.asarray(p, dtype=float)
    if k >= len(p):
        return np.ones(len(p))
    full = np.zeros(len(p), dtype=bool)
    while True:
        rest = p[~full].sum()
        pi = np.where(full, 1.0, (k - full.sum()) * p / rest if rest > 0 else 0.0)
        over = (pi > 1.0 - 1e-12) | full
        if (over == full).all():
            return pi
        full = over


def systematic(pi: np.ndarray, order: np.ndarray, rng: np.random.Generator) -> list[int]:
    """A fixed-size sample with exactly the inclusion probabilities `pi`: systematic over `order`."""
    k = int(round(float(pi.sum())))
    if k == 0:
        return []
    edges = np.cumsum(pi[order])
    points = rng.random() + np.arange(k)
    return [int(order[min(n, len(order) - 1)]) for n in np.searchsorted(edges, points, side="right")]


def allocate(cost: np.ndarray, base: np.ndarray,
             budget: float) -> tuple[np.ndarray, dict[float, tuple[np.ndarray, int]]]:
    """Inclusion probabilities in proportion to `base` for a draw that costs the same whatever comes up.

    Units are grouped by cost and each group gets a fixed number of draws: its share of the most draws
    whose expected cost the budget pays for, rounded down, then one more draw at a time for the group
    rounded down the most, while the budget pays for it; what is then left moves single draws to dearer
    groups. Returns the exact inclusion probabilities and,
    per cost, the positions of the group and its number of draws. Nothing here is random."""
    N = len(cost)
    pi = np.zeros(N)
    low, high = 0, N
    while low < high:
        k = (low + high + 1) // 2
        if float((inclusion(base, k) * cost).sum()) <= budget + 1e-9:
            low = k
        else:
            high = k - 1
    want = inclusion(base, low) if low else np.zeros(N)
    groups: dict[float, list[int]] = {}
    for n in range(N):
        groups.setdefault(round(float(cost[n]), 9), []).append(n)
    take = {c: int(np.floor(want[idx].sum() + 1e-9)) for c, idx in groups.items()}
    left = budget - sum(c * k for c, k in take.items())
    while True:
        room = [(float(want[idx].sum()) - take[c], -c) for c, idx in groups.items()
                if take[c] < len(idx) and c <= left + 1e-9]
        if not room:
            break
        c = -max(room)[1]
        take[c] += 1
        left -= c
    while left > 1e-9:
        # What is still left: move one draw to a dearer group when the budget pays for the difference.
        moves = [(b - a, a, b) for a in groups for b in groups
                 if take[a] and take[b] < len(groups[b]) and 1e-9 < b - a <= left + 1e-9]
        if not moves:
            break
        _, a, b = max(moves)
        take[a], take[b], left = take[a] - 1, take[b] + 1, left - (b - a)
    for c, idx in groups.items():
        if take[c]:
            share = base[idx] if base[idx].sum() > 0 else np.ones(len(idx))
            pi[idx] = inclusion(share, take[c])
    return pi, {c: (np.array(idx, dtype=int), take[c]) for c, idx in groups.items()}


def draw_groups(pi: np.ndarray, groups: dict[float, tuple[np.ndarray, int]], rng: np.random.Generator,
                strata: Sequence[tuple] | None = None) -> list[int]:
    """Draw what `allocate` laid out: a systematic draw within every cost group, in random order (in
    random order within the strata in turn when `strata` is given)."""
    drawn: list[int] = []
    for c in sorted(groups):
        idx, k = groups[c]
        if k:
            order = rng.permutation(len(idx))
            if strata is not None:
                order = np.array(sorted(order.tolist(), key=lambda n: strata[int(idx[n])]), dtype=int)
            drawn += [int(idx[n]) for n in systematic(pi[idx], order, rng)]
    return drawn


def _ways(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def worst_case_count(costs: Sequence[float], amount: float) -> int:
    """How many of a pool the amount pays for whichever are drawn: the dearest ones are counted first."""
    return int(np.searchsorted(np.cumsum(np.sort(np.asarray(costs, dtype=float))[::-1]), amount + 1e-9, side="right"))


# ---------------------------------------------------------------------------------------------------
# The plan

def pick_reviewer(fit: F.Fit, wanted: str | None, option: str = "--reviewer") -> str:
    persons = [row["id"] for row in fit["judges"] if row["kind"] == "person"]
    if wanted is None:
        if len(persons) == 1:
            return persons[0]
        problem = "the fit has no person judge" if not persons else f"the fit has {len(persons)} person judges"
        fix = ("add the reviewer to the judges file as kind person and fit again" if not persons
               else f"name one with {option}: {', '.join(persons)}")
        raise _usage(option, problem, fix, "no_reviewer")
    row = next((row for row in fit["judges"] if row["id"] == wanted), None)
    if row is None or row["kind"] != "person":
        problem = f'unknown judge "{wanted}"' if row is None else f'the judge "{wanted}" is of kind "{row["kind"]}"'
        fix = (f"use one of the person judges: {', '.join(persons)}" if persons
               else "add the reviewer to the judges file as kind person and fit again")
        raise _usage(option, problem, fix, "bad_reviewer")
    return wanted


def stale_warnings(fit: F.Fit, study: Path, strict: bool) -> list[str]:
    """What changed between the digests the fit recorded and the study folder now. A refusal with `strict`."""
    recorded, files = fit["inputs"], {}
    for role, name in STUDY_ROLES.items():
        if role in recorded or (study / name).is_file():
            files[role] = study / (recorded[role]["file"] if role in recorded else name)
    for role in recorded:
        if role not in STUDY_ROLES and (study / recorded[role]["file"]).is_file():
            files[role] = study / recorded[role]["file"]
    changes = C.stale_inputs({role: row for role, row in recorded.items() if role in files}, files)
    if changes and strict:
        raise C.ContractError(fit.path.name, "inputs", "; ".join(changes), "fit again, or drop --strict", "stale")
    return changes


def identifying(fit: F.Fit, offers: Sequence[Offer]) -> int:
    """How many offers have a run id or a ref that spells out an arm, a model id or a model family.

    A name of three or more characters counts wherever it appears in the text. A shorter name counts
    when it is a whole word of the text (split at anything that is not a letter or a digit)."""
    words = {str(arm).lower() for arm in fit.arms} | {str(f).lower() for f in fit.index["families"]}
    for arm in fit["arms"]:
        words |= {str(step["model"]["id"]).lower() for step in arm.get("steps", [])}
    long, short = {word for word in words if len(word) >= 3}, {word for word in words if 0 < len(word) < 3}

    def texts(value: Any):
        if isinstance(value, dict):
            for one in value.values():
                yield from texts(one)
        elif isinstance(value, str):
            yield value.lower()

    def names(text: str) -> bool:
        return any(word in text for word in long) or bool(short & set(re.split(r"[^0-9a-z]+", text)))

    return sum(any(names(text) for text in [str(fit.index["runs"][r]).lower() for r in offer.runs]
                   + list(texts(offer.refs))) for offer in offers)


def all_simulated(fit: F.Fit, study: Path, offers: Sequence[Offer]) -> bool:
    """Whether the study's run records mark every run of the offers as simulated.

    False when the study folder has no run records, when a run has no record there, or when any record
    of a run says it is not simulated: one real run keeps the blind refusal."""
    path = study / C.RUNS_FILE
    if not path.is_file():
        return False
    marked: dict[str, bool] = {}
    for row in C.read_lines(path, C.SCHEMA_RUNS):
        marked[row["run"]] = marked.get(row["run"], True) and bool(row.get("simulated", False))
    return all(marked.get(fit.index["runs"][r], False) for offer in offers for r in offer.runs)


def _budget(args: argparse.Namespace) -> tuple[str, float]:
    n, minutes = getattr(args, "n", None), getattr(args, "minutes", None)
    if (n is None) == (minutes is None):
        raise _usage("--n", "both --n and --minutes are given" if n is not None else "no budget is given",
                     "give the budget as --n assignments or as --minutes, one of the two")
    if n is not None and n < 1:
        raise _usage("--n", f"{n} is out of range", "give a whole number at least 1")
    if minutes is not None and not minutes > 0:
        raise _usage("--minutes", f"{minutes:g} is out of range", "give a number above 0")
    return ("n", float(n)) if n is not None else ("minutes", float(minutes))


def _share(args: argparse.Namespace, name: str, default: float, top: bool = False) -> float:
    """A share option. `top` lets it be 1: a uniform share of 1 draws every offer with the same probability."""
    value = getattr(args, name, None)
    value = default if value is None else float(value)
    if not (0.0 <= value <= 1.0 if top else 0.0 <= value < 1.0):
        raise _usage("--" + name.replace("_", "-"), f"{value:g} is out of range",
                     "give a share from 0 to 1" if top else "give a share from 0 up to 1, 1 left out")
    return value


def _digest(*parts: Any) -> str:
    return hashlib.sha256(C.dumps(list(parts)).encode("utf-8")).hexdigest()


@dataclass
class Result:
    """What `build` returns: the plan, the report, and what `--explain` and `--form` read."""

    plan: dict
    report: dict
    explain: dict[str, dict]
    fit: F.Fit
    study: Path


def build(args: argparse.Namespace) -> Result:
    """Read the inputs, score the candidates, draw the plan. Raises `ContractError` on a refusal."""
    mode = getattr(args, "mode", None) or "pilot"
    ask = getattr(args, "ask", None) or "scores"
    seed = int(getattr(args, "seed", 0) or 0)
    unit_name, budget = _budget(args)
    uniform_share = _share(args, "uniform_share", 0.2, top=True)
    repeat_share = _share(args, "repeat_share", 0.1)
    if mode == "pilot" and ask == "auto":
        raise _usage("--ask", "auto needs the scores of active mode to weigh scores against pairs",
                     "use --mode active, or ask for scores or for pairs")
    fit = F.read_fit(args.fit, allow_unconverged=bool(getattr(args, "allow_unconverged", False)) or mode == "pilot")
    study = Path(args.study) if getattr(args, "study", None) else fit.path.parent
    warnings, notes = [], []
    if not fit.converged:
        warnings.append("the fit did not converge" + (": pilot mode reads no model quantity from it"
                                                       if mode == "pilot" else ": read every score with care"))
    warnings += stale_warnings(fit, study, bool(getattr(args, "strict", False)))
    reviewer = pick_reviewer(fit, getattr(args, "reviewer", None))
    second = getattr(args, "overlap", None)
    if second is not None:
        second = pick_reviewer(fit, second, "--overlap")
        if second == reviewer:
            raise _usage("--overlap", f'"{second}" is the reviewer', "name a second person")
    spec, value_path = None, getattr(args, "value", None)
    if value_path is not None:
        spec = Q.compile_value(C.read_document(value_path, C.SCHEMA_VALUE), fit.dimensions, Path(value_path).name)
        if mode == "pilot":
            notes.append("pilot mode does not read the value file")
    dims = fit.dimensions
    named = getattr(args, "components", None)
    for name in named or []:
        if not any(dim["component"] == name for dim in dims):
            known = ", ".join(dict.fromkeys(dim["component"] for dim in dims))
            raise _usage("--components", f'unknown component "{name}"', f"use one of {known}", "unknown_component")
    keep = {d for d, dim in enumerate(dims) if not named or dim["component"] in named}
    arm_names = getattr(args, "arms", None)
    arms = {fit.arm(name) for name in arm_names} if arm_names else set(range(len(fit.arms)))

    targets = read_targets(fit)
    by_run = {(target.run, target.step): target for target in targets}
    offers: list[Offer] = []
    if ask in ("scores", "auto"):
        offers += score_offers(fit, targets, reviewer, keep, arms)
    if ask in ("pairs", "auto"):
        done = pairs_done(fit, study, reviewer)
        if done is None:
            notes.append(f"no {C.PAIRS_FILE} beside the fit: pairs this reviewer already judged are not known "
                         "and may be offered again")
        offers += pair_offers(fit, targets, reviewer, pair_dims(fit, named), arms, done or set())

    model = None
    if mode == "active":
        model = Model(fit, spec, fit.judge(reviewer))
        n_before = len(offers)
        for offer in offers:
            if offer.ask == "score":         # a score keeps the components the fit has an estimate for
                offer.dims = [d for d in offer.dims if d in model.at]
        offers = [offer for offer in offers if model.usable(offer)]
        if len(offers) < n_before:
            notes.append(f"{n_before - len(offers)} candidates are left out of active mode: the fit holds no "
                         "estimate for them (pilot mode can draw them)")
        if model.note:
            notes.append(model.note)
    if not offers:
        what = {"scores": "target", "pairs": "pair", "auto": "target or pair"}[ask]
        raise _usage("candidates", f'no {what} is left for "{reviewer}" to label',
                     "widen --components or --arms, name another reviewer, or add judge readings and fit again",
                     "no_candidates")

    # Offers into units: in active mode the paired score targets of one input are drawn together.
    full = {id(offer): list(offer.dims) for offer in offers}
    units: list[Unit] = []
    if mode == "active":
        blocks: dict[tuple, list[Offer]] = {}
        seen: dict[tuple, int] = {}
        for offer in offers:
            if offer.ask == "pair":
                model.score(offer)
                units.append(Unit([offer]))
                continue
            model.trim(offer)
            model.score(offer)
            r = offer.runs[0]
            cell = (int(fit.run_input[r]), offer.step, int(fit.run_arm[r]))
            seen[cell] = seen.get(cell, 0) + 1
            blocks.setdefault((cell[0], cell[1], seen[cell]), []).append(offer)
        units += [Unit(group) for _, group in sorted(blocks.items())]
        for unit in units:
            kept, unit.per_minute = model.narrow(unit.offers)
            unit.rest = [offer for offer in unit.offers if not any(offer is one for one in kept)]
            unit.offers = kept
            unit.minutes = float(sum(minutes_of(fit, offer.ask, offer.dims) for offer in kept))
            for offer in kept:
                offer.scores["per_minute"] = unit.per_minute
            for offer in unit.rest:
                offer.scores["per_minute"] = offer.gain / minutes_of(fit, offer.ask, offer.dims)
    else:
        units = [Unit([offer], minutes_of(fit, offer.ask, offer.dims)) for offer in offers]
        set_strata(fit, units, by_run)

    # The budget: a share for repeats and overlaps, the rest for the main draw.
    rng = np.random.default_rng(seed)
    again = []
    for target in targets:
        mine = [d for d, row in sorted(target.read.items()) if d in keep and _calls(row, reviewer)]
        if mine and int(fit.run_arm[target.run]) in arms and ask != "pairs":
            again.append(Offer("score", (target.run,), target.step, mine, dict(target.refs)))
    extra = repeat_share * budget if (again or second) else 0.0
    if unit_name == "n":
        extra = float(int(round(extra)))
    if repeat_share > 0 and not (again or second):
        notes.append("no repeats or overlaps: this reviewer has scored no target yet and --overlap names no "
                     "second reviewer, so their share of the budget goes to the main draw")
    # Blind or nothing: a run id or a ref that names the configuration would tell the reviewer the arm.
    n_named = identifying(fit, offers + again)
    simulated = n_named > 0 and all_simulated(fit, study, offers + again)
    if simulated:
        notes.append(f"the plan is not blind: the study is simulated, and {n_named} of {len(offers) + len(again)} "
                     "candidates have a run id or a ref that spells out the configuration. With real runs this is "
                     "refused")
    elif n_named:
        raise C.ContractError(fit.path.name, "runs", f"{n_named} of {len(offers) + len(again)} candidates have a run "
                              "id or a ref that spells out an arm, a model id or a model family, so the plan "
                              "would not be blind",
                              "give the runs ids and refs that do not name the configuration (r001, outputs/r001), "
                              "import the scores under those ids and fit again", "not_blind")
    by_count = unit_name == "n"
    main_budget = budget - extra
    groups: list[list[dict]] = []
    explain: dict[int, dict] = {}

    def record(offer: Offer, kind: str, p: float, who: str, unit: Unit | None = None,
               asked: Sequence[int] | None = None, p_scores: float | None = None) -> dict:
        asked = offer.dims if asked is None else list(asked)
        flip = bool(rng.integers(2)) if offer.ask == "pair" else False
        runs = offer.runs[::-1] if flip else offer.runs
        row: dict = {"ask": offer.ask}
        if offer.ask == "score":
            row["target"] = {"run": fit.index["runs"][runs[0]], "step": offer.step}
            refs = dict(offer.refs)
        else:
            row["a"] = {"run": fit.index["runs"][runs[0]], "step": offer.step}
            row["b"] = {"run": fit.index["runs"][runs[1]], "step": offer.step}
            row["shown_first"] = "a" if rng.integers(2) else "b"
            refs = {"a": offer.refs["b" if flip else "a"], "b": offer.refs["a" if flip else "b"]}
        row.update({"components": [dims[d]["component"] for d in asked], "reviewer": who, "kind": kind,
                    "inclusion_probability": float(p),
                    "scores": dict(offer.scores) if kind in ("active", "uniform") else dict.fromkeys(C.LABEL_SCORES),
                    "minutes": minutes_of(fit, offer.ask, asked), "refs": refs})
        if mode == "active" and unit is not None:
            row["ext"] = {"uniform_probability": unit.p_uniform}
        explain[id(row)] = {"offer": offer, "unit": unit, "row": row, "extra": asked != offer.dims,
                            "p_scores": p_scores}
        return row

    main = []
    n_uniform = 0
    if mode == "pilot":
        # One probability for every candidate of one cost. With a count budget that is every candidate.
        cost = np.array([1.0 if by_count else unit.minutes for unit in units])
        pi, laid = allocate(cost, np.full(len(units), 1.0 / len(units)), main_budget)
        for n, unit in enumerate(units):
            unit.p = float(pi[n])
        for n in draw_groups(pi, laid, rng, [unit.stratum for unit in units]):
            row = record(units[n].offers[0], "pilot", units[n].p, reviewer, units[n])
            groups.append([row])
            main.append((row, units[n].offers[0], units[n]))
    else:
        # The uniform draw: a simple random sample of single candidates, each on the full rubric. A
        # candidate whose offer leaves components out comes as two assignments (what the scores ask, and
        # the rest), so in a count budget it takes one assignment more.
        split = {id(offer) for unit in units for offer in unit.offers if offer.dropped}
        heavy = np.array([by_count and id(offer) in split for offer in offers])
        N, M = len(offers), int(heavy.sum())
        top = 1.0 if by_count else float(max(minutes_of(fit, offer.ask, full[id(offer)]) for offer in offers))
        if uniform_share > 0:
            each = top * (1.0 + M / N)
            n_uniform = min(N, max(1, int(round(uniform_share * main_budget / each))))
            while n_uniform and (n_uniform + min(M, n_uniform)) * top > main_budget + 1e-9:
                n_uniform -= 1
            if not n_uniform:
                notes.append("the budget does not pay for one draw of the uniform share: no uniform assignment")
        p_blind = n_uniform / N
        weight = np.sqrt(np.maximum([unit.per_minute for unit in units], 0.0))
        weight = weight / weight.sum() if weight.sum() > 0 else np.full(len(units), 1.0 / len(units))
        weight = np.maximum(weight, FLOOR / len(units))
        cost = np.array([len(unit.offers) if by_count else unit.minutes for unit in units], dtype=float)
        # What the uniform draw leaves is the budget of the draw by the scores. With a count budget it
        # depends on j, the number of two-assignment candidates the uniform draw picked, so the draw by
        # the scores is laid out for every j, and a recorded probability is the sum over j.
        most = min(M, n_uniform)
        laid = [allocate(cost, weight / weight.sum(),
                         0.0 if uniform_share >= 1 else main_budget - (n_uniform + j) * top)
                for j in range(most + 1)]
        for rank, n in enumerate(sorted(range(len(units)), key=lambda n: (-units[n].per_minute, n)), 1):
            units[n].rank = rank
        ways = _ways(N, n_uniform)
        at = {id(offer): n for n, unit in enumerate(units) for offer in unit.offers}
        p_scores = {}                                    # P(not in the uniform draw, and drawn by the scores)
        for position, offer in enumerate(offers):
            if id(offer) in at:
                h = int(heavy[position])
                p_scores[id(offer)] = float(sum(
                    _ways(M - h, j) * _ways(N - M - (1 - h), n_uniform - j) / ways * laid[j][0][at[id(offer)]]
                    for j in range(most + 1)))
        for n, unit in enumerate(units):
            unit.p, unit.p_uniform = float(laid[0][0][n]), p_blind
        picks = sorted(rng.choice(N, size=n_uniform, replace=False).tolist()) if n_uniform else []
        pi, groups_now = laid[int(heavy[picks].sum()) if picks else 0]
        picked = draw_groups(pi, groups_now, rng)
        blind = {id(offers[n]) for n in picks}
        home = {id(offer): unit for unit in units for offer in unit.offers + unit.rest}
        taken = {id(offer) for n in picked for offer in units[n].offers}
        by_unit: dict[int, list] = {}
        for offer in offers:
            if id(offer) not in taken and id(offer) not in blind:
                continue
            unit, mine = home[id(offer)], p_scores.get(id(offer), 0.0)
            either = p_blind + mine                      # asked when either draw picks it
            given = mine / (1.0 - p_blind) if p_blind < 1 else 0.0
            if id(offer) not in blind:
                group = [record(offer, "active", either, reviewer, unit, p_scores=given)]
            elif id(offer) in split:
                # What the scores left out is asked under the uniform draw only: an assignment of its own,
                # so a recorded probability holds for every component its assignment asks.
                others = [d for d in full[id(offer)] if d not in offer.dims]
                group = [record(offer, "uniform", either, reviewer, unit, p_scores=given),
                         record(offer, "uniform", p_blind, reviewer, unit, others, p_scores=0.0)]
            else:
                group = [record(offer, "uniform", either, reviewer, unit, full[id(offer)], p_scores=given)]
            key = id(unit) if id(offer) in taken else id(offer)
            by_unit.setdefault(key, []).extend(group)
            main += [(row, offer, unit) for row in group]
        groups += list(by_unit.values())
    if not main:
        raise _usage("--" + unit_name, f"the budget of {budget:g} {'assignments' if unit_name == 'n' else 'minutes'} "
                     "pays for no assignment", "raise the budget", "budget_too_small")

    def count(pool: Sequence[Offer], amount: float) -> int:
        """How many of `pool` the amount pays for, whichever are drawn."""
        if not pool or amount <= 0:
            return 0
        if unit_name == "n":
            return min(len(pool), int(round(amount)))
        return min(len(pool), worst_case_count([minutes_of(fit, offer.ask, offer.dims) for offer in pool], amount))

    pool = [one for one in main if second and not any(
        _calls(by_run[(r, one[1].step)].read.get(d, {}), second) for r in one[1].runs for d in one[1].dims)]
    n_overlap = count([offer for _, offer, _ in pool], extra / 2 if again else extra)
    spent = n_overlap if unit_name == "n" else (
        float(np.sort([row["minutes"] for row, _, _ in pool])[::-1][:n_overlap].sum()) if n_overlap else 0.0)
    n_repeat = count(again, extra - spent)
    for n in sorted(rng.choice(len(again), size=n_repeat, replace=False).tolist()) if n_repeat else []:
        groups.append([record(again[n], "repeat", n_repeat / len(again), reviewer)])
    for n in sorted(rng.choice(len(pool), size=n_overlap, replace=False).tolist()) if n_overlap else []:
        row, offer, unit = pool[n]
        copy = dict(row, reviewer=second, kind="overlap", scores=dict.fromkeys(C.LABEL_SCORES),
                    inclusion_probability=float(row["inclusion_probability"] * n_overlap / len(pool)))
        explain[id(copy)] = {"offer": offer, "unit": unit, "row": copy, "extra": True, "p_scores": None}
        groups.append([copy])

    # Order and ids: groups in random order, so the plan's order says nothing about arm or stratum.
    options = {"mode": mode, "ask": ask, unit_name: budget if unit_name == "minutes" else int(budget),
               "reviewer": reviewer, "overlap": second, "components": list(named) if named else None,
               "arms": ({"n": len(arm_names), "sha256": _digest(sorted(arm_names))} if arm_names else None),
               "uniform_share": uniform_share, "repeat_share": repeat_share, "value": value_path is not None,
               "floor": FLOOR, "drop_below": DROP_BELOW}
    files: dict[str, Any] = {"fit": fit.path, "draws": fit.path.with_name(fit["draws"]["file"])}
    if value_path is not None:
        files["value"] = value_path
    # Every file that decides the candidates or the form is in the provenance and in the ids.
    if ask in ("pairs", "auto") and (study / C.PAIRS_FILE).is_file():
        files["pairs"] = study / C.PAIRS_FILE
    if (study / C.ITEMS_FILE).is_file():
        files["items"] = study / C.ITEMS_FILE
    if simulated:
        files["runs"] = study / C.RUNS_FILE          # its simulated marks let the plan through
    inputs = C.input_digests(files)
    rows = [groups[g][n] for g in rng.permutation(len(groups)).tolist()
            for n in rng.permutation(len(groups[g])).tolist()]
    # The ids carry a digest of the inputs, the options, the seed and the assignments themselves: two
    # plans share an id only when they ask the same things in the same order.
    prefix = "lb" + _digest(inputs, options, seed, rows)[:6]
    together: dict[int, list[str]] = {}
    for n, row in enumerate(rows, 1):
        entry = explain[id(row)]
        entry["id"] = f"{prefix}-{n:03d}"
        if entry["unit"] is not None and row["kind"] != "overlap":
            together.setdefault(id(entry["unit"]), []).append(entry["id"])
        rows[n - 1] = {"id": entry["id"], **row}
    plan = C.validate_labels({"schema": C.SCHEMA_LABELS, "inputs": inputs, "options": options, "seed": seed,
                              "mode": mode, "reviewer": reviewer, "assignments": rows})

    by_id = {}
    for entry in explain.values():
        offer, unit, row = entry["offer"], entry["unit"], entry["row"]
        one = {"id": entry["id"], "kind": row["kind"], "ask": offer.ask, "components": row["components"],
               "dropped": [] if entry["extra"] else [dims[d]["component"] for d in offer.dropped],
               "scores": row["scores"], "minutes": row["minutes"],
               "inclusion_probability": row["inclusion_probability"]}
        if unit is not None:
            one.update({"candidates": len(units), "offer_minutes": unit.minutes,
                        "offer_probability": unit.p if entry["p_scores"] is None else entry["p_scores"],
                        "uniform_part": unit.p_uniform, "rank": unit.rank or None,
                        "offered_with": [x for x in together.get(id(unit), []) if x != entry["id"]]})
        by_id[entry["id"]] = one

    kinds = {kind: sum(row["kind"] == kind for row in plan["assignments"]) for kind in C.ASSIGNMENT_KINDS}
    report = {"experimental": True, "mode": mode, "ask": ask, "reviewer": reviewer, "seed": seed, "options": options,
              "candidates": {"offers": len(units), "assignments": len(offers), "repeats": len(again),
                             "left_to_uniform": sum(len(unit.rest) for unit in units), "uniform_draws": n_uniform},
              "assignments": len(plan["assignments"]), "kinds": {k: v for k, v in kinds.items() if v},
              "minutes": float(sum(row["minutes"] for row in plan["assignments"])),
              "budget": {"unit": "assignments" if unit_name == "n" else "minutes", "amount": budget},
              "expected_loss": model.expected_loss() if model is not None else None,
              "inputs": inputs, "notes": notes, "warnings": warnings}
    return Result(plan, report, by_id, fit, study)


# ---------------------------------------------------------------------------------------------------
# The form

def _csv_text(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def forms(result: Result) -> tuple[str | None, str | None]:
    """The blank forms of a plan as CSV text: `(readings, pairs)`, each None when the plan asks for none.

    Readings: one row per (assignment, component), and one row per check item of an `items` component
    when the study folder's `items.jsonl` lists items for the input (else one row to fill as a count,
    `value` passed out of `of`). Pairs: one row per (pair, component); `choice` takes left, right or tie,
    where left is the output named by `shown_first`."""
    fit = result.fit
    kinds = {dim["component"]: dim["scale"]["kind"] for dim in fit.dimensions}
    run_input = {run: fit.index["inputs"][i] for run, i in zip(fit.index["runs"], fit.index["run_input"])}
    items: dict[tuple, list[str]] = {}
    for row in C.read_lines(result.study / C.ITEMS_FILE, C.SCHEMA_ITEMS, missing_ok=True):
        items.setdefault((row["input"], row["component"]), []).append(row["id"])
    readings, pairs = [], []
    for row in result.plan["assignments"]:
        for component in row["components"]:
            if row["ask"] == "pair":
                pairs.append([row["id"], row["a"]["run"], row["b"]["run"], row["a"]["step"], component,
                              row["reviewer"], row["shown_first"], ""])
                continue
            run, step = row["target"]["run"], row["target"]["step"]
            listed = items.get((run_input[run], component), []) if kinds.get(component) == "items" else []
            for item in listed or [""]:
                readings.append([row["id"], run, step, component, row["reviewer"], item, "", "", ""])
    return (_csv_text(SCORE_FORM, readings) if readings else None, _csv_text(PAIR_FORM, pairs) if pairs else None)


def pairs_form_path(form: Path) -> Path:
    """Where the pairs form goes when a plan asks for scores and pairs: `labels.csv` and `labels.pairs.csv`."""
    return form.with_name(form.stem + ".pairs" + form.suffix)


# ---------------------------------------------------------------------------------------------------
# The command

def _fmt(value: float | None) -> str:
    return "none" if value is None else f"{value:.4g}"


def explain_lines(one: dict) -> list[str]:
    """Why one assignment was ranked where it was, in words. Blind like the plan."""
    lines = [f"{one['id']}: a {one['ask']} assignment of kind {one['kind']}, components "
             f"{', '.join(one['components'])}, {one['minutes']:g} minutes"]
    if one["kind"] == "pilot":
        lines.append(f"pilot mode draws every candidate with the same probability, "
                     f"{one['inclusion_probability']:.4g} over {one['candidates']} candidates, stratified by arm "
                     "and judge score band. No model quantity is used.")
        return lines
    if one["kind"] == "repeat":
        lines.append("a repeat of a target this reviewer scored before, drawn at random with probability "
                     f"{one['inclusion_probability']:.4g}. It measures the reviewer's own noise.")
        return lines
    scores = one["scores"]
    if one["kind"] == "overlap":
        lines.append("an overlap: a second reviewer labels a target drawn in this plan, to measure the noise "
                     f"between reviewers. Probability {one['inclusion_probability']:.4g}.")
    elif one.get("rank"):
        lines.append(f"its offer ranks {one['rank']} of {one['candidates']} by per_minute "
                     f"({one['offer_minutes']:g} minutes for the whole offer)")
        lines.append(f"judge_uncertainty {_fmt(scores['judge_uncertainty'])}: how open the judges leave this "
                     "quality (1 per component is as open as a run nobody scored)")
        lines.append(f"decision_weight {_fmt(scores['decision_weight'])}: the fall in decision loss if this "
                     "assignment alone were labelled" + (" (none: no value file was given)"
                                                          if scores["decision_weight"] is None else ""))
        lines.append(f"component_gain {_fmt(scores['component_gain'])}: what each component still holds given "
                     "the others")
        lines.append(f"per_minute {_fmt(scores['per_minute'])}: the fall from the whole offer per minute")
    if one.get("offer_probability") is not None:
        lines.append(f"offer probability {one['offer_probability']:.4g} under the scores, and "
                     f"{one['uniform_part']:.4g} under the uniform draw every candidate has")
    if one["kind"] == "uniform":
        lines.append("picked by the uniform draw: a simple random sample of single candidates, blind to the scores, "
                     "on the full rubric")
    if one.get("dropped"):
        lines.append(f"left out, because the other components tell enough: {', '.join(one['dropped'])}")
    if one.get("offered_with"):
        lines.append(f"offered together with {', '.join(one['offered_with'])} (the same input)")
    return lines


def run(args: argparse.Namespace) -> int:
    result = build(args)
    report, plan = result.report, result.plan
    wanted = getattr(args, "explain", None)
    out, form = getattr(args, "out", None), getattr(args, "form", None)
    written: dict[str, str] = {}
    if wanted is not None:
        if wanted not in result.explain:
            raise _usage("--explain", f'no assignment "{wanted}" in the plan these options give',
                         "pass the fit, the options and the seed the plan was made with (its options member "
                         "lists them)", "unknown_assignment")
        report["explain"] = dict(result.explain[wanted], lines=explain_lines(result.explain[wanted]))
    else:
        readings, pairs = forms(result) if form else (None, None)
        files: list[tuple[str, Path, str]] = []
        if readings is not None:
            files.append(("form", Path(form), readings))
        if pairs is not None:
            files.append(("pairs_form", pairs_form_path(Path(form)) if readings is not None else Path(form), pairs))
        if out and any(Path(out).resolve() == path.resolve() for _, path, _ in files):
            raise _usage("--form", "the form and the plan are the same file", "give two paths")
        if out:
            written["plan"] = str(C.write_document(out, plan))
        for role, path, text in files:
            written[role] = str(C.write_text_atomic(path, text))
    report["written"] = written
    report["plan"] = plan
    if args.json:
        emit_json(REPORT_SCHEMA, report)
    else:
        budget = report["budget"]
        kinds = ", ".join(f"{n} {kind}" for kind, n in report["kinds"].items())
        print(f"eval labels next (experimental): {report['mode']} mode, {report['assignments']} assignments for "
              f"{report['reviewer']} ({kinds}), {report['minutes']:g} minutes, budget {budget['amount']:g} "
              f"{budget['unit']}")
        print(f"  candidates: {report['candidates']['assignments']} in {report['candidates']['offers']} offers")
        if report["expected_loss"] is not None:
            print(f"  value expected to be lost by picking the arm that looks best now: {report['expected_loss']:.4g}")
        if wanted is not None:
            for line in report["explain"]["lines"]:
                print("  " + line)
        for role, path in written.items():
            print(f"  wrote {role.replace('_', ' ')}: {path}")
        if not written and wanted is None:
            print("  nothing written: give --out FILE for the plan and --form FILE for a blank form")
        for line in report["notes"]:
            print(f"  note: {line}")
    for line in report["warnings"]:
        print(f"warning: {line}", file=sys.stderr)
    return EXIT_OK
