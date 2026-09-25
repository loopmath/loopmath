"""Composition: from per-piece, per-round beliefs to a run (spec 04 section 3, paper Proposition 3.24).

Everything here works on arrays over the joint parameter draws (one entry per draw):
- `c_v(k) = width_v * exp(eta + sigma^2 / 2)`: the expected cost of piece `v` in round `k`
  given the parameters (log-normal mean);
- `p(k) = s(eta)`: the gate's pass chance in round `k`;
- `r(k) = prod_{j<k} (1 - p(j))`, `k = 1 .. K_max`: the chance a repair loop reaches round `k`;
- `E[C_run] = sum over pieces outside loops of c_v(1) + sum over loops of sum_k r(k) sum_{v in loop} c_v(k)`.

Repair loops that share pieces are one loop (design.structure), and a loop may have several
gates in piece order, as in the sweep: tests after implement, then review. Within round `k`
a piece runs only if the gates before it passed, so its weight is `r(k)` times those pass
chances, and the loop goes round again unless every gate passed:
`r(k + 1) = r(k) (1 - prod_g p_g(k))`. With one gate this is the formula above.

Cost intervals are predictive: alongside the expected cost per draw, one run is
simulated per draw (run noise and gate outcomes), and the interval is its 10th and 90th
percentiles. Rounds, success and `ell` are parameter-draw quantities. Without a generator,
`compose` computes the expectations only (the mean path in state.py).

Per piece, `PieceDraws.exp_*` is the piece's expected full-run contribution, summed over
the rounds it runs, so pieces add up to the run. `cost_per_round()` gives the expected cost of
one execution: the per-execution prediction when the piece's cost does not depend on the round
(pieces that run once, or no round effects in the fit), else `E[C_piece] / E[executions]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

from .design import Structure

LEVEL = 0.8
Q_LO, Q_HI = 10.0, 90.0


@dataclass
class PieceDraws:
    exp_usd: np.ndarray
    exp_tokens: np.ndarray
    sim_usd: np.ndarray
    sim_tokens: np.ndarray
    rounds: np.ndarray  # expected number of times the piece runs
    gate_pass: np.ndarray | None = None  # first-round pass chance of the gate after this piece
    sim_runs: np.ndarray | None = None  # times the piece ran in the simulated run
    one_exp_usd: np.ndarray | None = None  # one round-1 execution: expected cost given the parameters
    one_exp_tokens: np.ndarray | None = None
    one_sim_usd: np.ndarray | None = None  # one simulated round-1 execution
    one_sim_tokens: np.ndarray | None = None
    round_free: bool = True  # the per-execution cost is the same in every round


@dataclass
class RunDraws:
    exp_usd: np.ndarray  # E[C_run | parameters], per draw
    exp_tokens: np.ndarray
    sim_usd: np.ndarray  # one simulated run per draw
    sim_tokens: np.ndarray
    rounds: np.ndarray  # expected rounds of the longest loop, per draw
    pieces: dict[str, PieceDraws] = field(default_factory=dict)
    reach: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)  # (piece, k) -> P(piece runs in round k)


def compose(st: Structure, cost_eta: dict, tokens_eta: dict, gate_eta: dict, *, sigma_usd: float,
            sigma_tokens: float, rng: np.random.Generator | None) -> RunDraws:
    """Compose one configuration. `*_eta[(piece, k)]` and `gate_eta[(gate index, k)]` are draw arrays.
    With `rng` None nothing is simulated: the `sim_*` arrays stay zero."""
    n = len(next(iter(cost_eta.values())))
    half_u, half_t = 0.5 * sigma_usd ** 2, 0.5 * sigma_tokens ** 2
    exp_u, exp_t = np.zeros(n), np.zeros(n)
    sim_u, sim_t = np.zeros(n), np.zeros(n)
    rounds = np.ones(n)
    pieces: dict[str, PieceDraws] = {}
    reach: dict[tuple[str, int], np.ndarray] = {}
    for p in st.pieces:
        pieces[p] = PieceDraws(np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n), sim_runs=np.zeros(n))
        if st.piece_loop.get(p) is not None:
            pieces[p].round_free = all(
                np.array_equal(cost_eta[(p, k)], cost_eta[(p, 1)]) and np.array_equal(tokens_eta[(p, k)], tokens_eta[(p, 1)])
                for k in range(2, st.k_max + 1))
    gate_after = {g.after: gi for gi, g in enumerate(st.gates)}
    for piece, gi in gate_after.items():
        if piece in pieces and (gi, 1) in gate_eta:
            pieces[piece].gate_pass = expit(gate_eta[(gi, 1)])

    def run_piece(piece: str, k: int, weight: np.ndarray, alive: np.ndarray):
        w = st.widths[piece]
        eu, et = cost_eta[(piece, k)], tokens_eta[(piece, k)]
        cu = w * np.exp(eu + half_u)
        ct = w * np.exp(et + half_t)
        pd = pieces[piece]
        pd.exp_usd += weight * cu
        pd.exp_tokens += weight * ct
        su = np.zeros(n)
        stt = np.zeros(n)
        for _ in range(w if rng is not None else 0):
            su += np.exp(eu + sigma_usd * rng.standard_normal(n))
            stt += np.exp(et + sigma_tokens * rng.standard_normal(n))
        pd.sim_usd += alive * su
        pd.sim_tokens += alive * stt
        pd.sim_runs += alive
        if k == 1:
            pd.one_exp_usd, pd.one_exp_tokens, pd.one_sim_usd, pd.one_sim_tokens = cu, ct, su, stt
        return weight * cu, weight * ct, alive * su, alive * stt

    for piece in st.pieces:
        if st.piece_loop.get(piece) is not None:
            continue
        one, alive = np.ones(n), np.ones(n)
        reach[(piece, 1)] = one
        a, b, c, d = run_piece(piece, 1, one, alive)
        exp_u += a
        exp_t += b
        sim_u += c
        sim_t += d
    for gidx, members in st.loops:
        after: dict[str, list[int]] = {}
        for gi in gidx:
            after.setdefault(st.gates[gi].after, []).append(gi)
        r = np.ones(n)
        alive = np.ones(n)
        loop_rounds = np.zeros(n)
        runs = {piece: np.zeros(n) for piece in members}
        for k in range(1, st.k_max + 1):
            loop_rounds += r
            m, live = r, alive
            for piece in members:
                reach[(piece, k)] = m
                runs[piece] += m
                a, b, c, d = run_piece(piece, k, m, live)
                exp_u += a
                exp_t += b
                sim_u += c
                sim_t += d
                for gi in after.get(piece, ()):
                    p = expit(gate_eta[(gi, k)])
                    m = m * p
                    if rng is not None:
                        live = live * (rng.random(n) < p)
            alive = alive - live  # runs that passed every gate this round are done
            r = r - m
        for piece in members:
            pieces[piece].rounds = runs[piece]
        rounds = np.maximum(rounds, loop_rounds)
    return RunDraws(exp_u, exp_t, sim_u, sim_t, rounds, pieces, reach)


def cost_per_round(pd: PieceDraws):
    """The expected cost of one execution of a piece, as ((usd mean, usd values), (tokens mean,
    tokens values)); None when the piece is expected never to run. The values give the interval:
    one simulated execution per draw when the cost does not depend on the round, else the
    simulated run's cost per execution (draws where the piece did not run are NaN)."""
    runs = float(np.mean(pd.rounds))
    if runs <= 0:
        return None
    if pd.round_free:
        return ((float(np.mean(pd.one_exp_usd)), pd.one_sim_usd),
                (float(np.mean(pd.one_exp_tokens)), pd.one_sim_tokens))
    ran = pd.sim_runs > 0
    if not ran.any():  # never ran in any simulated run: the per-draw expectation gives the range
        per = np.maximum(pd.rounds, 1e-300)
        return ((float(np.mean(pd.exp_usd)) / runs, pd.exp_usd / per),
                (float(np.mean(pd.exp_tokens)) / runs, pd.exp_tokens / per))
    count = np.where(ran, pd.sim_runs, 1.0)
    return ((float(np.mean(pd.exp_usd)) / runs, np.where(ran, pd.sim_usd / count, np.nan)),
            (float(np.mean(pd.exp_tokens)) / runs, np.where(ran, pd.sim_tokens / count, np.nan)))


def interval(values: np.ndarray, mean: float | None = None) -> tuple[float, float, float]:
    """(mean, 10th, 90th percentile)."""
    v = np.asarray(values, float)
    lo, hi = np.percentile(v, [Q_LO, Q_HI])
    return (float(np.mean(v)) if mean is None else float(mean)), float(lo), float(hi)


def predictive(expected: np.ndarray, simulated: np.ndarray) -> tuple[float, float, float]:
    """The mean is E[C_run]; the range is the 10th and 90th percentiles of one simulated run per draw."""
    lo, hi = np.percentile(simulated, [Q_LO, Q_HI])
    return float(np.mean(expected)), float(lo), float(hi)


class Intervals:
    """Collects draw arrays and takes all their percentiles in one call (the hot path of predict_many)."""

    def __init__(self) -> None:
        self._means: list[float] = []
        self._arrays: list[np.ndarray] = []

    def add(self, values: np.ndarray, mean: float | None = None) -> int:
        """`interval(values)`, resolved later; returns a handle."""
        self._arrays.append(values)
        self._means.append(float(np.mean(values)) if mean is None else float(mean))
        return len(self._arrays) - 1

    def add_predictive(self, expected: np.ndarray, simulated: np.ndarray) -> int:
        return self.add(simulated, float(np.mean(expected)))

    def resolve(self) -> list[tuple[float, float, float]]:
        if not self._arrays:
            return []
        stacked = np.stack(self._arrays)
        q = np.percentile(stacked, [Q_LO, Q_HI], axis=1)
        gaps = np.isnan(stacked).any(axis=1)  # cost per round: draws where the piece did not run
        if gaps.any():
            q[:, gaps] = np.nanpercentile(stacked[gaps], [Q_LO, Q_HI], axis=1)
        return [(m, float(lo), float(hi)) for m, lo, hi in zip(self._means, q[0], q[1])]
