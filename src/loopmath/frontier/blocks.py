"""Strategy `adaptive-blocks`: buy the run or block of runs that adds the most value per planned dollar.

The value (see `value.py`) sums, over a grid of budgets per run, the mean pass rate of the setting a user would pick
at that budget. A candidate is one run, or a block of runs of one setting; its gain is the expected rise of the value
once its outcomes are in, read from the success model's lookahead. One decision:

  1. Open pairs. A pair (setting, task) is open while it has fewer than `pair_cap` runs (default 4) and, in study
     mode, while the pool holds runs of it. The cap keeps the strategy from rerunning one pair, whose copies the
     success model would count as new evidence.
  2. Prices. Which budgets a setting can serve is judged by an optimistic price, the `q0` quantile (default 0.2) of
     the belief about its mean cost per run; the planned cost of a run is the posterior median of that belief. The
     report keeps the posterior mean.
  3. No penalty. The value that chooses runs has no term for the uncertainty of the pick (kappa = 0).
  4. Candidates, by the task rule `tasks`:
       chosen    (default) every open pair as a single run, scored by the exact one-run gain; and per setting,
                 blocks of 8 and 64 runs on its open tasks ranked by the fall of its own variance per planned
                 dollar (second runs only when fewer tasks are open), scored by the closed-form block gain.
       balanced  per setting, blocks of 1, 8 and 64 runs on its least-run open tasks, and one run on every open
                 task (a round), each scored on that fixed allocation by the closed-form block gain.
  5. Choice: among candidates with a positive gain, the largest gain per planned dollar; ties go to the lower
     planned cost, then the lower setting, then the shorter block, then the lower task (chosen) or the order
     1, 8, 64, round (balanced). With no positive gain, the largest gain.
  6. Tasks that run. chosen: a block runs its ranked tasks; a single run keeps its task except with chance
     `explore` (default 0.1), when the task is drawn at equal chance from the setting's open tasks (a probe for the
     misfit test). A setting the misfit test flags gets the balanced rule for its single runs. balanced: the block
     is drawn again with ties among least-run tasks broken at random, one generator per setting; every run is a
     probe.

Ported from the design study behind this strategy, with its three fixes (blocks, no penalty, optimistic price) and
the pair cap, arithmetic unchanged.
"""
from __future__ import annotations

import numpy as np

from .value import block_gains, gain_pairs

TASK_RULES = ("chosen", "balanced")
UNLIMITED = np.iinfo(np.int32).max        # runs of a pair with no pool limit
CHOSEN_BLOCKS = (8, 64)                   # with the single runs: k = 1, 8, 64
BALANCED_BLOCKS = (1, 8, 64)              # plus the round
RUN_BUDGETS = (0.0145, 26.5, 36)          # USD per run, log-spaced, rounded to 6 decimals; equal weights
POLICY_DEFAULTS = dict(tasks="chosen", pair_cap=4, q0=0.2, explore=0.1, refit_every=20)


def budget_grid(lo: float, hi: float, n: int) -> list[float]:
    return np.round(np.geomspace(lo, hi, n), 6).tolist()


def balanced_tasks(s, L, n, remaining, g=None):
    """L runs of setting s, each on one of the tasks with runs remaining and the fewest runs so far. With g=None the
    tie goes to the lowest task (the allocation a candidate is scored with), else it is drawn with g. Returns
    (task, p_task, support) triples, p_task = 1 / support, support the size of the tie; shorter than L if the
    runs remaining give out."""
    count = n[s].astype(float).copy()
    left = np.minimum(remaining[s], L).astype(int)
    out = []
    for _ in range(L):
        cap = left > 0
        if not cap.any():
            break
        low = np.flatnonzero(cap & (count == count[cap].min()))
        t = int(low[0] if g is None else g.choice(low))
        out.append((t, 1.0 / len(low), len(low)))
        count[t] += 1
        left[t] -= 1
    return out


def round_tasks(s, remaining, g=None):
    """One run of setting s on every task with runs remaining. With g, in random order and each run logged with the
    chance of its draw given the runs before it (support m - i at position i of m); with g=None in task order."""
    open_ = np.flatnonzero(remaining[s] > 0)
    if g is None:
        return [(int(t), 1.0, 1) for t in open_]
    m = len(open_)
    return [(int(t), 1.0 / (m - i), m - i) for i, t in enumerate(g.permutation(open_))]


def choose_flat(gain, cost, keys):
    """(index, branch) over flat candidate arrays; keys are the tie keys after the planned cost."""
    priced = np.isfinite(cost) & (cost > 0)
    pos = priced & (gain > 0.0)
    if pos.any():
        value, branch = np.where(pos, gain / np.where(pos, cost, 1.0), -np.inf), "positive"
    else:
        value, branch = np.asarray(gain, float).copy(), "nonpositive"
    idx = np.flatnonzero(value == value.max())
    c = np.where(priced[idx], cost[idx], np.inf)
    return int(idx[np.lexsort(tuple(k[idx] for k in reversed(keys)) + (c,))[0]]), branch


def choose_ranked(gain, cost, s, length, rank):
    """(index, branch) for the balanced rule: the same choice, ties by cost, setting, length, kind."""
    pos = gain > 0
    if pos.any():
        idx = np.flatnonzero(pos)
        key = gain[idx] / cost[idx]
        branch = "positive"
    else:
        idx = np.arange(len(gain))
        key = gain
        branch = "nonpositive"
    order = np.lexsort((rank[idx], length[idx], s[idx], cost[idx], -key))
    return int(idx[order[0]]), branch


class AdaptiveBlocks:
    """One decision at a time. `decide(belief, remaining, streams)` returns the next block as a list of dicts
    (s, t, p_task, explore, support, why), empty when no pair is open. `streams` holds the generators:
    "explore" (chosen rule) and "tasks", a function of the setting (balanced rule)."""

    def __init__(self, B, w, *, tasks="chosen", pair_cap=4, q0=0.2, explore=0.1, kappa=0.0):
        if tasks not in TASK_RULES:
            raise ValueError("tasks is chosen or balanced")
        if isinstance(pair_cap, bool) or not isinstance(pair_cap, int) or pair_cap < 1:
            raise ValueError("pair_cap is a positive integer")
        if not 0.0 < float(q0) < 0.5 or not 0.0 <= float(explore) <= 1.0:
            raise ValueError("q0 lies in (0, 0.5) and explore in [0, 1]")
        self.B, self.w = np.asarray(B, float), np.asarray(w, float)
        self.tasks, self.pair_cap = tasks, int(pair_cap)
        self.q0, self.e, self.kappa = float(q0), float(explore), float(kappa)

    # ---- shared parts

    def open_runs(self, belief, remaining):
        """(S, T) int: the runs each pair can still take under the pair cap."""
        return np.minimum(np.asarray(remaining), np.maximum(self.pair_cap - np.asarray(belief.n), 0))

    def prices(self, belief):
        """(c_elig (S,), cost (S, T)): the price that decides eligibility, and the planned cost of each pair."""
        c_hat = np.asarray(belief.c(), float)
        c_pair = np.asarray(belief.c_pair(), float)
        lo, hi = belief.c_interval(1.0 - 2.0 * self.q0)
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        ok = np.isfinite(lo) & (lo > 0) & np.isfinite(hi) & (hi > 0) & np.isfinite(c_hat) & (c_hat > 0)
        med = np.where(ok, np.sqrt(np.where(ok, lo * hi, 1.0)), c_hat)
        c_elig = np.where(ok, lo, c_hat)
        scale = np.where(ok, med / np.where(ok, c_hat, 1.0), 1.0)
        return c_elig, c_pair * scale[:, None]

    def decide(self, belief, remaining, streams):
        remaining = self.open_runs(belief, remaining)
        if not (remaining > 0).any():
            return []
        if self.tasks == "balanced":
            return self._balanced(belief, remaining, streams["tasks"])
        return self._chosen(belief, remaining, streams["explore"])

    # ---- tasks chosen

    def flags(self, belief):
        m = belief.misfit()
        fs = np.asarray(m.get("flag_s", []), bool)
        return (fs if fs.shape == (belief.S,) else np.zeros(belief.S, bool)), bool(m.get("flag"))

    def _pair_table(self, belief, cand, c_elig):
        mu, Sigma = belief.normal()
        mu, var = np.asarray(mu, float), np.diag(np.asarray(Sigma, float)).copy()
        out, own = np.full(belief.n.shape, np.nan), np.full(belief.n.shape, np.nan)
        rows = [s for s in range(len(cand)) if len(cand[s])]
        if rows:
            looks = [belief.lookahead_pairs(s, cand[s]) for s in rows]
            sh = np.concatenate([np.asarray(x[0], float) for x in looks])
            va = np.concatenate([np.asarray(x[1], float) for x in looks])
            g = gain_pairs(mu, var, sh, va, c_elig, self.B, self.w, self.kappa)
            i = 0
            for s in rows:
                out[s, cand[s]] = g[i:i + len(cand[s])]
                own[s, cand[s]] = sh[i:i + len(cand[s]), s] ** 2
                i += len(cand[s])
        return out, own, mu, var

    @staticmethod
    def _fill(ranked, rem_s, L):
        """The first L tasks of `ranked`, one run each; second runs only if fewer than L tasks are open."""
        out, k = [], 1
        while len(out) < L:
            add = [int(t) for t in ranked if rem_s[t] >= k]
            if not add:
                break
            out.extend(add[:L - len(out)])
            k += 1
        return np.array(out, dtype=int)

    def _block_table(self, belief, remaining, flagged, own, cost, c_elig, mu, var):
        rows = []
        for s in np.flatnonzero((remaining > 0).any(1)):
            s = int(s)
            avail = np.flatnonzero(remaining[s] > 0)
            if flagged[s]:
                make = lambda L, s=s: np.array([t for t, _, _ in balanced_tasks(s, L, belief.n, remaining)], dtype=int)
            else:
                ok = np.isfinite(cost[s, avail]) & (cost[s, avail] > 0) & np.isfinite(own[s, avail])
                per = np.where(ok, own[s, avail] / np.where(ok, cost[s, avail], 1.0), -np.inf)
                ranked = avail[np.argsort(-per, kind="stable")]
                make = lambda L, ranked=ranked, s=s: self._fill(ranked, remaining[s], L)
            seen = set()
            for L in CHOSEN_BLOCKS:
                tasks = make(L)
                key = tuple(sorted(tasks.tolist()))
                if len(tasks) < 2 or key in seen:
                    continue
                seen.add(key)
                sh, va = belief.lookahead(s, np.sort(tasks))
                rows.append((s, tasks, np.asarray(sh, float), np.asarray(va, float), float(cost[s, tasks].sum())))
        if not rows:
            return []
        g = block_gains(mu, var, np.stack([r[2] for r in rows]), np.stack([r[3] for r in rows]), c_elig,
                        self.B, self.w, self.kappa)
        return [(r[0], r[1], float(gi), r[4]) for r, gi in zip(rows, g)]

    def _chosen(self, belief, remaining, g_explore):
        open_ = remaining > 0
        flagged, flag = self.flags(belief)
        cand = []
        for s in range(belief.S):
            if not open_[s].any():
                cand.append(np.zeros(0, dtype=int))
            elif flagged[s]:
                cand.append(np.array([balanced_tasks(s, 1, belief.n, remaining)[0][0]]))
            else:
                cand.append(np.flatnonzero(open_[s]))
        c_elig, cost = self.prices(belief)
        gains, own, mu, var = self._pair_table(belief, cand, c_elig)
        ps, pt = np.nonzero(np.isfinite(gains))
        blk = self._block_table(belief, remaining, flagged, own, cost, c_elig, mu, var)
        cs = np.concatenate([ps, np.array([b[0] for b in blk], dtype=int)])
        ct = np.concatenate([pt, np.array([b[1][0] for b in blk], dtype=int)])
        cL = np.concatenate([np.ones(len(ps), dtype=int), np.array([len(b[1]) for b in blk], dtype=int)])
        cg = np.concatenate([gains[ps, pt], np.array([b[2] for b in blk])])
        cc = np.concatenate([cost[ps, pt], np.array([b[3] for b in blk])])
        i, branch = choose_flat(cg, cc, (cs, cL, ct))
        s, t, g, c = int(cs[i]), int(ct[i]), float(cg[i]), float(cc[i])
        score = float(g / c) if np.isfinite(c) and c > 0 else None
        why = dict(rule="flag" if flagged[s] else "single", branch=branch, t_rule=t, gain=g, score=score, flag=flag)
        if i >= len(ps):
            tasks = blk[i - len(ps)][1]
            why.update(rule="block", L=int(len(tasks)))
            return [dict(s=s, t=int(u), p_task=1.0, explore=False, support=0, why=why) for u in tasks]
        if flagged[s]:
            (u, p_task, support), = balanced_tasks(s, 1, belief.n, remaining, g_explore)
            return [dict(s=s, t=int(u), p_task=float(p_task), explore=True, support=int(support), why=why)]
        avail = np.flatnonzero(remaining[s] > 0)
        m = len(avail)
        explore = bool(g_explore.random() < self.e)
        u = int(avail[g_explore.integers(m)]) if explore else t
        p_task = float((1.0 - self.e) * (u == t) + self.e / m)
        return [dict(s=s, t=u, p_task=p_task, explore=explore, support=m if explore else 0, why=why)]

    # ---- tasks balanced

    def _balanced(self, belief, remaining, g_tasks):
        cands = []
        for s in np.flatnonzero((remaining > 0).any(1)):
            s = int(s)
            lengths = set()
            for L in BALANCED_BLOCKS:
                plan = balanced_tasks(s, L, belief.n, remaining, None)
                if not plan or len(plan) in lengths:   # cut by the runs remaining to a shorter block's runs and law
                    continue
                lengths.add(len(plan))
                cands.append((s, L, np.sort(np.array([t for t, _, _ in plan], dtype=int))))
            plan = round_tasks(s, remaining, None)
            if plan:
                cands.append((s, "round", np.sort(np.array([t for t, _, _ in plan], dtype=int))))
        mu, Sig = belief.normal()
        var = np.diag(Sig)
        c_elig, c_pair = self.prices(belief)
        look, seen = [], {}
        for s, _, tasks in cands:                      # the same runs of a setting are looked ahead once
            key = (s, tasks.tobytes())
            if key not in seen:
                seen[key] = belief.lookahead(s, tasks)
            look.append(seen[key])
        gain = block_gains(mu, var, np.stack([x[0] for x in look]), np.stack([x[1] for x in look]), c_elig,
                           self.B, self.w, self.kappa)
        cost = np.array([c_pair[s, tasks].sum() for s, _, tasks in cands])
        rank = np.array([len(BALANCED_BLOCKS) if k == "round" else BALANCED_BLOCKS.index(k) for _, k, _ in cands])
        i, branch = choose_ranked(gain, cost, np.array([c[0] for c in cands]),
                                  np.array([len(c[2]) for c in cands]), rank)
        s, kind, _ = cands[i]
        g = g_tasks(s)
        plan = round_tasks(s, remaining, g) if kind == "round" else balanced_tasks(s, kind, belief.n, remaining, g)
        why = dict(rule="balanced", block=kind, branch=branch, gain=float(gain[i]), cost=float(cost[i]))
        return [dict(s=s, t=int(t), p_task=float(p), explore=True, support=int(m), why=why) for t, p, m in plan]
