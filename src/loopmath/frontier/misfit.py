"""The standing misfit test of `adaptive-blocks`: does the success model miss something the runs show?

A probe is a run whose task was drawn at equal chance from its support without looking at the posterior, scored
against the model's own prediction from before the outcome. Three families of statistics over the probes, each
turned into an e-value that is valid at every look (a mixture of likelihood ratios against logit shifts, Ville's
inequality):

  level    per setting, two-sided: are the passes more or fewer than predicted.
  group    per setting, one-sided: residuals on tasks where the same model has so far done better than predicted,
           against residuals on tasks where it has done worse. Positive when a model-by-task effect is missing.
  pooled   the group statistic over all settings, one-sided.

Epochs. The predictions change only when the posterior is refreshed, at the end of a block, so the probes of one
block share one state of the model. tick() closes the epoch. Per statistic and tilt, the factor of an epoch is the
average of the likelihood ratios of its probes; the product over closed epochs is a nonnegative martingale as long
as the rule that closes epochs cannot depend on the outcomes of the open epoch (a block always runs to its end).

The flags latch: a statistic's latch turns on when its adjusted p-value falls below alpha and off only when it rises
above `clear` (ten times alpha by default). A flagged setting gets the balanced task rule in `adaptive-blocks`.

The monitor only watches: it changes nothing the posterior reports, draws nothing, reads only the model names.
Ported from the design study's misfit component (epoch rule "mean", efforts of one model share their task
groups), arithmetic unchanged.
"""
from __future__ import annotations

import numpy as np

ALPHA = 0.05                              # the false-alarm rate aimed at, per frontier, at any number of looks
TILTS = np.geomspace(0.1, 2.5, 9)         # logit shifts the likelihood ratios are taken against
CLEAR = 10.0                              # a latch turns off when the adjusted p-value exceeds CLEAR * alpha


class Monitor:
    def __init__(self, models, T: int, alpha: float = ALPHA):
        self.S, self.T = len(models), int(T)
        self.alpha = float(alpha)
        self.clear = float(min(1.0, CLEAR * self.alpha))
        names = sorted(set(models))
        self.unit = np.array([names.index(m) for m in models], dtype=int)
        S, L = self.S, len(TILTS)
        self._lam = np.concatenate([TILTS, -TILTS])            # columns: +tilts, then -tilts
        n = 2 * S + 1                      # statistics: 0..S-1 level, S..2S-1 group, 2S pooled group
        self.logm = np.zeros((n, 2 * L))   # closed epochs: sum of log(average likelihood ratio), per tilt
        self.sum = np.zeros(n)             # closed epochs: sum of g * (y - p)
        self.var = np.zeros(n)             # closed epochs: sum of p * (1 - p)
        self.terms = np.zeros(n, dtype=int)
        self._acc = np.zeros((n, 2 * L))   # the open epoch: sum of likelihood ratios
        self._cnt = np.zeros(n, dtype=int)
        self._sum = np.zeros(n)
        self._var = np.zeros(n)
        self.mem = np.zeros((int(self.unit.max()) + 1, self.T))   # summed residuals of closed epochs
        self._pending = []                 # (unit, task, residual) of the open epoch
        self.n = 0                         # probes, open epoch included
        self._open = 0                     # probes of the open epoch
        self.on = np.zeros(n, dtype=bool)  # the latches
        self.mult = np.full(n, 3.0 * S)    # adjusted p = mult * p: three families, S statistics in the first two
        self.mult[2 * S] = 3.0
        self.epochs = 0                    # ticks that closed an epoch with a probe in it
        self.ticks = 0                     # all calls of tick()

    def _add(self, i: int, g: float, y: int, p: float) -> None:
        lg = self._lam * g
        self._acc[i] += np.exp(lg * y - np.log1p(p * np.expm1(lg)))    # the Bernoulli likelihood ratio at each tilt
        self._cnt[i] += 1
        self._sum[i] += g * (y - p)
        self._var[i] += p * (1 - p)

    def probe(self, s: int, t: int, y: int, p_pred: float) -> None:
        """One probe: setting s, task t drawn at equal chance without looking at the posterior, outcome y, and the
        model's pass chance p_pred for (s, t) from before the block began."""
        s, t, y, p = int(s), int(t), int(y), float(p_pred)
        if not 0.0 <= p <= 1.0:            # also refuses nan; no clipping, so each ratio has mean exactly 1
            raise ValueError("a prediction must be a probability in [0, 1]")
        if y not in (0, 1):
            raise ValueError("the outcome of a probe is 0 or 1")
        self._add(s, 1.0, y, p)
        g = float(np.sign(self.mem[self.unit[s], t]))          # from closed epochs only
        if g != 0.0:
            self._add(self.S + s, g, y, p)
            self._add(2 * self.S, g, y, p)
        self._pending.append((int(self.unit[s]), t, y - p))
        self._open += 1
        self.n += 1

    def tick(self) -> None:
        """The posterior has been refreshed: close the epoch and move the latches."""
        self.ticks += 1
        self.epochs += bool(self._open)
        idx = np.flatnonzero(self._cnt > 0)
        if len(idx):
            self.logm[idx] += np.log(self._acc[idx] / self._cnt[idx, None])
            self.sum[idx] += self._sum[idx]
            self.var[idx] += self._var[idx]
            self.terms[idx] += self._cnt[idx]
            self._acc[idx], self._cnt[idx], self._sum[idx], self._var[idx] = 0.0, 0, 0.0, 0.0
            adj = self._adjusted(idx)[0]
            self.on[idx] = np.where(self.on[idx], adj <= self.clear, adj < self.alpha)
        for u, t, r in self._pending:
            self.mem[u, t] += r
        self._pending, self._open = [], 0

    def _adjusted(self, idx=None):
        """(adjusted p-values, not capped at 1; z) of the statistics `idx` (all if None), over closed epochs."""
        S, L = self.S, len(TILTS)
        idx = np.arange(2 * S + 1) if idx is None else idx
        expo = self.logm[idx]
        m = expo.max(axis=1, keepdims=True)
        two = (m[:, 0] + np.log(np.exp(expo - m).mean(axis=1)))                 # both signs
        mp = expo[:, :L].max(axis=1, keepdims=True)
        one = (mp[:, 0] + np.log(np.exp(expo[:, :L] - mp).mean(axis=1)))        # positive tilts only
        loge = np.where(idx < S, two, one)
        den = np.sqrt(self.var[idx])
        z = np.divide(self.sum[idx], den, out=np.zeros(len(idx)), where=den > 0)
        return np.exp(-np.maximum(loge, 0.0)) * self.mult[idx], z               # p = min(1, 1 / e)

    def misfit(self) -> dict:
        """flag, flag_s and pooled are the latched states; p is the current smallest adjusted p-value; stat the
        sum of residuals over the root of the summed Bernoulli variances of that statistic (a description)."""
        S = self.S
        adj, z = self._adjusted()
        flag_s = self.on[:S] | self.on[S:2 * S]
        pooled = bool(self.on[2 * S])
        i = int(np.argmin(adj))
        kind = "level" if i < S else ("group" if i < 2 * S else "pooled")
        return dict(stat=float(z[i]), p=float(min(1.0, adj[i])), flag=bool(flag_s.any() or pooled), n=int(self.n),
                    flag_s=[bool(x) for x in flag_s], alpha=self.alpha, clear=self.clear, kind=kind,
                    s=(int(i % S) if i < 2 * S else None), pooled=pooled, n_open=int(self._open),
                    epochs=int(self.epochs), ticks=int(self.ticks))
