"""Edit effects: least squares selects, the screen model is reported beside it.

Least squares: y = a + sum_j b_j z_j + e, fit on every scored probe run and, with `screen.base_rows`, on each
scored base run as a row where no edit is on; z_j is 1 when edit j is on. b_j is edit j's on-versus-off effect in
score points (0 to 100). The fit is refused when it has fewer scored rows than it needs, when its columns are
linearly dependent, or when nothing is left over to estimate the error; the refusal and its reason go into the
record, and no edit is kept on a refused fit. An edit is kept when b_j is above `signal.min_t` standard errors
(0, the default, keeps any positive effect).

The head view: `loopmath.screen.model.fit_head(rows, d, SUCCESS_SCALE.scaled(100), prior="auto", shared=True)` on
the scored probe rows only, coded +1/-1, with d = 2 x (score minus their mean). On the full 12 rows of PB12 the
columns are orthogonal with spare rows, so fit_head mixes its sparse and ridge views by evidence. `fit.select =
"head"` keeps `best_subset` of its draws instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .contracts import Fit, RunResult
from .design import Design

__all__ = ["HEAD_FUNCTION", "LeastSquares", "failures_by_edit", "fit", "head", "inverse", "least_squares", "solve"]

HEAD_FUNCTION = "loopmath.screen.model.fit_head"
SCORE_SCALE = 100.0


def solve(a: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Solve a x = b by Gauss-Jordan with partial pivoting; ValueError when singular."""
    n = len(a)
    m = [list(map(float, row)) + [float(v)] for row, v in zip(a, b)]
    scale = max((abs(v) for row in m for v in row[:n]), default=1.0) or 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-10 * scale:
            raise ValueError("the design is singular for the rows that were scored")
        m[c], m[p] = m[p], m[c]
        piv = m[c][c]
        m[c] = [v / piv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c]
                m[r] = [vr - f * vc for vr, vc in zip(m[r], m[c])]
    return [m[r][n] for r in range(n)]


def inverse(a: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(a)
    cols = [solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


@dataclass
class LeastSquares:
    edits: list[str]
    intercept: float
    effects: list[float]
    se: list[float | None]
    sigma: float | None
    df: int
    n_rows: int
    residuals: list[float] = field(default_factory=list)


def least_squares(rows: Sequence[Sequence[int]], y: Sequence[float], edits: Sequence[str]) -> LeastSquares:
    """y = a + Z b by ordinary least squares, Z 0/1 (edit off/on) per row. SEs use the residual SD when there are
    residual degrees of freedom, else None. ValueError for too few rows or a singular design."""
    edits = list(edits)
    p = len(edits)
    if len(rows) != len(y):
        raise ValueError("rows and scores differ in length")
    x = [[1.0] + [float(v) for v in row] for row in rows]
    n = len(x)
    if n < p + 1:
        raise ValueError(f"{n} scored rows cannot fit {p} effects and an intercept")
    xtx = [[sum(r[a] * r[b] for r in x) for b in range(p + 1)] for a in range(p + 1)]
    xty = [sum(r[a] * v for r, v in zip(x, y)) for a in range(p + 1)]
    coef = solve(xtx, xty)
    resid = [v - sum(c * w for c, w in zip(coef, r)) for v, r in zip(y, x)]
    df = n - p - 1
    sigma = math.sqrt(sum(e * e for e in resid) / df) if df > 0 else None
    if sigma is not None:
        inv = inverse(xtx)
        se: list[float | None] = [sigma * math.sqrt(max(inv[j + 1][j + 1], 0.0)) for j in range(p)]
    else:
        se = [None] * p
    return LeastSquares(edits=edits, intercept=coef[0], effects=coef[1:], se=se, sigma=sigma, df=df, n_rows=n,
                        residuals=resid)


def head(signed_rows: Sequence[Sequence[int]], y: Sequence[float], edits: Sequence[str], *, seed: int = 0,
         draws: int = 4000, keep_draws: bool = False) -> dict:
    """The screen model's estimate per edit, or {"available": False, "reason": ...}. In process: loopmath's own
    model, no subprocess."""
    base = {"function": HEAD_FUNCTION, "prior": "auto", "shared": True}
    if not edits:
        return {**base, "available": False, "reason": "no edits"}
    if len(y) < 2:
        return {**base, "available": False, "reason": "fewer than 2 scored probe runs"}
    try:
        import numpy as np

        from ...screen import model
    except ImportError as exc:  # pragma: no cover - numpy is a core dependency
        return {**base, "available": False, "reason": f"{type(exc).__name__}: {exc}"}
    mean = sum(y) / len(y)
    X = np.array([list(r) for r in signed_rows], dtype=float)
    d = np.array([2.0 * (v - mean) for v in y], dtype=float)
    try:
        f = model.fit_head(X, d, scale=model.SUCCESS_SCALE.scaled(SCORE_SCALE), prior="auto", shared=True)
        dr = f.draws(draws, seed=seed % (2 ** 32))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {**base, "available": False, "reason": f"{type(exc).__name__}: {exc}"[:300]}
    sd = np.sqrt(np.maximum(f.var(), 0.0))
    per = {e: {"mean": float(f.mean[j]), "sd": float(sd[j]), "lo": float(np.quantile(dr[:, j], 0.1)),
               "hi": float(np.quantile(dr[:, j], 0.9)), "p_helps": float((dr[:, j] > 0).mean())}
           for j, e in enumerate(edits)}
    best = model.best_subset(dr)
    out = {**base, "available": True, "n_rows": len(y), "view": type(f).__name__,
           "scales": {k: float(v) for k, v in f.scales().items()}, "edits": per,
           "best": [list(edits)[j] for j in best["on"]]}
    if keep_draws:
        out["_draws"] = dr
    return out


def failures_by_edit(rows: Sequence[Sequence[int]], failed: Sequence[bool], edits: Sequence[str]) -> dict:
    """Runs with no deliverable per edit, on versus off: a diagnostic, never a selector."""
    out = {}
    for j, e in enumerate(edits):
        on = [f for row, f in zip(rows, failed) if row[j]]
        off = [f for row, f in zip(rows, failed) if not row[j]]
        out[e] = {"on": [sum(on), len(on)], "off": [sum(off), len(off)]}
    return out


def fit(design: Design, probes: Sequence[RunResult], base: Sequence[RunResult], *, select: str = "least-squares",
        report: Sequence[str] = ("head",), min_t: float = 0.0, base_rows: bool = True, seed: int = 0) -> Fit:
    """The effects of the design's edits from the scored runs. `probes[r]` is the run of design row r. A run with no
    score (left out, or no score after the judge's tries) is out of both fits and listed in `excluded`."""
    ids = list(design.edits)
    rows, y, excluded = [], [], []
    if base_rows:
        for s in base:
            if s.value is None:
                excluded.append(s.label)
            else:
                rows.append([0] * len(ids))
                y.append(float(s.value))
    drows, dy, srows = [], [], []
    for r, s in enumerate(probes):
        if s.value is None:
            excluded.append(s.label)
            continue
        drows.append(design.rows[r])
        dy.append(float(s.value))
        srows.append(design.signed()[r])
    out = Fit(edits=ids, select=select, excluded=excluded)
    if not ids:
        out.refused = "no edits"
        return out
    ls = None
    try:
        ls = least_squares(rows + drows, y + dy, ids)
    except ValueError as exc:
        out.refused = str(exc)
        out.n_rows = len(rows) + len(drows)
    if ls is not None:
        out.intercept, out.sigma, out.df, out.n_rows = ls.intercept, ls.sigma, ls.df, ls.n_rows
        out.effects = dict(zip(ids, ls.effects))
        out.se = dict(zip(ids, ls.se))
        out.t = {e: (b / s if s else None) for e, b, s in zip(ids, ls.effects, ls.se)}
        if ls.df < 1:
            out.refused = (f"no residual degrees of freedom ({ls.n_rows} scored rows for {len(ids)} effects and an "
                           "intercept)")
    want_head = select == "head" or "head" in report
    hv = head(srows, dy, ids, seed=seed, keep_draws=select == "head") if want_head else {}
    out.head = {k: v for k, v in hv.items() if not k.startswith("_")}
    if out.refused:
        return out
    if select == "head":
        if not hv.get("available"):
            out.refused = f"the head fit is not available: {hv.get('reason', '')}"
            return out
        out.kept = list(hv.get("best") or [])
    else:
        out.kept = [e for e in ids if out.effects[e] > 0
                    and (min_t <= 0 or (out.se[e] is not None and out.effects[e] > min_t * out.se[e]))]
    out.separated = bool(out.kept)
    return out
