"""The fold-over design: one +-1 row per pair, arm A runs the edits at +1 and arm B (the mirror) the edits at -1.

A pair's difference (A minus B) is then the sum of the edits' on-versus-off effects, signed by the row, plus
noise: the task's level cancels, and so does every two-edit interaction, since A and B flip every edit at once.
Columns are balanced (each edit is on in A for half the pairs). For 2, 4, 8, 16, 32 or 64 pairs (Sylvester
Hadamard) and 12, 20 or 24 pairs (Plackett-Burman), and stacks of these, the columns are also orthogonal, so no
edit's estimate leans on another's; that needs at least one pair more than there are edits. Other counts get
balanced columns of low correlation, found by random search.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Plackett-Burman generators: the rows are the cyclic shifts of the generator and a row of all -1.
_PB = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}


def _sylvester(n: int) -> np.ndarray:
    h = np.ones((1, 1), dtype=int)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def _plackett_burman(n: int) -> np.ndarray:
    g = np.array([1 if c == "+" else -1 for c in _PB[n]], dtype=int)
    rows = [np.roll(g, k) for k in range(n - 1)] + [-np.ones(n - 1, dtype=int)]
    return np.array(rows, dtype=int)


def _orthogonal(n: int) -> np.ndarray | None:
    """n rows of n - 1 balanced, orthogonal +-1 columns, or None when no built-in design has n rows."""
    if n in _PB:
        return _plackett_burman(n)
    if n >= 2 and n & (n - 1) == 0 and n <= 64:
        return _sylvester(n)[:, 1:]  # the first column is all +1: it is the pair's mean, not an edit
    return None


SIZES = (2, 4, 8, 12, 16, 20, 24, 32, 64)


def plan(p: int, m: int) -> dict:
    """How `fold_over` builds p columns over m pairs: {"kind", "block", "orthogonal"}."""
    for s in sorted(SIZES, reverse=True):
        if s >= p + 1 and m % s == 0:
            kind = "plackett-burman" if s in _PB else "hadamard"
            return {"kind": f"{kind}-{s}" + (f" x{m // s}" if m > s else ""), "block": s, "orthogonal": True}
    return {"kind": "balanced-random", "block": None, "orthogonal": False}


def fold_over(p: int, m: int, seed: int = 0) -> np.ndarray:
    """m rows (pairs) by p columns (edits) of +-1, balanced, and orthogonal where `plan` says so. Rows are
    shuffled and columns flipped at random, so which edits share a pair with which is not fixed."""
    if p < 1:
        raise ValueError("a screen needs at least one edit")
    if m < 2 or m % 2:
        raise ValueError(f"{m} pairs: a balanced design needs an even number of pairs, at least 2")
    rng = np.random.default_rng(seed)
    how = plan(p, m)
    if how["orthogonal"]:
        s = how["block"]
        blocks = []
        for _ in range(m // s):
            d = _orthogonal(s)
            cols = rng.choice(d.shape[1], size=p, replace=False)
            blocks.append(d[:, cols] * rng.choice([-1, 1], size=p))
        rows = np.vstack(blocks)
        return rows[rng.permutation(m)]
    return _balanced_random(p, m, rng)


def _balanced_random(p: int, m: int, rng: np.random.Generator, tries: int = 400) -> np.ndarray:
    """Balanced +-1 columns with the smallest largest correlation found in `tries` random draws."""
    base = np.array([1] * (m // 2) + [-1] * (m // 2))
    best, best_score = None, np.inf
    for _ in range(tries):
        rows = np.column_stack([rng.permutation(base) for _ in range(p)])
        if p == 1:
            return rows
        g = np.abs(rows.T @ rows - m * np.eye(p, dtype=int))
        score = g.max() * 1000 + (g ** 2).sum()
        if score < best_score:
            best, best_score = rows, score
    return best


def arms(row: Sequence[int], names: Sequence[str]) -> tuple[list[str], list[str]]:
    """(the edits on in arm A, the edits on in arm B) for one row."""
    return [n for x, n in zip(row, names) if x > 0], [n for x, n in zip(row, names) if x < 0]
