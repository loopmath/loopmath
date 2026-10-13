"""Block Cholesky of a posterior precision whose leaf block is diagonal.

Nodes split into leaves `L`, no two of which share a row or a prior factor (every task node is
one: a row carries exactly one task), and the core `C`. In the order (L, C) the precision is

    P = [[diag(d), B'], [B, P_CC]],   B = P_CL (c x l, sparse),

and its lower Cholesky factor is [[D^1/2, 0], [B D^-1/2, R]] with R R' = S = P_CC - B D^-1 B'.
Only the Schur complement `S` (c x c) is dense, so the work is O(c^3) instead of O(p^3), and the
factor is exactly the dense Cholesky factor of P in that order.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse


def leaf_split(pattern: sparse.spmatrix) -> tuple[np.ndarray, int]:
    """(perm, l): an independent set of the co-occurrence graph first (greedy, lowest degree
    first, ties by index), then the other nodes, each part in its original order."""
    G = sparse.csr_matrix(pattern, dtype=bool)
    G = (G + G.T).tocsr()
    G.setdiag(False)
    G.eliminate_zeros()
    p = G.shape[0]
    deg = np.diff(G.indptr)
    blocked = np.zeros(p, bool)
    leaf = np.zeros(p, bool)
    indptr, indices = G.indptr, G.indices
    for j in np.lexsort((np.arange(p), deg)):
        if blocked[j]:
            continue
        leaf[j] = True
        blocked[indices[indptr[j]:indptr[j + 1]]] = True
    return np.concatenate([np.flatnonzero(leaf), np.flatnonzero(~leaf)]), int(leaf.sum())


def leading_diagonal(P: sparse.csr_matrix) -> int:
    """The largest k such that P[:k, :k] is diagonal."""
    P = sparse.csr_matrix(P)
    p = P.shape[0]
    if p == 0:
        return 0
    rows = np.repeat(np.arange(p), np.diff(P.indptr))
    off = (P.indices != rows) & (P.data != 0)
    first = np.full(p, p)
    np.minimum.at(first, rows[off], P.indices[off])
    ok = np.minimum.accumulate(first) >= np.arange(1, p + 1)
    return int(np.argmin(ok)) if not ok.all() else p


def _chol(S: np.ndarray) -> np.ndarray:
    if len(S) == 0:
        return np.zeros((0, 0))
    return np.tril(linalg.cholesky(S, lower=True, check_finite=False))


class BlockFactor:
    """The factor of P = [[diag(d), B'], [B, P_CC]] (leaves first); vectors in that order."""

    def __init__(self, d: np.ndarray, B: sparse.spmatrix, PCC: np.ndarray, *, jitter: float = 0.0):
        d = np.asarray(d, float) + jitter
        if not np.all(d > 0):
            raise linalg.LinAlgError("leaf block is not positive definite")
        self.d = d
        self.l = len(d)
        self.c = len(PCC)
        self.B = sparse.csr_matrix(B, shape=(self.c, self.l))
        self.sd = np.sqrt(d)
        self.Bs = self.B.multiply(1.0 / self.sd[None, :]).tocsr()  # B D^-1/2
        self.Bd = self.B.multiply(1.0 / d[None, :]).tocsr()  # B D^-1
        S = PCC + jitter * np.eye(self.c) if jitter else PCC.copy()
        if self.l and self.c:
            S -= (self.Bs @ self.Bs.T).toarray()
        self.R = _chol(S)
        self._Rinv: np.ndarray | None = None

    @classmethod
    def factor(cls, d, B, PCC, strict: bool = False) -> "BlockFactor":
        """With `gaussian._cho`'s jitter when P is not numerically positive definite (unless strict)."""
        if strict:
            return cls(d, B, PCC)
        jitter = 0.0
        scale = float(np.mean(np.concatenate([np.asarray(d, float), np.diag(PCC)]))) if len(d) + len(PCC) else 1.0
        for _ in range(6):
            try:
                return cls(d, B, PCC, jitter=jitter)
            except linalg.LinAlgError:
                jitter = max(1e-10, jitter * 10 or 1e-10 * scale)
        raise linalg.LinAlgError("posterior precision is not positive definite")

    @property
    def logdet(self) -> float:
        return float(np.sum(np.log(self.d))) + 2.0 * float(np.sum(np.log(np.diag(self.R))))

    @property
    def Rinv(self) -> np.ndarray:
        if self._Rinv is None:
            if self.c == 0:
                self._Rinv = np.zeros((0, 0))
            else:
                Ri, info = linalg.lapack.dtrtri(self.R, lower=1)
                if info != 0:
                    Ri = linalg.solve_triangular(self.R, np.eye(self.c), lower=True)
                self._Rinv = np.tril(Ri)
        return self._Rinv

    def solve(self, b: np.ndarray) -> np.ndarray:
        """P^-1 b, b (p,) or (p, k)."""
        b = np.asarray(b, float)
        dl = self.d if b.ndim == 1 else self.d[:, None]
        bL, bC = b[:self.l], b[self.l:]
        t = bC - self.Bd @ bL if self.l else bC
        xC = linalg.cho_solve((self.R, True), t, check_finite=False) if self.c else t
        xL = (bL - self.B.T @ xC) / dl if self.c else bL / dl
        return np.concatenate([xL, xC])

    def diag_inv(self) -> np.ndarray:
        """diag(P^-1)."""
        Rinv = self.Rinv
        diag_c = np.sum(Rinv * Rinv, axis=0)
        diag_l = 1.0 / self.d
        if self.l and self.c:
            Y = np.asarray(self.Bd.T @ Rinv.T)  # (l, c): rows (R^-1 B D^-1 e_j)'
            diag_l = diag_l + np.sum(Y * Y, axis=1)
        return np.concatenate([diag_l, diag_c])

    def half(self, M: sparse.spmatrix) -> tuple[sparse.csr_matrix, np.ndarray]:
        """M R^-T in two parts, (M_L D^-1/2 sparse, (M_C - M_L D^-1 B') R_C^-T dense): the rows'
        covariance is the sum of the two parts' inner products."""
        M = sparse.csr_matrix(M)
        ML, MC = M[:, :self.l], M[:, self.l:]
        first = ML.multiply(1.0 / self.sd[None, :]).tocsr()
        Z = MC - ML @ self.Bd.T if self.l else MC
        second = np.asarray(Z @ self.Rinv.T) if self.c else np.zeros((M.shape[0], 0))
        return first, second

    def quad(self, M: sparse.spmatrix) -> np.ndarray:
        """diag(M P^-1 M') for the rows of a sparse M (k, p)."""
        first, second = self.half(M)
        return np.asarray(first.multiply(first).sum(axis=1)).ravel() + np.sum(second * second, axis=1)

    def solve_lt(self, z: np.ndarray) -> np.ndarray:
        """R_full^-T z: `mean + solve_lt(z)` are draws, as `solve_triangular(chol, z, trans="T")`."""
        zL, zC = z[:self.l], z[self.l:]
        xC = linalg.solve_triangular(self.R, zC, trans="T", lower=True, check_finite=False) if self.c else zC
        sd = self.sd if z.ndim == 1 else self.sd[:, None]
        xL = (zL - self.Bs.T @ xC) / sd if self.c else zL / sd
        return np.concatenate([xL, xC])

    def cov_factor(self) -> np.ndarray:
        """Dense U = R_full^-T (upper triangular, U U' = P^-1), built block by block."""
        p = self.l + self.c
        U = np.zeros((p, p))
        U[np.arange(self.l), np.arange(self.l)] = 1.0 / self.sd
        if self.c:
            RiT = self.Rinv.T
            U[self.l:, self.l:] = RiT
            if self.l:
                U[:self.l, self.l:] = -np.asarray(self.Bd.T @ RiT)
        return U


def blocks(P: sparse.spmatrix, l: int) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray]:
    """(d, B, P_CC) of a sparse P whose first l nodes are leaves."""
    P = sparse.csr_matrix(P)
    d = P.diagonal()[:l]
    return d, P[l:, :l].tocsr(), P[l:, l:].toarray()
