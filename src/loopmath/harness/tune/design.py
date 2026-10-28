"""The probe design: which edits are on in which probe run.

pb12 (the default): 12 single runs over up to 10 edits, from the Plackett-Burman 12-run array
(`loopmath.screen.design._plackett_burman(12)`: the 11 cyclic shifts of `++-+++---+-` and a row of all -1). Every
column is +1 in 6 rows and every pair of columns agrees in 6, so the columns are balanced and orthogonal: each edit's
least-squares effect is a difference of two means of 6 runs, variance sigma^2 / 3, and no edit's estimate leans on
another's. Two-edit interactions are not separated: each leaks into every other edit's main effect with weight
+-1/3 (`alias_summary`).

m edits take columns 1 to m. A column's sign only says which level counts as on, so the flips are chosen to switch
on as close to m / 2 edits per run as the array allows (ties to the fewest all-off and all-on runs, then the first
pattern), which from 5 edits on leaves no all-off run (the base runs measure that harness) and no run stacking every
edit. Which edit takes which column, and the run order, follow the seed. Fewer than 10 edits keep all 12 runs: the
same precision per edit and more residual degrees of freedom.

fold-over: `loopmath.screen.design.fold_over` rows, each a pair of runs (the edits at +1 on, then the mirror).
one-at-a-time: one run per edit, that edit alone on. full: every subset, up to 4 edits. Each design's rows repeat
`screen.repeats` times.

The fit adds the base runs as all-off rows. That breaks exact orthogonality a little: with 10 edits and 2 base rows
an interaction then leaks up to 0.47 into another edit's effect and 0.14 into its own pair's, and each edit's
standard error falls from 0.577 to 0.560 residual SDs; `alias_summary` reports both views for the design at hand.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field

from .contracts import _Record

__all__ = ["DESIGNS", "MAX_EDITS", "PB12_GENERATOR", "Design", "alias_summary", "pb12", "plan_design",
           "probe_count", "residual_df", "seed_of"]

PB12_GENERATOR = "++-+++---+-"
PB12_RUNS = 12
MAX_EDITS = {"pb12": 10, "full": 4, "one-at-a-time": 31, "fold-over": 31}
DESIGNS = tuple(MAX_EDITS)


def pb12() -> list[list[int]]:
    """The 12 x 11 Plackett-Burman array of +1/-1, rows in the textbook order (loopmath.screen's own array)."""
    from ...screen.design import _plackett_burman

    return [[int(v) for v in row] for row in _plackett_burman(PB12_RUNS)]


def seed_of(*parts: object) -> int:
    """A stable integer seed from any parts (the task id, `screen.seed`, the round)."""
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha1(text.encode()).hexdigest()[:12], 16)


@dataclass
class Design(_Record):
    """Which edits are on in which probe run: `rows[r][j]` is 1 when edit j is on in run r, in run order. `repeat`
    and `pair` give each row's repeat (from 1) and fold-over pair (from 1, else None); `row_of` its distinct design
    row (from 0), the same for every repeat."""

    kind: str
    edits: list[str]
    rows: list[list[int]]
    columns: list[int] = field(default_factory=list)
    flips: list[int] = field(default_factory=list)
    seed: int = 0
    repeat: list[int] = field(default_factory=list)
    pair: list = field(default_factory=list)
    row_of: list[int] = field(default_factory=list)

    @property
    def m(self) -> int:
        return len(self.edits)

    @property
    def n(self) -> int:
        return len(self.rows)

    def on(self, r: int) -> list[str]:
        return [e for e, z in zip(self.edits, self.rows[r]) if z]

    def signed(self) -> list[list[int]]:
        """The rows coded +1 (on) / -1 (off), as the screen model takes them."""
        return [[1 if z else -1 for z in row] for row in self.rows]

    def properties(self) -> dict:
        """Balance, orthogonality and run sizes, computed from the rows, not asserted."""
        if not self.m:
            return {"runs": 0, "edits": 0}
        x = self.signed()
        on_counts = [sum(row[j] for row in self.rows) for j in range(self.m)]
        cross = [sum(x[r][a] * x[r][b] for r in range(self.n)) for a, b in itertools.combinations(range(self.m), 2)]
        sizes = [sum(row) for row in self.rows]
        return {"runs": self.n, "edits": self.m, "on_per_edit": sorted(set(on_counts)),
                "orthogonal": all(c == 0 for c in cross), "edits_on_per_run": sorted(sizes),
                "all_off_runs": sizes.count(0), "all_on_runs": sizes.count(self.m), "columns": self.columns,
                "flips": self.flips}


def _best_flips(cols: list[list[int]], m: int) -> list[int]:
    best: tuple | None = None
    best_flips = [1] * m
    for pattern in itertools.product((1, -1), repeat=m):
        sizes = [sum(1 for j in range(m) if cols[r][j] * pattern[j] > 0) for r in range(len(cols))]
        key = (sum((2 * s - m) ** 2 for s in sizes), sizes.count(0) + sizes.count(m))
        if best is None or key < best:
            best, best_flips = key, list(pattern)
    return best_flips


def _pb12_rows(m: int, rng: random.Random) -> tuple[list[list[int]], list[int], list[int]]:
    array = pb12()
    cols = [[row[j] for j in range(m)] for row in array]
    flips = _best_flips(cols, m)
    assign = list(range(m))  # assign[j]: the column (0-based) edit j takes
    rng.shuffle(assign)
    order = list(range(PB12_RUNS))
    rng.shuffle(order)
    rows = [[1 if cols[r][assign[j]] * flips[assign[j]] > 0 else 0 for j in range(m)] for r in order]
    return rows, [assign[j] + 1 for j in range(m)], [flips[assign[j]] for j in range(m)]


def plan_design(edits: list[str], kind: str = "pb12", *, seed: int = 0, repeats: int = 1, pairs: int = 8) -> Design:
    """The design over `edits`; an empty list gives an empty design. ValueError for too many edits, repeated ids or
    an unknown kind."""
    edits = list(edits)
    if kind not in MAX_EDITS:
        raise ValueError(f"unknown design {kind!r}; one of {', '.join(DESIGNS)}")
    if len(edits) > MAX_EDITS[kind]:
        raise ValueError(f"{len(edits)} edits: the {kind} design takes at most {MAX_EDITS[kind]}")
    if len(set(edits)) != len(edits):
        raise ValueError(f"edit ids must be distinct: {edits}")
    if repeats < 1:
        raise ValueError("screen.repeats is at least 1")
    m = len(edits)
    if m == 0:
        return Design(kind=kind, edits=[], rows=[], seed=seed)
    rng = random.Random(seed)
    columns: list[int] = []
    flips: list[int] = []
    pair: list = []
    if kind == "pb12":
        base, columns, flips = _pb12_rows(m, rng)
        pair = [None] * len(base)
    elif kind == "one-at-a-time":
        base = [[1 if j == i else 0 for j in range(m)] for i in range(m)]
        rng.shuffle(base)
        pair = [None] * m
    elif kind == "full":
        base = [list(bits) for bits in itertools.product((0, 1), repeat=m)]
        rng.shuffle(base)
        pair = [None] * len(base)
    else:  # fold-over
        from ...screen.design import fold_over

        x = fold_over(m, pairs, seed=seed % (2 ** 32))
        base = []
        for i, row in enumerate(x.tolist(), start=1):
            base.append([1 if v > 0 else 0 for v in row])
            base.append([1 if v < 0 else 0 for v in row])
            pair += [i, i]
    rows, rep, prs, row_of = [], [], [], []
    for k in range(1, repeats + 1):
        for i, row in enumerate(base):
            rows.append(list(row))
            rep.append(k)
            prs.append(pair[i])
            row_of.append(i)
    return Design(kind=kind, edits=edits, rows=rows, columns=columns, flips=flips, seed=seed, repeat=rep, pair=prs,
                  row_of=row_of)


def probe_count(kind: str, edits: int, repeats: int = 1, pairs: int = 8) -> int:
    """Probe runs the design makes over `edits` edits: pb12 12, fold-over 2 x pairs, one-at-a-time one per edit,
    full 2^edits; each times `repeats`. 0 edits make no probe."""
    if edits <= 0:
        return 0
    per = {"pb12": PB12_RUNS, "fold-over": 2 * pairs, "one-at-a-time": edits, "full": 2 ** edits}[kind]
    return per * max(1, repeats)


def residual_df(kind: str, edits: int, *, repeats: int = 1, pairs: int = 8, base_runs: int = 2,
                base_rows: bool = True) -> int | None:
    """The least-squares fit's residual degrees of freedom at the plan: scored rows minus edits minus the intercept.
    PB12 with 10 edits and 2 base rows: 14 - 11 = 3. None with no edit."""
    if edits <= 0:
        return None
    rows = probe_count(kind, edits, repeats, pairs) + (base_runs if base_rows else 0)
    return rows - edits - 1


def alias_summary(design: Design, *, base_rows: int = 2) -> dict:
    """How much each two-edit interaction leaks into the main effects, on the +1/-1 scale.

    For the least-squares fit with an intercept on the design rows plus `base_rows` all-off rows, the alias matrix
    is (X'X)^-1 X'W, W the interaction columns. Returned per view (`design_only`, `with_base`): the largest leak into
    an edit outside the pair, the largest into an edit inside it (zero for the bare PB12), and the range of each
    edit's standard error on the on-versus-off scale in residual SDs (0.577 for the bare PB12). A view whose rows
    cannot fit the effects is None.
    """
    from .effects import inverse, solve

    out: dict = {}
    m = design.m
    if m < 2:
        return {"with_base": None, "design_only": None}
    for name, extra in (("design_only", 0), ("with_base", base_rows)):
        x = design.signed() + [[-1] * m for _ in range(extra)]
        cols = [[1.0] + [float(v) for v in row] for row in x]
        xtx = [[sum(r[a] * r[b] for r in cols) for b in range(m + 1)] for a in range(m + 1)]
        try:
            inv = inverse(xtx)
        except ValueError:
            out[name] = None
            continue
        outside = inside = 0.0
        for i, j in itertools.combinations(range(m), 2):
            w = [row[i] * row[j] for row in x]
            xtw = [sum(r[a] * wv for r, wv in zip(cols, w)) for a in range(m + 1)]
            coef = solve(xtx, xtw)
            for k in range(m):
                v = abs(coef[k + 1])
                if k in (i, j):
                    inside = max(inside, v)
                else:
                    outside = max(outside, v)
        se = [2.0 * max(inv[k + 1][k + 1], 0.0) ** 0.5 for k in range(m)]
        out[name] = {"max_leak_outside_pair": round(outside, 4), "max_leak_inside_pair": round(inside, 4),
                     "se_factor": [round(min(se), 4), round(max(se), 4)]}
    return out
