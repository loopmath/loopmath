"""Map a function over independent inputs in worker processes, results in input order.

Onboarding parses and scans thousands of session files, each on its own. The
worker count comes from `LOOPMATH_WORKERS` (1 keeps everything in this process);
the default is the CPU count, at most `MAX_DEFAULT`. Short lists run in process,
where starting workers would cost more than it saves.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Sequence

MAX_DEFAULT = 8
MIN_ITEMS = 64


def workers() -> int:
    raw = os.environ.get("LOOPMATH_WORKERS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, min(MAX_DEFAULT, os.cpu_count() or 1))


def ordered_map(fn: Callable[[Any], Any], items: Sequence[Any], *, weights: Sequence[int] | None = None,
                done: Callable[[int, int], None] | None = None) -> list[Any]:
    """`[fn(x) for x in items]`, in worker processes when it pays. `fn` must be a
    module-level function. Heaviest items (by `weights`) start first so one large
    file does not finish alone at the end. An exception in `fn` is raised here."""
    n = workers()
    if n <= 1 or len(items) < MIN_ITEMS:
        out = []
        for i, x in enumerate(items):
            out.append(fn(x))
            if done is not None:
                done(i + 1, len(items))
        return out
    order = sorted(range(len(items)), key=lambda i: -weights[i]) if weights is not None else range(len(items))
    out: list[Any] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=min(n, len(items))) as ex:
        futures = {ex.submit(fn, items[i]): i for i in order}
        for k, fut in enumerate(as_completed(futures), 1):
            out[futures[fut]] = fut.result()
            if done is not None:
                done(k, len(items))
    return out
