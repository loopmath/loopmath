"""`loopmath verify-receipts` with no ledger: check the receipts in the store (0.2.4).

Each `receipts/<id>.json` must read as an object whose `id` is its file name, name a run the store
holds, and carry a `before` (the prediction made at `run start`) with a cost interval and a chance of
success. A receipt with an `after` must belong to a finished run, its dollars must be the run's, and
its `scored.cost_in_interval` must agree with those dollars and `before`'s interval. A receipt without
`after` waits for its run to finish. Read only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..recommend.receipts import realized_from_doc
from . import run_names
from . import runs as R
from .home import NotFound, Store, StoreError
from .lock import read_json


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _get(d: Any, *path: str) -> Any:
    for key in path:
        d = d.get(key) if isinstance(d, dict) else None
    return d


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 + 1e-9 * max(abs(a), abs(b))


def check_receipt(store: Store, path: Path) -> tuple[str, str | None]:
    """(`scored`, `waiting` or `failed`, and the reason when failed)."""
    try:
        rc = read_json(path)
    except (OSError, ValueError) as exc:
        return "failed", f"cannot read it ({exc.__class__.__name__})"
    if not isinstance(rc, dict):
        return "failed", "not a JSON object"
    if rc.get("id") != path.stem:
        return "failed", f"its id {rc.get('id')!r} is not its file name"
    run = rc.get("run")
    if run_names.file_stem(run) is None:  # the store's run id rule, `:` included
        return "failed", f"names no valid run ({run!r})"
    try:
        doc = store.run_doc(run)
    except (NotFound, StoreError, ValueError):
        return "failed", f"its run {run} is not in the store"
    before = rc.get("before") if isinstance(rc.get("before"), dict) else {}
    usd = _get(before, "cost", "usd")
    usd_lo, usd_mean, usd_hi = (_num(usd.get(k)) for k in ("lo", "mean", "hi")) if isinstance(usd, dict) else (None,) * 3
    p = _num(_get(before, "p_success", "mean"))
    if usd_lo is None or usd_mean is None or usd_hi is None or usd_lo > usd_hi or p is None or not 0 <= p <= 1:
        return "failed", "its prediction (before) is missing or malformed: it needs a cost interval and a chance of success"
    after = rc.get("after")
    if not after:
        return "waiting", None
    if not isinstance(after, dict):
        return "failed", "its result (after) is not an object"
    if not R.is_finished(doc):
        return "failed", f"it has a result but run {run} is not finished"
    usd = _num((after.get("cost") or {}).get("usd")) if isinstance(after.get("cost"), dict) else None
    real = _num(realized_from_doc(doc).get("cost_usd"))
    if (usd is None) != (real is None) or (usd is not None and real is not None and not _same(usd, real)):
        return "failed", f"its cost {usd} is not the run's cost {real}"
    scored = rc.get("scored")
    if not isinstance(scored, dict):
        return "failed", "it has a result but no score (scored)"
    inside = scored.get("cost_in_interval")
    want = None if usd is None else bool(usd_lo <= usd <= usd_hi)
    if inside != want:
        return "failed", f"cost_in_interval is {inside}, but {usd} against {usd_lo} to {usd_hi} gives {want}"
    return "scored", None


def verify_store(store: Store) -> dict[str, Any]:
    """`{folder, checked, scored, waiting, failed: [{receipt, reason}]}`."""
    folder = store.home / "receipts"
    out: dict[str, Any] = {"folder": str(folder), "checked": 0, "scored": 0, "waiting": 0, "failed": []}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        state, reason = check_receipt(store, path)
        out["checked"] += 1
        if state == "failed":
            out["failed"].append({"receipt": path.stem, "reason": reason})
        else:
            out[state] += 1
    return out


def verify_lines(res: dict[str, Any]) -> list[str]:
    n, bad = res["checked"], res["failed"]
    counts = f"{res['scored']} scored, {res['waiting']} waiting for their run to finish"
    if not bad:
        if not n:
            return [f"OK: no receipts yet in {res['folder']} (run start --rec writes one)"]
        return [f"OK: {n} receipt(s) in {res['folder']} check out ({counts})"]
    lines = [f"FAIL: {len(bad)} of {n} receipt(s) in {res['folder']} do not check out ({counts})"]
    lines += [f"  {b['receipt']}: {b['reason']}" for b in bad[:20]]
    if len(bad) > 20:
        lines.append(f"  and {len(bad) - 20} more")
    return lines
