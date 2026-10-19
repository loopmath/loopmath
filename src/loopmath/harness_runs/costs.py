"""One cost per rollout, whichever path reads it: the importer, the harness2 reader and the gate's wrapper.

`rollout_cost` is `streams.rollout_usage` plus pricing from loopmath's price table. A model with no price
row keeps its tokens and gets no dollars, with the reason; it is never priced at 0. A partial reported
subtotal is never a rollout's `usd` (spec/harness-runs.md section 4).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .streams import rollout_usage


@lru_cache(maxsize=1)
def _table() -> Any:
    from ..price import load_prices

    return load_prices()


def price_tokens(model: str | None, tokens: dict[str, int]) -> tuple[float | None, str | None]:
    """(usd, None) when the table prices `model`, else (None, the reason)."""
    if not model:
        return None, "no model recorded, so the tokens cannot be priced"
    from ..price import price_run

    rec = {"model": model, "tokens": {"in": int(tokens.get("input") or 0), "cache_read": int(tokens.get("cache_read") or 0),
                                      "cache_write": int(tokens.get("cache_write") or 0),
                                      "out": int(tokens.get("output") or 0)}}
    priced = price_run(rec, _table())
    if priced.get("usd") is None:
        return None, str(priced.get("reason") or f"no price row for {model}")
    return float(priced["usd"]), None


def rollout_cost(paths: str | Path | Iterable[str | Path], kind: str, *, model: str | None = None) -> dict[str, Any]:
    """The `cost` object of a harness-runs row for one rollout's stream files.

    Returns `{usd, tokens, recovered, usd_source, unpriced, complete, incomplete, partial_usd, files,
    skipped_lines}`: OpenCode's own dollars when every billable step reported them, else the whole token
    total priced at `model`, else no dollars and the reason. Incomplete counts never get a `usd`.
    """
    usage = dict(rollout_usage(paths, kind))
    uncosted = usage.pop("uncosted_steps", 0)
    out = {**usage, "unpriced": None}
    if not usage["complete"]:
        out["usd"], out["usd_source"] = None, None
        out["unpriced"] = f"the stream's counts are incomplete ({usage['incomplete']})"
    elif usage["usd"] is None and usage["tokens"]:
        usd, reason = price_tokens(model, usage["tokens"])
        out["usd"] = None if usd is None else round(usd, 6)
        out["usd_source"] = "loopmath-prices" if usd is not None else None
        if reason and uncosted:
            reason += f"; {uncosted} billable step(s) reported no cost"
        out["unpriced"] = reason
    return out
