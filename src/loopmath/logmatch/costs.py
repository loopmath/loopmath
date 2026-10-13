"""Dollars from four token streams under the tariff in effect at the attempt's time.

`price_attempt` prices one model's four streams. `cost_record` builds the OCP
`attempt.cost` object for a match: the four streams summed over the session
and its children, dollars summed over the parts with each part priced at its
own model, `basis` (`measured` for a verified match, `allocated` for a
heuristic one), the tariff, and `ext["dev.loopmath.logmatch"]` with the tier,
the children, the per-part breakdown and, for a match clipped to a
`--session self` attempt's window, `clip`.

When the tokens ran on more than one model, `ext["dev.loopmath.model_tokens"]`
splits the same streams by model under the OCP cost field names, for `price.price_model_tokens` to reprice: tokens under no model
go under `"unknown"`, never priced, and a thread that switched models but
could not be split is `{}`. A single-model attempt has no split. It holds
model ids and counts only.

Dollars are never invented: a part with no model or no price row, or priced
at a row that is only a guess (`todo = true`), leaves `usd` and `tariff` out
of the cost object and names the part in the ext record, with the guessed
figure as `estimate_usd`, the same rule the adapter services follow. A part
with zero tokens costs nothing whatever its model. A match whose session and
children made no model request at all has no dollars either: zero would read
as a real, cheap attempt, so the ext record says `no_requests` instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..ingest.base import Tokens
from ..price import UNLABELLED_MODEL, price_run
from .tariff import UnpricedModel, price_table, tariff_for

EXT_KEY = "dev.loopmath.logmatch"
MODEL_TOKENS_KEY = "dev.loopmath.model_tokens"


def price_attempt(
    tokens: Tokens, model: str, at: str, *, path: str | Path | None = None
) -> tuple[float, dict]:
    """Dollars for `tokens` on `model` at the table's current rate, and the tariff used.

    `at` is the attempt's time; it does not pick an older rate (see
    `tariff`), so the figure is a repricing under the returned tariff
    `{id, date, source}`. Raises `UnpricedModel` when the table has no row
    for `model`. A row that is a guess still prices; its tariff `source` says
    so ("guess: ...").
    """
    tariff = tariff_for(model, at, path=path)
    priced = price_run({"model": model, "tokens": tokens.as_dict()}, price_table(path))
    if not priced["priced"]:
        raise UnpricedModel(priced["reason"])
    return float(priced["usd"]), tariff


def _stream_fields(tokens: Tokens) -> dict[str, int]:
    return {
        "input_tokens": int(tokens.in_),
        "cached_input_tokens": int(tokens.cache_read),
        "cache_creation_tokens": int(tokens.cache_write),
        "output_tokens": int(tokens.out),
    }


def model_tokens_from_parts(parts: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """`{model_id: {OCP cost token fields}}` summed over logmatch `parts`.

    `parts` are `ext["dev.loopmath.logmatch"]["parts"]`, each with `model`
    and `tokens` as `Tokens.as_dict()`. A part with no tokens is left out;
    tokens under no model go under `"unknown"`.
    """
    split: dict[str, dict[str, int]] = {}
    for part in parts:
        counts = part.get("tokens") or {}
        tokens = Tokens(
            in_=counts.get("in", 0),
            cache_read=counts.get("cache_read", 0),
            cache_write=counts.get("cache_write", 0),
            out=counts.get("out", 0),
        )
        if tokens.total == 0:
            continue
        model = part.get("model") or UNLABELLED_MODEL
        fields = _stream_fields(tokens)
        if model in split:
            fields = {name: n + split[model][name] for name, n in fields.items()}
        split[model] = fields
    return split


def cost_record(match: Any, *, path: str | Path | None = None) -> dict[str, Any]:
    """The OCP `attempt.cost` object for `match` (see the module docstring)."""
    table = price_table(path)
    at = match.started_at
    parts = list(getattr(match, "parts", None) or [])
    if not parts:
        parts = [
            {
                "session": match.session_id,
                "role": "session",
                "harness": match.harness,
                "model": match.model,
                "tokens": match.tokens,
            }
        ]

    total = 0.0
    tariff: dict | None = None
    withheld = False
    floor = False
    ext_parts = []
    for part in parts:
        model = part.get("model")
        tokens: Tokens = part["tokens"]
        entry: dict[str, Any] = {
            "session": part["session"],
            "role": part["role"],
            "model": model,
            "tokens": tokens.as_dict(),
        }
        if part.get("parent"):
            entry["parent"] = part["parent"]
        if tokens.total == 0:
            entry["usd"] = 0.0  # no request, nothing to price, whatever the model
        elif not model:
            entry["unpriced"] = part.get("unpriced") or "no model in the log"
            withheld = True
        else:
            try:
                usd, part_tariff = price_attempt(tokens, model, at, path=path)
            except UnpricedModel:
                entry["unpriced"] = f"no price row for {model}"
                withheld = True
            else:
                entry["tariff_date"] = part_tariff["date"]
                if table.is_todo(model):
                    entry["estimate_usd"] = round(usd, 6)
                    entry["unpriced"] = f"the price row for {model} is a guess"
                    withheld = True
                else:
                    entry["usd"] = round(usd, 6)
                    total += usd
                    if tariff is None or part["role"] == "session":
                        tariff = part_tariff
                # Codex logs no cache writes, which GPT-6 bills: such a part
                # is priced as if it wrote none (prices.toml, GPT-6 notes).
                rates = table.rate(model) or {}
                if part.get("harness") == "codex" and rates.get("cache_write", 0) > 0:
                    floor = True
        ext_parts.append(entry)

    cost: dict[str, Any] = _stream_fields(match.tokens)
    no_requests = match.tokens.total == 0
    if not withheld and not no_requests:
        cost["usd"] = round(total, 6)
        if tariff is not None:
            cost["tariff"] = tariff
    cost["basis"] = "measured" if match.tier == "verified" else "allocated"
    ext: dict[str, Any] = {
        "tier": match.tier,
        "session": match.session_id,
        "children": list(match.children),
        "parts": ext_parts,
    }
    reason = getattr(match, "reason", "")
    if reason:
        ext["reason"] = reason
    clip = getattr(match, "clip", None)
    if clip:
        ext["clip"] = dict(clip)
    if no_requests:
        # Zero dollars would read as a real cheap attempt; there is no cost
        # observation here at all.
        ext["no_requests"] = "the session made no model requests, so it has no dollars"
    if floor:
        ext["usd_is_floor"] = "Codex logs no cache writes; GPT-6 bills them"
    cost["ext"] = {EXT_KEY: ext}
    split = model_tokens_from_parts(ext_parts)
    if split and all(p.get("unsplit") for p in parts if p["tokens"].total):
        cost["ext"][MODEL_TOKENS_KEY] = {}  # more than one model, no split
    elif len(split) > 1 or UNLABELLED_MODEL in split:
        cost["ext"][MODEL_TOKENS_KEY] = split
    return cost
