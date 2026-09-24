"""Dated tariffs from prices.toml: {id, date, source}.

Owner: lane 02. Spec: design/0.1/ (01 section 2.5, 03 section 6).

A tariff names the rates an attempt was priced at, so a run file stays
traceable after the table changes:

- `id`: the first 12 hex digits of the sha256 of the price table's bytes. Any
  edit to the table gives a new id.
- `date`: the date the row's rates were read off the provider's page, taken
  from "checked YYYY-MM-DD" in the row's `source`; a row without one takes the
  table's `as_of`.
- `source`: the row's `source` text (provider URL, or how a guess was made).

Current rates, not invoices. The table holds one rate per model, so every
attempt, however old, is priced at the rate the table holds now: `at` is
checked (a caller cannot pass something that is not a timestamp) but does
not select an older rate. A cost for an attempt before the tariff `date` is
a repricing at that tariff, not a reconstruction of what was billed then;
the tariff `id` and `date` say exactly which rates were used. A rate history
needs a loader change in `price.py` (lane 02 report, requests).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..ingest.base import parse_ts
from ..price import DEFAULT_PRICES, PriceTable, load_prices


class UnpricedModel(LookupError):
    """The price table has no row for this model; it is never priced at zero."""


_CHECKED_RE = re.compile(r"\bchecked (\d{4}-\d{2}-\d{2})\b")
_TABLES: dict[tuple[str, int, int], tuple[PriceTable, str]] = {}


def _table_and_id(path: str | Path | None = None) -> tuple[PriceTable, str]:
    """The loaded table and its id, cached on the file's path, mtime and size."""
    p = Path(path) if path is not None else DEFAULT_PRICES
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    cached = _TABLES.get(key)
    if cached is None:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        cached = (load_prices(p), digest)
        _TABLES[key] = cached
    return cached


def price_table(path: str | Path | None = None) -> PriceTable:
    """The price table at `path` (the packaged one by default)."""
    return _table_and_id(path)[0]


def table_id(path: str | Path | None = None) -> str:
    """The tariff id of the price table at `path`: its sha256 prefix."""
    return _table_and_id(path)[1]


def tariff_for(model: str, at: str, *, path: str | Path | None = None) -> dict:
    """The tariff `{id, date, source}` that prices `model` today, for an attempt at `at`.

    The rate is the table's current one whatever `at` is (see the module
    docstring: a repricing, not the invoice of the day). Raises
    `UnpricedModel` when the table has no row for `model`, and `ValueError`
    when `at` is not a timestamp.
    """
    if parse_ts(at) is None:
        raise ValueError(f"not a timestamp: {at!r}")
    table, digest = _table_and_id(path)
    if table.rate(model) is None:
        raise UnpricedModel(f"no price row for model {model!r} in {table.path}")
    source = table.source_for(model) or ""
    checked = _CHECKED_RE.search(source)
    return {
        "id": digest,
        "date": checked.group(1) if checked else table.as_of,
        "source": source,
    }
