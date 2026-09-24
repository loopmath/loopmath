"""Price-table schema and staleness validation."""

from __future__ import annotations

globals().pop("open", None)

import datetime as _dt
import importlib as _importlib
import math as _math
import sys as _sys
import tomllib
from pathlib import Path
from types import FunctionType as _FunctionType


# Standalone definitions let this module finish its own initialization before
# importing ``loopmath.price``.  Synchronization below replaces these with the
# original module's objects when price is the module being imported/reloaded.
DEFAULT_PRICES = Path(__file__).resolve().parent / "prices.toml"
_RATE_KEYS = ("input", "cache_read", "cache_write", "output")
_MODEL_METADATA_KEYS = ("source", "todo", "zero_ok")


# ---------------------------------------------------------------------------
# The price table validator (SPEC section 8, item 2).
#
# `load_prices` is a loader: it raises on the first thing that would make it
# invent a number, and it stops there. The validator below is the opposite
# shape on purpose. It parses the TOML itself rather than going through
# `load_prices`, so that one broken entry does not hide the other sixteen,
# and it reports EVERY problem it finds in one pass, split into hard errors
# (the table is wrong) and warnings (the table is usable but the reader is
# owed a sentence about it).
#
# The sign rules are the part worth stating carefully, because zero is not
# one thing here:
#   * A NEGATIVE rate is a hard error. No provider pays you per token; a
#     negative rate is a typo or a sign flip, and pricing a run against it
#     would produce a dollar figure that is not merely uncertain but
#     backwards.
#   * A ZERO rate is LEGAL. OpenAI genuinely charges nothing for cache
#     writes, and every GPT entry in the packaged table is 0.00 on that
#     stream for that reason. Erroring on zero would make the shipped table
#     invalid and push a real fact out of the file.
#   * But an unmarked zero is also exactly what a dropped digit looks like,
#     so a zero is never silent: it produces a warning naming the model and
#     the stream. `zero_ok = true` is deliberately limited to `cache_write`,
#     the only free stream supported by the current table, or to a row whose
#     four rates are all zero (a free model: one dropped digit cannot zero
#     every rate). Silent typo zeros in input, cache_read, or output of a
#     paid row never slip through; real free cache writes never block.
# ---------------------------------------------------------------------------

# How old `as_of` may get before the table is called stale. Thirty days is a
# warning, never an error: an old table still prices runs, and refusing to run
# on it would be a worse failure than saying how old it is.
STALE_AFTER_DAYS = 30


def _as_of_date(as_of) -> _dt.date | None:
    """`as_of` as a date, or None when it is not one.

    TOML gives an unquoted `2026-08-31` back as a `datetime.date` already,
    while the packaged table quotes it and yields a string; both spellings
    are the same fact and both are accepted.
    """
    if isinstance(as_of, _dt.datetime):
        return as_of.date()
    if isinstance(as_of, _dt.date):
        return as_of
    try:
        return _dt.date.fromisoformat(str(as_of))
    except (TypeError, ValueError):
        return None


def validate_prices(path: str | Path | None = None, *, today: _dt.date | None = None) -> dict:
    """Check a price table's schema, staleness, sign rules and todo flags.

    Returns a dict with `errors` and `warnings` (both lists of plain
    sentences, in file order), `ok` (no errors), `path`, `as_of` and
    `todo_models`. Nothing is raised for a bad table: reporting every problem
    at once is the whole point, and a caller that wants an exception can
    check `ok`.

    Hard errors: an unreadable or unparseable file, a missing or
    unparseable `as_of`, an unexpected top-level field, an unknown field in a
    model table, an entry missing one or more of the four rates, a non-numeric
    or non-finite rate, a non-boolean marker, and any negative rate.

    Warnings: an `as_of` more than `STALE_AFTER_DAYS` days before `today`
    (defaults to the local date), every `todo = true` entry listed by name,
    and every zero rate not covered by `zero_ok = true` (for `cache_write`,
    or for every rate of a free model whose four rates are all zero), named
    by model and stream.
    """
    p = Path(path) if path is not None else DEFAULT_PRICES
    errors: list[str] = []
    warnings: list[str] = []

    try:
        with open(p, "rb") as fh:
            raw = tomllib.load(fh)
    except OSError as exc:
        return {
            "ok": False,
            "path": str(p),
            "as_of": None,
            "errors": [f"price table {p} could not be read: {exc}"],
            "warnings": [],
            "todo_models": [],
        }
    except tomllib.TOMLDecodeError as exc:
        return {
            "ok": False,
            "path": str(p),
            "as_of": None,
            "errors": [f"price table {p} is not valid TOML: {exc}"],
            "warnings": [],
            "todo_models": [],
        }

    as_of = raw.get("as_of")
    as_of_date = None
    if not as_of:
        errors.append(f"price table {p} has no 'as_of' date")
    else:
        as_of_date = _as_of_date(as_of)
        if as_of_date is None:
            errors.append(f"price table {p} has an unparseable 'as_of' date: {as_of!r}")
        else:
            age = ((today or _dt.date.today()) - as_of_date).days
            if age > STALE_AFTER_DAYS:
                warnings.append(
                    f"WARNING: the price table is {age} days old (as of {as_of_date}, "
                    f"threshold {STALE_AFTER_DAYS} days). Rates move; every dollar "
                    "figure derived from it may be out of date."
                )

    todo_models: list[str] = []
    zero_notes: list[str] = []
    n_models = 0
    for key, val in raw.items():
        if key == "as_of":
            continue
        if not isinstance(val, dict):
            errors.append(
                f"price table has an unexpected top-level field '{key}'; "
                "every field except 'as_of' must be a model table"
            )
            continue
        n_models += 1
        model = key
        for field_name in val:
            if field_name not in _RATE_KEYS and field_name not in _MODEL_METADATA_KEYS:
                errors.append(
                    f"price table entry '{model}' has an unknown field "
                    f"'{field_name}'; allowed metadata fields are: "
                    f"{', '.join(_MODEL_METADATA_KEYS)}"
                )
        missing = [r for r in _RATE_KEYS if r not in val]
        if missing:
            errors.append(
                f"price table entry '{model}' is missing rate(s): {', '.join(missing)}"
            )

        zero_ok_value = val.get("zero_ok", False)
        if not isinstance(zero_ok_value, bool):
            errors.append(
                f"price table entry '{model}' has a non-boolean 'zero_ok' "
                f"marker: {zero_ok_value!r}"
            )
            zero_ok = False
        else:
            zero_ok = zero_ok_value

        todo_value = val.get("todo", False)
        if not isinstance(todo_value, bool):
            errors.append(
                f"price table entry '{model}' has a non-boolean 'todo' "
                f"marker: {todo_value!r}"
            )
            todo = False
        else:
            todo = todo_value

        # A free model: marked, and all four rates present and zero.
        free = zero_ok and all(
            isinstance(val.get(k), (int, float)) and not isinstance(val.get(k), bool) and val.get(k) == 0
            for k in _RATE_KEYS
        )
        for rate_key in _RATE_KEYS:
            if rate_key not in val:
                continue
            rate_value = val[rate_key]
            if isinstance(rate_value, bool) or not isinstance(rate_value, (int, float)):
                errors.append(
                    f"price table entry '{model}' has a non-numeric '{rate_key}' "
                    f"rate: {rate_value!r}"
                )
                continue
            rate = float(rate_value)
            if not _math.isfinite(rate):
                errors.append(
                    f"price table entry '{model}' has a non-numeric '{rate_key}' "
                    f"rate: {rate_value!r}"
                )
                continue
            if rate < 0:
                errors.append(
                    f"price table entry '{model}' has a negative '{rate_key}' rate "
                    f"({rate:g} $/Mtok). No provider pays per token; this is a typo "
                    "or a sign flip."
                )
            elif rate == 0 and not free and not (rate_key == "cache_write" and zero_ok):
                zero_notes.append(f"{model}.{rate_key}")
        if todo:
            todo_models.append(model)

    if not n_models:
        errors.append(f"price table {p} has no model entries")

    if zero_notes:
        warnings.append(
            f"WARNING: {len(zero_notes)} rate(s) are zero and not marked as intended "
            f"({', '.join(zero_notes)}). A free stream is real (OpenAI charges nothing "
            "for cache writes), but so is a dropped digit. Add 'zero_ok = true' to the "
            "entry only when its cache_write zero is meant, or when the model is free "
            "and all four of its rates are zero."
        )

    if todo_models:
        warnings.append(
            f"WARNING: {len(todo_models)} of {n_models} entries are placeholder rates "
            f"flagged todo = true ({', '.join(todo_models)}). These are best "
            "public-price guesses, not confirmed rates, and every dollar figure that "
            "uses one inherits that."
        )

    return {
        "ok": not errors,
        "path": str(p),
        "as_of": str(as_of) if as_of else None,
        "errors": errors,
        "warnings": warnings,
        "todo_models": todo_models,
    }


def validation_lines(result: dict) -> list[str]:
    """Render `validate_prices` output as terminal lines, errors first.

    Always ends with a verdict line, so a clean table says so out loud rather
    than printing nothing and leaving the reader to guess whether the check
    actually ran.
    """
    lines = [f"checking {result['path']}"]
    lines.extend(f"ERROR: {e}" for e in result["errors"])
    lines.extend(result["warnings"])
    if result["ok"]:
        n = len(result["warnings"])
        tail = f" ({n} warning{'s' if n != 1 else ''})" if n else " (no warnings)"
        lines.append(f"price table is valid{tail}.")
    else:
        n = len(result["errors"])
        lines.append(f"price table is INVALID: {n} error{'s' if n != 1 else ''}.")
    return lines


_VALIDATOR_TEMPLATES = (_as_of_date, validate_prices, validation_lines)
_VALIDATOR_GLOBAL_NAMES = (
    "DEFAULT_PRICES",
    "Path",
    "STALE_AFTER_DAYS",
    "_MODEL_METADATA_KEYS",
    "_RATE_KEYS",
    "_dt",
    "_math",
    "tomllib",
)
_PRICE_GLOBAL_NAMES = tuple(
    name for name in _VALIDATOR_GLOBAL_NAMES if name != "STALE_AFTER_DAYS"
)


def _rebuild_exports(price_globals: dict) -> None:
    """Publish validators whose code lives here and globals live in price."""
    exports = {}
    for template in _VALIDATOR_TEMPLATES:
        function = _FunctionType(
            template.__code__,
            price_globals,
            template.__name__,
            template.__defaults__,
            template.__closure__,
        )
        function.__module__ = "loopmath.price"
        function.__qualname__ = template.__qualname__
        function.__doc__ = template.__doc__
        function.__annotations__ = dict(template.__annotations__)
        function.__kwdefaults__ = (
            dict(template.__kwdefaults__) if template.__kwdefaults__ is not None else None
        )
        function.__dict__.update(template.__dict__)
        if hasattr(template, "__type_params__"):
            function.__type_params__ = template.__type_params__
        exports[function.__name__] = function
    price_globals.update(exports)
    globals().update(exports)


def _copy_validator_globals(source: dict, target: dict) -> None:
    for name in _VALIDATOR_GLOBAL_NAMES:
        target[name] = source[name]
    target.pop("open", None)


def _synchronize_from_price(price_globals: dict) -> None:
    """Restore this module from a freshly imported or reloaded price module."""
    for name in _PRICE_GLOBAL_NAMES:
        globals()[name] = price_globals[name]
    globals().pop("open", None)
    price_globals["STALE_AFTER_DAYS"] = STALE_AFTER_DAYS
    _rebuild_exports(price_globals)


def _synchronize_from_validation(price_globals: dict) -> None:
    """Restore price exports and globals after this module is reloaded."""
    _copy_validator_globals(globals(), price_globals)
    _rebuild_exports(price_globals)


_price_module = _sys.modules.get(f"{__package__}.price")
if _price_module is None:
    _price_module = _importlib.import_module(".price", __package__)
if getattr(_price_module, "_PRICE_MODULE_READY", False):
    _synchronize_from_validation(_price_module.__dict__)
