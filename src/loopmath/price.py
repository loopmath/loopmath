"""Tokens to dollars, with loud warnings (SPEC section 5, amended 08-31).

The price table carries four rates per model, in $/Mtok: `input`, `cache_read`,
`cache_write`, `output`. A run record carries four matching token streams:
`in`, `cache_read`, `cache_write`, `out` (see `loopmath.ingest.base.Tokens`). The
two vocabularies use deliberately different words on the `in`/`out` side
(`in` versus `input`, `out` versus `output`), so this module never assumes a
token key and a rate key are the same string. `_TOKEN_TO_RATE` is the one
place that says which token stream is priced by which rate; every loop below
walks the token tuple or the rate tuple, never both interchangeably.

Cache writes get their own rate instead of folding into `input` because the
gap is too large to approximate away: on Anthropic a cache write can cost up
to 20x a cache read (opus-5: $10.00 against $0.50 per Mtok), while OpenAI
charges nothing for a cache write at all (those table entries are 0.00). An
earlier version of this record folded cache writes into `in` on the theory
that writes bill at or above the input rate, which was true but lossy in both
directions at once: it overcharged Anthropic runs and left the OpenAI figure
looking dearer than it is. The streams are now one-to-one with the rates, and
nothing is summed on the way in.

Anthropic entries in `prices.toml` are keyed by full model name
(`claude-opus-5`), while a parsed run record carries the short canonical name
(`opus-5`, from `loopmath.ingest.base.canonical_model`). Model lookup tries an
exact key match first and falls back to canonicalizing both the table key and
the record's model string, so a run is never left unpriced just because the
two sides spell a model differently.

Honest limit, unchanged: most of `prices.toml` is `todo = true`, meaning best
public-price guesses rather than confirmed rates. This module never hides
that. It prices `todo` models anyway, because a flagged estimate is more
useful than nothing, but every caller-facing surface (`price_all`,
`warning_lines`) must say so loudly. It also never prices an unknown model at
zero: zero looks like a real, cheap answer, and SPEC section 5 forbids
exactly that failure mode.

Second honest limit, new in this amendment: GPT-5.6 long-context requests
(roughly 272k tokens or more) carry a surcharge that a per-attempt record
cannot see, so those requests are priced at the standard rate and any GPT-5.6
dollar figure this module produces is a floor, not an exact bill.
`warning_lines` says so by name whenever it applies, rather than modeling a
threshold it cannot verify per run.
"""

from __future__ import annotations

import datetime as _dt
import importlib as _importlib
import math as _math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .ingest.base import canonical_model


_PRICE_MODULE_RELOAD = globals().get("_PRICE_MODULE_READY", False)
_PRICE_MODULE_READY = False
globals().pop("open", None)

# The packaged price table, shipped alongside this module (see
# tool.setuptools.package-data in pyproject.toml). Callers that want a
# different table (tests, a future `--prices PATH` flag) pass `path=`.
DEFAULT_PRICES = Path(__file__).resolve().parent / "prices.toml"

# The run record's token streams (loopmath.ingest.base.Tokens.as_dict) and the
# price table's rate keys, kept as two separate tuples on purpose: the two
# vocabularies differ (`in`/`out` versus `input`/`output`) and a loop that
# conflated them would silently price the wrong stream. `_TOKEN_TO_RATE` is
# the single explicit mapping between them; nothing else should guess it.
_TOKEN_STREAMS = ("in", "cache_read", "cache_write", "out")
_RATE_KEYS = ("input", "cache_read", "cache_write", "output")
_TOKEN_TO_RATE = {
    "in": "input",
    "cache_read": "cache_read",
    "cache_write": "cache_write",
    "out": "output",
}

# Rate tables may carry these annotations in addition to the four required
# rates. Keep this allowlist explicit: a misspelled rate or metadata field
# must not become inert TOML that looks like it is being honored.
_MODEL_METADATA_KEYS = ("source", "todo", "zero_ok")


@dataclass
class PriceTable:
    """A loaded price table: as-of date, per-model $/Mtok rates, todo flags, provenance."""

    as_of: str
    rates: dict[str, dict[str, float]]  # table key -> {"input","cache_read","cache_write","output"} $/Mtok
    todo: set[str]  # table keys whose rates are placeholders, not confirmed
    source: dict[str, str]  # table key -> provenance string, e.g. "derived from billing (exact)"
    path: str
    canonical_index: dict[str, str] = field(default_factory=dict)  # canonical model name -> table key

    def _resolve(self, model: str | None) -> str | None:
        """The table key for `model`: exact match first, canonicalized fallback.

        A run record may carry a model's short canonical name (`opus-5`) while
        the table keys the same model by its full name (`claude-opus-5`). An
        exact match is tried first so a table that already uses short names
        (as every `todo` entry here does) is not paid a canonicalization tax;
        the fallback runs both the table key and `model` through
        `canonical_model` so either spelling resolves to the same entry.
        """
        if model is None:
            return None
        if model in self.rates:
            return model
        canon = canonical_model(model)
        if canon is None:
            return None
        return self.canonical_index.get(canon)

    def rate(self, model: str | None) -> dict[str, float] | None:
        """The $/Mtok rates for `model`, or None when the model is unpriced."""
        key = self._resolve(model)
        return self.rates.get(key) if key is not None else None

    def is_todo(self, model: str | None) -> bool:
        """Whether `model`'s rates are a placeholder estimate, not confirmed."""
        key = self._resolve(model)
        return key is not None and key in self.todo

    def source_for(self, model: str | None) -> str | None:
        """The provenance string for `model`'s rates, or None if there is none."""
        key = self._resolve(model)
        return self.source.get(key) if key is not None else None

def load_prices(path: str | Path | None = None) -> PriceTable:
    """Load a price table from TOML (the packaged table by default).

    Validation is deliberately strict: a table with no `as_of`, a model
    entry missing a rate, or an unknown model-table key is a data bug, and
    silently accepting any of them would produce a dollar figure that looks
    fine and is wrong. Each raises `ValueError` naming the bad entry.
    """
    p = Path(path) if path is not None else DEFAULT_PRICES
    with open(p, "rb") as fh:
        raw = tomllib.load(fh)

    as_of = raw.get("as_of")
    if not as_of:
        raise ValueError(f"price table {p} has no 'as_of' date")

    rates: dict[str, dict[str, float]] = {}
    todo: set[str] = set()
    source: dict[str, str] = {}
    allowed_model_keys = frozenset((*_RATE_KEYS, *_MODEL_METADATA_KEYS))
    for key, val in raw.items():
        if key == "as_of":
            continue
        if not isinstance(val, dict):
            # Skip non-table top-level keys rather than erroring: a stray
            # scalar setting is not a model entry and should not block load.
            continue
        model = key
        unknown = sorted(set(val) - allowed_model_keys)
        if unknown:
            raise ValueError(
                f"price table entry '{model}' has unknown key(s): {', '.join(unknown)}"
            )
        missing = [r for r in _RATE_KEYS if r not in val]
        if missing:
            raise ValueError(
                f"price table entry '{model}' is missing rate(s): {', '.join(missing)}"
            )
        rates[model] = {r: float(val[r]) for r in _RATE_KEYS}
        if val.get("todo"):
            todo.add(model)
        src = val.get("source")
        if src is not None:
            source[model] = str(src)

    # Built once at load time rather than per lookup: every table key run
    # through canonical_model, first occurrence wins on a collision (none
    # expected in practice, but a silent overwrite would be harder to notice
    # than a documented "first wins" rule).
    canonical_index: dict[str, str] = {}
    for model in rates:
        canon = canonical_model(model)
        if canon is not None and canon not in canonical_index:
            canonical_index[canon] = model

    return PriceTable(
        as_of=as_of,
        rates=rates,
        todo=todo,
        source=source,
        path=str(p),
        canonical_index=canonical_index,
    )


def price_run(record: dict, table: PriceTable) -> dict:
    """Price one run record's tokens against `table`.

    Returns a dict with `usd`, `usd_breakdown`, `priced`, `todo`, `model`, and
    `reason`. `usd`/`usd_breakdown` are None (never 0.0) whenever the model is
    unknown or absent, so callers can never mistake "not priced" for "free".

    The token block is validated just as strictly: `tokens` must be a dict
    carrying all four streams in `_TOKEN_STREAMS`, each a finite, non-negative
    number. A missing stream, a missing `tokens` block, or a non-numeric
    stream all return unpriced with a `reason` naming the problem, rather
    than defaulting the stream to 0 tokens. A stream that is present and
    legitimately `0` is priced normally: zero tokens is a real value, a
    missing count is not, and conflating the two would understate the total
    instead of surfacing the exclusion (SPEC sections 0 and 5).

    `usd_breakdown` is keyed by the four TOKEN stream names (`in`,
    `cache_read`, `cache_write`, `out`), not the table's rate names, so a
    caller can line it up directly against the record's `tokens` block.

    A record whose session ran more than one model carries `tokens_by_model`
    (`RunRecord.to_dict`), and each part is priced at its own model's rate
    and summed (Analyst D62). When any part has no price row, or has tokens
    under no model, or the session could not be split by model, `usd` is
    None and `unpriced_models` names the parts: no part is ever priced as
    another model.
    """
    parts = record.get("tokens_by_model")
    if isinstance(parts, list):
        return _price_parts(record, parts, table)
    return _price_one(record, table)


# The OCP cost record's token fields for the four streams; `cache_write` is
# `cache_creation_tokens`, else its 5m and 1h halves, else 0 (it is optional).
_OCP_TOKEN_FIELDS = {"in": "input_tokens", "cache_read": "cached_input_tokens", "out": "output_tokens"}
_CACHE_WRITE_HALVES = ("cache_creation_5m_tokens", "cache_creation_1h_tokens")
# The split's key for tokens no model is known for; never priced (Analyst D71).
UNLABELLED_MODEL = "unknown"


def price_model_tokens(model_tokens: dict, table: PriceTable) -> dict:
    """Price a per-model token split, each model at its own rate, summed (Analyst D62, D67, D71).

    `model_tokens` is `cost.ext["dev.loopmath.model_tokens"]`:
    `{model_id: {input_tokens, cached_input_tokens, cache_creation_tokens,
    output_tokens}}`, the OCP cost record's token fields per model. Returns
    `price_run`'s dict with `model` None: when a model has no price row or
    is `"unknown"` (or None), `usd` is None and `unpriced_models` names
    them, so no part is ever priced as another model; `{}` (more than one
    model, no split) and a missing or malformed count withhold too. A row
    that is only a guess prices, with `todo` True, as in `price_run`.
    """
    if not isinstance(model_tokens, dict):
        return _price_parts({}, [None], table)
    return _price_parts(
        {}, [{"model": model, "tokens": _ocp_streams(fields)} for model, fields in model_tokens.items()], table
    )


def _ocp_streams(fields: dict) -> dict | None:
    """One model's OCP token fields as the four streams `_price_one` reads."""
    if not isinstance(fields, dict):
        return None
    tokens = {stream: fields[name] for stream, name in _OCP_TOKEN_FIELDS.items() if name in fields}
    write = fields.get("cache_creation_tokens")
    if write is None:
        try:
            write = sum(fields[k] for k in _CACHE_WRITE_HALVES if fields.get(k) is not None)
        except TypeError:
            pass  # a non-numeric half stays None and is withheld as one
    tokens["cache_write"] = write
    return tokens


def _price_parts(record: dict, parts: list, table: PriceTable) -> dict:
    """Every model part of a mixed-model record at its own rate, summed (D62)."""
    model = record.get("model")

    def withheld(reason: str, unpriced: list | None = None) -> dict:
        out = {
            "usd": None,
            "usd_breakdown": None,
            "priced": False,
            "todo": False,
            "model": model,
            "reason": reason,
        }
        if unpriced is not None:
            out["unpriced_models"] = unpriced
        return out

    if not parts:
        return withheld("run switched models and its tokens could not be split by model")
    breakdown = dict.fromkeys(_TOKEN_STREAMS, 0.0)
    todo = False
    unpriced: list = []
    for part in parts:
        part_model = part.get("model") if isinstance(part, dict) else None
        if not isinstance(part, dict) or not isinstance(part.get("tokens"), dict):
            return withheld("run has a malformed per-model token part")
        if part_model in (None, UNLABELLED_MODEL) or table.rate(part_model) is None:
            unpriced.append(part_model)
            continue
        one = _price_one(part, table)
        if not one["priced"]:
            return withheld(one["reason"])
        for stream, usd in one["usd_breakdown"].items():
            breakdown[stream] += usd
        todo = todo or one["todo"]
    if unpriced:
        names = ", ".join(
            f"'{m}'" if m not in (None, UNLABELLED_MODEL) else "(tokens with no model label)" for m in unpriced
        )
        return withheld(
            f"no price entry for model {names} in a per-model split; "
            "its dollars are withheld, never priced as another model",
            unpriced,
        )
    return {
        "usd": sum(breakdown.values()),
        "usd_breakdown": breakdown,
        "priced": True,
        "todo": todo,
        "model": model,
        "reason": None,
    }


def _price_one(record: dict, table: PriceTable) -> dict:
    """All of `record`'s tokens at its one model's rate."""
    model = record.get("model")
    if model is None:
        return {
            "usd": None,
            "usd_breakdown": None,
            "priced": False,
            "todo": False,
            "model": None,
            "reason": "run has no model label",
        }

    rates = table.rate(model)
    if rates is None:
        return {
            "usd": None,
            "usd_breakdown": None,
            "priced": False,
            "todo": False,
            "model": model,
            "reason": f"no price entry for model '{model}'",
        }

    tokens = record.get("tokens")
    if not isinstance(tokens, dict):
        return {
            "usd": None,
            "usd_breakdown": None,
            "priced": False,
            "todo": False,
            "model": model,
            "reason": "run has no token counts",
        }

    for stream in _TOKEN_STREAMS:
        if stream not in tokens:
            return {
                "usd": None,
                "usd_breakdown": None,
                "priced": False,
                "todo": False,
                "model": model,
                "reason": f"run is missing the '{stream}' token count",
            }
        try:
            count = float(tokens[stream])
        except (TypeError, ValueError):
            return {
                "usd": None,
                "usd_breakdown": None,
                "priced": False,
                "todo": False,
                "model": model,
                "reason": f"run has a non-numeric '{stream}' token count",
            }
        if not _math.isfinite(count) or count < 0:
            return {
                "usd": None,
                "usd_breakdown": None,
                "priced": False,
                "todo": False,
                "model": model,
                "reason": f"run has a non-numeric '{stream}' token count",
            }

    breakdown = {
        stream: (float(tokens[stream]) / 1_000_000.0) * rates[_TOKEN_TO_RATE[stream]]
        for stream in _TOKEN_STREAMS
    }
    usd = sum(breakdown.values())

    return {
        "usd": usd,
        "usd_breakdown": breakdown,
        "priced": True,
        "todo": table.is_todo(model),
        "model": model,
        "reason": None,
    }


def price_all(records: list[dict], table: PriceTable) -> tuple[list[dict], dict]:
    """Price every record, attaching `usd`/`usd_breakdown` to copies.

    Records are never mutated in place: callers may reuse the input list.
    A priced record is its own file's usage (one session file, or one resumed
    Codex thread) and never includes a child session: a sub-agent's own file
    and a Codex child thread are records of their own, priced on their own,
    so summing a parent and its children counts each once (Analyst D64).
    Returns `(priced_records, warnings)`; `warnings` tallies todo-priced and
    unpriced runs by model, unpriced runs by `reason` (a record with a
    missing or malformed token stream still has a model, so it would
    otherwise blend into the unknown-model tally), and how many priced runs
    used a `gpt-5.6-*` model (the long-context caveat in `warning_lines`
    applies to the whole batch, not to a single run, since a per-attempt
    record cannot say which runs crossed the long-context threshold).
    """
    out: list[dict] = []
    todo_models: dict[str, int] = {}
    unpriced_models: dict[str, int] = {}
    unpriced_reasons: dict[str, int] = {}
    n_todo = 0
    n_unpriced = 0
    n_gpt56_priced = 0

    for record in records:
        result = price_run(record, table)
        priced_record = dict(record)
        priced_record["usd"] = result["usd"]
        priced_record["usd_breakdown"] = result["usd_breakdown"]
        out.append(priced_record)

        if result["priced"] and str(result["model"] or "").lower().startswith("gpt-5.6"):
            n_gpt56_priced += 1

        if result["todo"]:
            n_todo += 1
            todo_models[result["model"]] = todo_models.get(result["model"], 0) + 1
        if not result["priced"]:
            n_unpriced += 1
            # A mixed-model run is tallied under each model that had no price.
            for model in result.get("unpriced_models") or [result["model"]]:
                label = model if model is not None else "no-model-label"
                unpriced_models[label] = unpriced_models.get(label, 0) + 1
            reason = result["reason"] or "unknown reason"
            unpriced_reasons[reason] = unpriced_reasons.get(reason, 0) + 1

    warnings = {
        "todo_models": todo_models,
        "unpriced_models": unpriced_models,
        "unpriced_reasons": unpriced_reasons,
        "n_todo_runs": n_todo,
        "n_unpriced_runs": n_unpriced,
        "n_gpt56_priced_runs": n_gpt56_priced,
        "as_of": table.as_of,
    }
    return out, warnings


def warning_lines(warnings: dict, table: PriceTable) -> list[str]:
    """Render `warnings` as plain, loud, jargon-free lines for a terminal report.

    Empty list only when there is truly nothing to say (no todo runs, no
    unpriced runs and no GPT-5.6 runs); the as-of line always prints. This
    function reports on the runs that were priced, not on the table itself:
    the table's own schema, staleness and todo audit belong to
    `validate_prices`, and nothing here computes or judges the table's age.
    """
    lines: list[str] = []

    def run_count(n: int) -> str:
        return f"{n} {'run' if n == 1 else 'runs'}"

    if warnings.get("n_todo_runs"):
        todo_models = warnings["todo_models"]
        parts = ", ".join(f"{model}: {n}" for model, n in sorted(todo_models.items()))
        # Every model entry in prices.toml carries a free-text `source` label
        # explaining its guess, and a reader is entitled to it. But that text
        # is authored data this module does not control, so it is never
        # echoed verbatim into terminal output: quoting it here would bypass
        # the jargon filter this module otherwise enforces on every line it
        # prints. Point at the field by name instead of inlining its content.
        n_sources = len({s for s in (table.source_for(m) for m in todo_models) if s})
        source_note = (
            " See the 'source' field per model in prices.toml for how each was derived."
            if n_sources
            else ""
        )
        lines.append(
            f"WARNING: {run_count(warnings['n_todo_runs'])} priced from placeholder rates "
            f"({parts}). These are best public-price guesses, not confirmed rates."
            f"{source_note}"
        )

    if warnings.get("n_unpriced_runs"):
        n_unpriced = warnings["n_unpriced_runs"]
        parts = ", ".join(
            f"{model}: {n}" for model, n in sorted(warnings["unpriced_models"].items())
        )
        exclusion = (
            "It is excluded from every dollar figure, never counted as zero."
            if n_unpriced == 1
            else "They are excluded from every dollar figure, never counted as zero."
        )
        lines.append(
            f"WARNING: {run_count(n_unpriced)} could not be priced at all "
            f"({parts}). {exclusion}"
        )
        # The line above already names every run whose model has no price entry,
        # by model. Repeating those same runs keyed by reason would make a reader
        # add two numbers that describe one set. Only reasons the model-keyed
        # line cannot express (a malformed or absent token block) go here.
        extra = {
            reason: n
            for reason, n in warnings.get("unpriced_reasons", {}).items()
            if not str(reason).startswith("no price entry for model")
        }
        if extra:
            reason_parts = ", ".join(f"{r}: {n}" for r, n in sorted(extra.items()))
            plural_subject = "those runs"
            reason_subject = (
                "That run failed for a reason other"
                if n_unpriced == 1
                else f"{sum(extra.values())} of {plural_subject} failed for a reason other"
            )
            lines.append(
                f"WARNING: {reason_subject} than a missing price entry ({reason_parts})."
            )

    if warnings.get("n_gpt56_priced_runs"):
        n = warnings["n_gpt56_priced_runs"]
        plural = "s" if n != 1 else ""
        lines.append(
            f"NOTE: {n} priced GPT-5.6 run{plural} may include long-context requests "
            "(roughly 272k tokens or more) that carry a surcharge no per-attempt record can "
            "show. Those requests are priced at the standard rate here, so any GPT-5.6 "
            "dollar figure above is a floor, not an exact bill."
        )

    # The as-of date is a fact the table carries and the report must show.
    # How OLD that makes the table, and whether that age is a problem, is
    # `validate_prices`'s job, so nothing here computes or judges an age.
    as_of = warnings.get("as_of", table.as_of)
    lines.append(f"Price table as of {as_of}.")

    return lines


def ratio_pair(a_tokens: dict, b_tokens: dict, a_usd: float, b_usd: float) -> dict:
    """Token ratio and dollar ratio side by side (the amplifier rule, SPEC 5).

    3x tokens on a cheaper model can be ~10x dollars on an expensive one;
    `amplifier` is how much the dollar gap outruns the token gap. The token
    total sums all four streams in `_TOKEN_STREAMS`; a caller that only has
    an aggregate figure may pass it under one key (commonly `in`) and leave
    the rest at 0, since a missing key here defaults to 0 rather than raising.
    Either ratio is None (not raised, not zero) when its denominator is zero,
    since a divide-by-zero here is a data condition, not a program error.
    """
    a_total = sum(float(a_tokens.get(s, 0)) for s in _TOKEN_STREAMS)
    b_total = sum(float(b_tokens.get(s, 0)) for s in _TOKEN_STREAMS)

    token_ratio = (a_total / b_total) if b_total else None
    dollar_ratio = (a_usd / b_usd) if b_usd else None

    amplifier = None
    if dollar_ratio is not None and token_ratio:
        amplifier = dollar_ratio / token_ratio

    return {
        "token_ratio": token_ratio,
        "dollar_ratio": dollar_ratio,
        "amplifier": amplifier,
    }


from . import price_validation as _price_validation

if _PRICE_MODULE_RELOAD:
    _price_validation = _importlib.reload(_price_validation)
_PRICE_MODULE_READY = True
_price_validation._synchronize_from_price(globals())
