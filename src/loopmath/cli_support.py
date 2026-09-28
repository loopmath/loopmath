"""Stateless command and graph-accounting helpers for the CLI."""

from __future__ import annotations

import argparse
import re
import sys


def _scan_snapshot_id(discovered, *, limit: int | None = None) -> str:
    """Hash one frozen discovery manifest; touching a file changes the next id."""
    import hashlib
    import json

    files = [
        (harness, str(path), mtime_ns, size)
        for harness, path, mtime_ns, size in discovered
    ]
    payload = json.dumps(
        {"files": files, "limit": limit},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _graph_format(value: str) -> str:
    return value


def scan_verb(args: argparse.Namespace) -> int:
    print(
        "Scanning is implicit in loopmath graph and loopmath analyze; "
        "run either command to scan local logs."
    )
    return 0


def _extremes_ratio(df, result, price_mod) -> dict:
    """Token ratio and dollar ratio for the dearest against the cheapest.

    SPEC section 5's amplifier rule: reports must show both side by side,
    because a cheaper model burning 3x the tokens can still cost 10x the
    dollars.

    On success, returns the `price_mod.ratio_pair` dict plus `label` and
    `basis` (and `basis_note`, see below). On every failure path, returns
    `{"unavailable": "<reason>"}` instead of `None`: there are several
    different reasons this can fail (fewer than two ranked configurations, an
    extreme with no accepted runs, an extreme with no accepted run that has
    both a known token count and a known dollar cost together, or a
    non-positive dollar total), and only one of them is "only one
    configuration met the minimum run count". Collapsing them all into `None`
    let the report (terminal.py) print that one sentence regardless of which
    had actually happened, asserting a cause never measured; returning which
    reason fired here is what lets the report say only what was checked.

    Both ratios are computed PER ACCEPTED RUN, which is the same basis the
    table above ranks on. Per-run means were wrong here and printed a visibly
    silly line: a configuration can be the dearest per accepted run while
    using fewer tokens per run than the cheapest, purely because it gets
    accepted a fifth as often, so the dollar ratio came out above 1 while the
    token ratio came out below it. Dividing both by the same accepted count
    keeps the two halves of the amplifier comparable to each other and to the
    ranking that chose the pair.

    The tokens and dollars used here are the raw per-accepted sums, not the
    table's mix-standardized and pooled `cost_per_accepted`. That is
    deliberate: the amplifier is a statement about what was actually spent,
    and mixing an adjusted dollar figure with an unadjusted token figure would
    make the ratio of the two meaningless.

    FIX 1 (reviewer round, blocker): the token sum and the dollar sum for a
    configuration used to each skip their own missing values independently
    (`.sum(skipna=True)`) while the accepted-run denominator still counted
    EVERY accepted run in that configuration. A configuration with some
    unpriced or untokenized runs then got a numerator built from a subset and
    a denominator built from the whole, understating both ratios -- and the
    two halves could rest on DIFFERENT subsets (whichever rows happened to
    have a token count vs. whichever had a price), making their ratio (the
    amplifier itself) meaningless. Fixed by restricting each configuration to
    the rows where BOTH `total_tokens` and `usd` are known and computing the
    token sum, the dollar sum, AND the accepted-run count from that one same
    restricted set, so numerator and denominator always describe the same
    runs.

    The reviewer asked instead for the whole comparison to be withheld
    whenever a single value anywhere is missing. That is NOT what this does:
    the amplifier is the headline finding of the report, and dropping it
    entirely because one run out of sixty lacks a price would hide more than
    it protects. Disclosing the exact basis is the honest middle ground: when
    a configuration's both-known row count is smaller than its full
    known-outcome row count, `basis_note` on the returned dict says exactly
    how many runs the numbers rest on and why, for both configurations if
    they differ, and naming neither when nothing was dropped (measured
    against tonight's real data: nothing is, so this key is not set in the
    real run -- see the FIX 1 note above `drops` for what "dropped" means
    now and why it changed).
    """
    table = result.get("table")
    n_configs = 0 if table is None else len(table)
    if n_configs < 2:
        if n_configs == 0:
            return {
                "unavailable": (
                    "no workflow configuration met the minimum run count, so there "
                    "is nothing to compare"
                )
            }
        return {
            "unavailable": (
                "only one workflow configuration met the minimum run count, so "
                "there is nothing to compare"
            )
        }
    ordered = table.sort_values("cost_per_accepted")
    cheap = ordered.iloc[0]["arm"]
    dear = ordered.iloc[-1]["arm"]

    import math

    per: dict[str, dict] = {}
    # (arm, n_rows_used, n_rows_considered) for every configuration where the
    # both-known restriction actually dropped a ROW, in `(dear, cheap)`
    # order; feeds `basis_note` below.
    #
    # FIX 1 (final reviewer round, blocker): this used to compare ACCEPTED
    # counts (`n_acc` against `n_acc_all`) to decide whether anything was
    # dropped. But the numerator (`tokens`, `usd` below) is a sum over every
    # row in `sub` -- accepted AND rejected alike, per the comment on
    # `sub_all` -- while `n_acc`/`n_acc_all` only ever count accepted rows.
    # A REJECTED row that has a known acceptance outcome but is missing its
    # token count or dollar cost gets dropped from `sub` by the both-known
    # filter below, shrinking the numerator, yet it was never part of either
    # accepted count to begin with, so `n_acc == n_acc_all` stayed true and
    # `basis_note` never fired: the understated spend went unreported. Fixed
    # by comparing ROW counts (`len(sub)` against `len(sub_all)`), which is
    # the actual population the numerator sums over, and wording the note as
    # runs actually used out of runs considered rather than "accepted" runs.
    drops: list[tuple[str, int, int]] = []
    for arm in (dear, cheap):
        # `sub_all`: every row for this configuration with a KNOWN acceptance
        # outcome (accepted or rejected) -- the same population
        # `naive_arm_estimates` sums over (design section 3: total spend
        # across all known-outcome attempts, divided by the accepted count),
        # not just the accepted rows.
        sub_all = df[(df["arm"] == arm) & df["accepted"].notna()]
        n_rows_all = len(sub_all)
        n_acc_all = int(sub_all["accepted"].astype(bool).sum())
        if not n_acc_all:
            return {
                "unavailable": (
                    "the cheapest or most expensive configuration has no accepted "
                    "run, so a per-accepted comparison is not defined"
                )
            }

        # FIX 1: restrict to rows where BOTH streams are known, then take the
        # token sum, the dollar sum, AND the accepted count from this SAME
        # restricted set -- never a subset for one half and the whole set for
        # the other.
        both_known = sub_all["total_tokens"].notna() & sub_all["usd"].notna()
        sub = sub_all[both_known]
        n_rows = len(sub)
        n_acc = int(sub["accepted"].astype(bool).sum())
        if not n_acc:
            # Distinguish the cause when it is cleanly attributable to one
            # side; only fall back to the joint sentence when neither side is
            # wholly unknown (each has SOME known rows, just never together
            # on the same accepted run) -- never assert a cause not measured.
            if sub_all["total_tokens"].isna().all():
                return {
                    "unavailable": (
                        "the token total for one of the two configurations is "
                        "unknown, so the comparison would be guesswork"
                    )
                }
            if sub_all["usd"].isna().all():
                return {
                    "unavailable": (
                        "the dollar total for one of the two configurations is "
                        "unknown, so the comparison would be guesswork"
                    )
                }
            return {
                "unavailable": (
                    "no accepted run in one of the two configurations has both a "
                    "known token count and a known dollar cost, so the comparison "
                    "would be guesswork"
                )
            }
        if n_rows < n_rows_all:
            drops.append((arm, n_rows, n_rows_all))

        tokens = float(sub["total_tokens"].sum())
        usd = float(sub["usd"].sum())
        if math.isnan(usd) or usd <= 0:
            return {
                "unavailable": (
                    "the dollar total for one of the two configurations is unknown, "
                    "so the comparison would be guesswork"
                )
            }
        per[arm] = {"tokens": tokens / n_acc, "usd": usd / n_acc, "n_accepted": n_acc}

    # `ratio_pair(a, b)` reports a relative to b, so the dearer configuration
    # goes first and both ratios read as "the dear one costs Nx".
    # `ratio_pair` only needs the totals, and it sums whatever streams it is
    # given, so the whole per-accepted token figure goes in one stream and the
    # other three are zero. Four keys, per SPEC amendment 1.
    def _as_streams(total: float) -> dict:
        return {"in": total, "cache_read": 0, "cache_write": 0, "out": 0}

    pair = price_mod.ratio_pair(
        _as_streams(per[dear]["tokens"]),
        _as_streams(per[cheap]["tokens"]),
        per[dear]["usd"],
        per[cheap]["usd"],
    )
    pair["label"] = f"{dear} vs {cheap}"
    pair["basis"] = "per accepted run"
    if drops:
        # FIX 1: name exactly what was dropped rather than staying silent
        # about it; report both configurations' counts when they differ, one
        # when only one configuration lost any rows. Worded as runs actually
        # used out of runs considered (row counts), not accepted counts --
        # see the FIX 1 note above `drops` for why that distinction matters.
        pieces = [f"{n_used} of {n_considered} runs considered ({a})" for a, n_used, n_considered in drops]
        pair["basis_note"] = (
            "computed on " + " and ".join(pieces) + " where both the token count "
            "and the dollar cost are known"
        )
    return pair


def _jsonable(value):
    """Plain JSON values from the surface: frames as row lists, numpy scalars as numbers, NaN as null."""
    import math

    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item) and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def analyze_payload(surface: dict, *, diag: dict, coverage: dict, price_lines: list[str],
                    grading: list[str] | None, walkdown: list[dict]) -> dict:
    """`analyze --json`: the numbers the text report prints, as one object."""
    steps = {s["step"]: s for s in walkdown}
    spread = {"naive": surface.get("S_naive"), "matched": surface.get("S_matched"),
              "pooled": surface.get("S_honest")}
    band = None  # a band belongs to a pooled spread: none when that spread is n/a
    if all(isinstance(surface.get(k), (int, float)) and surface.get(k) == surface.get(k)
           for k in ("S_honest", "S_lo", "S_hi")):
        band = {"ci": surface.get("ci"), "lo": surface.get("S_lo"), "hi": surface.get("S_hi")}
    payload = {
        "cost": surface.get("cost_col"),
        "min_n": surface.get("min_n"),
        "read": {k: diag.get(k) for k in ("logs", "since_days", "files_seen", "files_outside_window", "skipped",
                                          "skip_reasons", "cache_hits", "parse_seconds", "ocp_documents",
                                          "ocp_attempts") if k in diag},
        "coverage": coverage,
        "runs": surface.get("n_rows"),
        "runs_used": surface.get("n_rows_used"),
        "configurations": surface.get("table"),
        "spread": {**spread, "band": band, "notes": {k: v.get("note") for k, v in steps.items()}},
        "task_mix_matched": not surface.get("overlap_caveat"),
        "cheapest_vs_dearest": surface.get("ratio_pair"),
        "excluded": [{k: e.get(k) for k in ("reason", "n", "detail", "n_censored")}
                     for e in surface.get("exclusions") or []],
        "price_warnings": list(price_lines),
    }
    if grading is not None:
        payload["grading"] = grading
    return _jsonable(payload)


def prices_verb(args: argparse.Namespace) -> int:
    """Print the active price table: models, stream rates, as-of date, provenance."""
    from . import price as price_mod

    try:
        table = price_mod.load_prices(args.prices)
    except FileNotFoundError:
        shown = args.prices if args.prices is not None else str(price_mod.DEFAULT_PRICES)
        print(f"error: price table not found: {shown}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"price table as of {table.as_of} ({table.path})")
    print()
    for model in sorted(table.rates):
        rates = table.rates[model]
        flag = " (rate not confirmed: a best public-price guess)" if model in table.todo else ""
        print(
            f"  {model}{flag}\n"
            f"    input={rates['input']:.4f}  cache_read={rates['cache_read']:.4f}  "
            f"cache_write={rates['cache_write']:.4f}  output={rates['output']:.4f}  ($/Mtok)"
        )
        src = _public_source(table.source.get(model) or "")
        if src:
            print(f"    source: {src}")
    if table.todo:
        print()
        print(f"{len(table.todo)} of {len(table.rates)} rates are not confirmed; every dollar figure priced "
              "with one says so")
    return 0


# Build notes in the packaged table's `source` text that mean nothing to a user. The file itself stays as it
# is: its bytes are the tariff id the prior bundle and every priced attempt name.
_NOTE_PARENS = re.compile(r"\s*\((?:needed for [^)]*|[^)]*\bAnalyst\b[^)]*)\)")
_NOTE_ADDED = re.compile(r",?\s*added \d{4}-\d{2}-\d{2}")
_NOTE_WORDS = re.compile(r"\bAnalyst\b|\b[DE]\d+\b")


def _public_source(text: str) -> str:
    """A price row's `source` as `loopmath prices` shows it: the provenance, without build notes (who added
    a row and for which experiment); a sentence that still names one is left out."""
    text = _NOTE_ADDED.sub("", _NOTE_PARENS.sub("", text)).strip()
    kept = [s for s in re.split(r"(?<=\.)\s+", text) if s and not _NOTE_WORDS.search(s)]
    return " ".join(kept).strip()


def _sweep_error(sweep_dir, args: argparse.Namespace | None = None) -> str | None:
    """Why the sweep folder cannot be fitted (no folder named, not a folder, no usable run
    records) or, given the verb's `args`, why its mask cannot (malformed, or no run to observe);
    checked before any sampling and before run_fit prints its data notes."""
    from . import research_paths
    from .fit_assembly import assemble_table

    try:
        root = research_paths.sweep_dir(sweep_dir)
    except research_paths.ResearchPathError as exc:
        return str(exc)
    if not root.is_dir():
        return f"the sweep folder {root} does not exist"
    df = assemble_table(sweep_dir=root)
    if df.empty:
        seen = df.attrs.get("assembly", {}).get("files_seen", 0)
        return (f"no sweep run records in {root} ({seen} file(s) read, none usable): the research verbs "
                "read the result folders of a model sweep, which are not part of the package")
    if args is not None:
        from .fit_masks import apply_mask, parse_mask

        try:  # the mask is the flags' to get right, so its ValueError is the user's to fix
            observed, _ = apply_mask(df, parse_mask(args.observe, holdout=args.holdout, reveal=args.reveal))
        except ValueError as exc:
            return str(exc)
        if observed.empty:
            flags = ", ".join(f"--{name} {value!r}" for name in ("observe", "holdout", "reveal")
                              if (value := getattr(args, name)) is not None)
            return (f"the mask selected 0 of {len(df)} runs to observe ({flags}); check it against the models "
                    "and efforts in this sweep")
    return None


def fit_verb(args: argparse.Namespace) -> int:
    from . import fit as fit_mod
    from .research_paths import ResearchPathError

    fit_mod.require_bayes()
    why = _sweep_error(args.sweep_dir, args)
    if why:
        print(f"error: {why}", file=sys.stderr)
        return 1
    try:
        result = fit_mod.run_fit(
            sweep_dir=args.sweep_dir,
            observe=args.observe,
            holdout=args.holdout,
            reveal=args.reveal,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            seed=args.seed,
            out=args.out,
        )
    except (RuntimeError, ResearchPathError) as exc:
        # run_fit raises RuntimeError only for conditions it has already
        # turned into an actionable sentence (a missing NetCDF backend, for
        # one). Print that sentence, not a traceback: a stack trace here tells
        # the reader nothing the message does not already say.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"assembled {result['n_rows']} runs from {result['sweep_dir']}")
    print(f"observed cells: {result['n_observed']} runs, held out: {result['n_heldout']} runs")
    print(f"mask: {result['mask_spec']}")
    print(f"sampled {args.draws} draws, {args.tune} tuning, {args.chains} chain(s), seed {args.seed}")
    if args.out:
        print(f"wrote the fit to {args.out}")
    return 0


def transfer_test_verb(args: argparse.Namespace) -> int:
    from . import fit as fit_mod
    from . import scoring
    from .research_paths import ResearchPathError

    fit_mod.require_bayes()
    why = _sweep_error(args.sweep_dir, args)
    if why:
        print(f"error: {why}", file=sys.stderr)
        return 1
    try:
        result = fit_mod.run_fit(
            sweep_dir=args.sweep_dir,
            observe=args.observe,
            holdout=args.holdout,
            reveal=args.reveal,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            seed=args.seed,
            out=None,
        )
    except ResearchPathError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    scores = fit_mod.run_transfer_test(result, ci=args.ci)
    print(f"scored {scores['n_heldout']} held-out runs against {scores['n_observed']} observed")
    for line in scoring.summarize(scores):
        print(f"  {line}")
    # SPEC section 7: descriptive only. The disclaimer is worded to match
    # `fit.TRANSFER_TEST_NOTE` so the two never drift apart, and it states
    # what the output IS rather than reaching for verdict words to deny.
    print(
        "\nThese numbers describe how the estimate transferred: "
        f"{fit_mod.TRANSFER_TEST_NOTE}."
    )
    return 0
def _slug(s: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s)).strip("_") or "unknown"


def _stage_number(stage: dict | None, key: str, empty_reason: str) -> tuple[int | None, str | None]:
    """(value, reason): the stage's number for `key`, or `None` with the reason
    it is unknown. A number is only ever what the stage said; a stage that said
    nothing, or nothing for this key, yields `None`, never 0. `key` may be dotted (`tiers.ungraded`); a per-harness
    dict counts through its `total`, or the sum of its integer members."""
    if not stage:
        return None, empty_reason
    v: object = stage
    for part in key.split("."):
        if not isinstance(v, dict) or part not in v:
            return None, f"the stage reported no {key!r}"
        v = v[part]
    if isinstance(v, dict):
        if "total" in v:
            v = v["total"]
        else:
            members = [x for k, x in v.items() if isinstance(x, int) and not isinstance(x, bool)]
            if not members:
                return None, f"the stage's {key!r} carries no counts"
            v = sum(members)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None, f"the stage's {key!r} is {str(v)[:40]!r}, not a number"
    return int(v), None


def _pipeline_counters(diag: dict, coverage: dict, price_warnings: dict, *, since, limit, files_outside_window) -> dict:
    """Flatten what ingest, grading and pricing excluded or could not do into
    `Graph.meta` keys. Prefixed by stage so they never
    collide with the extractor's own counters. A number a stage did not report
    is `None` beside a `<key>_reason` string, never 0: zero only when the stage
    said zero."""
    diag = diag if isinstance(diag, dict) else {}
    coverage = coverage if isinstance(coverage, dict) else {}
    price_warnings = price_warnings if isinstance(price_warnings, dict) else {}

    c: dict = {"ingest_since_days": since, "ingest_limit": limit, "ingest_files_outside_window": files_outside_window}

    def put(name: str, stage: dict, key: str, empty_reason: str) -> None:
        value, reason = _stage_number(stage, key, empty_reason)
        c[name] = value
        if reason:
            c[f"{name}_reason"] = reason

    no_diag = "parse_all returned no diagnostics"
    put("ingest_files_seen", diag, "files_seen", no_diag)
    put("ingest_records", diag, "records", no_diag)
    put("ingest_files_skipped", diag, "skipped", no_diag)
    put("ingest_files_omitted_by_limit", diag, "files_omitted_by_limit", no_diag)
    put("ingest_zero_token_synthetic", diag, "zero_token_synthetic", no_diag)
    for reason, n in sorted((diag.get("skip_reasons") or {}).items()):
        put(f"ingest_skipped_{_slug(reason)}", {"n": n}, "n", no_diag)
    no_cov = "grade_all returned no coverage"
    put("grading_records_total", coverage, "n_total", no_cov)
    put("grading_records_graded", coverage, "n_graded", no_cov)
    put("grading_ungraded", coverage, "tiers.ungraded", no_cov)
    put("grading_unevaluable_heuristic", coverage, "unevaluable_heuristic", no_cov)
    put("grading_synthetic_excluded", coverage, "n_synthetic_excluded", no_cov)
    no_price = "price_all returned no warnings"
    put("pricing_unpriced_runs", price_warnings, "n_unpriced_runs", no_price)
    put("pricing_todo_priced_runs", price_warnings, "n_todo_runs", no_price)
    for reason, n in sorted((price_warnings.get("unpriced_reasons") or {}).items()):
        put(f"pricing_unpriced_{_slug(reason)}", {"n": n}, "n", no_price)
    for model, n in sorted((price_warnings.get("unpriced_models") or {}).items()):
        put(f"pricing_unpriced_model_{_slug(model)}", {"n": n}, "n", no_price)
    for model, n in sorted((price_warnings.get("todo_models") or {}).items()):
        put(f"pricing_todo_model_{_slug(model)}", {"n": n}, "n", no_price)
    return c
