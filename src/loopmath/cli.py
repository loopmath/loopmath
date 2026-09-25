"""loopmath CLI.

Verbs:
  analyze          end to end on local harness logs (light core, seconds)
  scan             explain where implicit scanning occurs
  adapt            emit OCP v0.2 from a registered ADE adapter
  validate-prices  check a price table's schema, staleness and todo flags
  analyze-e0       the frozen E0 corpus walkdown
  prices           show the active price table
  graph            the workflow graph of one or more workspaces (ocp, json, dot, run, html: self-contained interactive HTML viewer)
  research         E1 v2 cost model: research fit, research transfer-test (needs the [bayes] extra)
  verify-receipts  verify an E2 spend ledger and frozen preregistration

0.1.0 verbs (design/0.1/02-commands.md, registered in cli_registry.py):
  plan:    task-types  workflows  recommend  plan
  record:  run start|attempt|artifact|finish|import  outcome  budget  config
  learn:   fit  onboard  share
  view:    runs  posterior  status  report  doctor  skill  ocp

No verb in this CLI makes a network call. Verbs write only to the store
($LOOPMATH_HOME, default ~/.loopmath, which also holds the parse cache) and to
paths the user names.
"""

from __future__ import annotations

import argparse
import importlib.util as _importlib_util
import sys
import time
from types import ModuleType

from . import __version__
from . import cli_graph as _cli_graph
from . import cli_support as _cli_support
from .cli_support import (
    _extremes_ratio,
    _pipeline_counters,
    _slug,
    _stage_number,
    fit_verb,
    prices_verb,
    transfer_test_verb,
)
from .cli_graph import (
    GRAPH_FORMATS,
    _GRAPH_META_TOTALS,
    _resolve_out,
    _worktree_root,
    graph_verb,
)


_SUPPORT_REBINDINGS = frozenset({"sys", "_slug", "_stage_number"})
_GRAPH_REBINDINGS = frozenset(
    {
        "argparse",
        "sys",
        "time",
        "_pipeline_counters",
        "_GRAPH_META_TOTALS",
        "_resolve_out",
        "_worktree_root",
    }
)


class _CliModule(ModuleType):
    """Keep moved helpers responsive to cli-module rebinding."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name in _SUPPORT_REBINDINGS:
            setattr(_cli_support, name, value)
        if name in _GRAPH_REBINDINGS:
            setattr(_cli_graph, name, value)


sys.modules[__name__].__class__ = _CliModule

DEFAULT_SINCE_DAYS = 14.0


def _counted_noun(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def analyze(args: argparse.Namespace) -> int:
    """Read native logs and/or OCP, then build the cost surface and report."""
    from . import grade as grade_mod
    from . import ingest, price as price_mod, surface as surface_mod
    from .ingest.ocp import OCPError, coverage_from_records, read_document, records_from_ocp
    from .report import terminal

    t0 = time.time()
    ocp_paths = list(args.ocp or [])
    ocp_records: list[dict] = []
    try:
        for path in ocp_paths:
            ocp_records.extend(records_from_ocp(read_document(path)))
    except OCPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Supplying only --ocp is deliberately bounded and must not also scan the
    # machine's default log roots.  An explicit --logs opts into a mixed run.
    use_native = not ocp_paths or args.logs is not None
    since = None if args.all else args.since
    if args.logs is not None:
        # An explicit --logs path is a deliberate, bounded choice; do not also
        # narrow it by mtime behind the user's back.
        since = None

    def _progress(harness: str, i: int, n: int) -> None:
        if not args.quiet and n:
            print(f"  parsing {harness}: {i}/{n}", end="\r", file=sys.stderr)

    records: list[dict] = []
    diag: dict = {}
    coverage: dict = {}
    price_lines: list[str] = []
    native_coverage: dict = {}
    if use_native:
        found = ingest._discovery_manifest(
            ingest.discover(args.logs, since_days=since)
        )
        n_found = len(found)
        print(
            f"scan snapshot: {_cli_support._scan_snapshot_id(found, limit=args.limit)}",
            file=sys.stderr,
        )
        if not args.quiet:
            print(f"scanning {_counted_noun(n_found, 'session file')} ...", file=sys.stderr)

        records, diag = ingest.parse_all(
            args.logs,
            limit=args.limit,
            use_cache=not args.no_cache,
            progress=_progress,
            since_days=since,
            discovered=found,
        )
        if not args.quiet:
            print(" " * 60, end="\r", file=sys.stderr)
        diag["since_days"] = since
        if since is not None:
            # SPEC section 0: exclusions print, never hide. A default time window
            # silently drops most of the local estate, so count what it dropped and
            # let beat 1 say so. One extra directory listing, no extra parsing.
            every = ingest.discover(args.logs, since_days=None)
            diag["files_outside_window"] = max(0, sum(len(v) for v in every.values()) - n_found)
        diag["logs"] = str(args.logs) if args.logs else "default local log roots"
        diag["parse_seconds"] = round(time.time() - t0, 2)

        records, native_coverage = grade_mod.grade_all(records)
        table = price_mod.load_prices(args.prices)
        records, price_warnings = price_mod.price_all(records, table)
        price_lines.extend(price_mod.warning_lines(price_warnings, table))

    if ocp_paths:
        records.extend(ocp_records)
        diag["ocp_documents"] = len(ocp_paths)
        diag["ocp_attempts"] = len(ocp_records)
        coverage = coverage_from_records(records)
        if native_coverage:
            coverage["unevaluable_heuristic"] = native_coverage.get("unevaluable_heuristic", 0)
            coverage["n_synthetic_excluded"] = native_coverage.get("n_synthetic_excluded", 0)
            coverage["n_total_parsed"] = coverage["n_total"] + coverage["n_synthetic_excluded"]
        price_lines.append(
            f"OCP costs from {len(ocp_paths)} document(s) were "
            "preserved as supplied; "
            "loopmath did not re-price imported attempts."
        )
    else:
        coverage = native_coverage

    df = surface_mod.build_frame(records)
    result = surface_mod.cost_surface(
        df,
        cost_col=surface_mod.COST_TOKENS if args.tokens else surface_mod.COST_DOLLARS,
        min_n=args.min_n,
        n_boot=args.boot,
        seed=args.seed,
        ci=args.ci,
    )
    result["ratio_pair"] = _extremes_ratio(df, result, price_mod)

    print(
        terminal.render(
            ingest_diag=diag,
            coverage=coverage,
            coverage_line=grade_mod.coverage_line(coverage),
            surface=result,
            walkdown_line=surface_mod.walkdown_line(result),
            price_warnings=price_lines,
        )
    )

    if args.grading:
        print()
        for line in grade_mod.grading_report(records, coverage):
            print(line)

    if not args.quiet:
        # cache_hits counts session FILES served from the cache, and a file may
        # yield one record or none, so it must not be printed against the record
        # count: that produced lines like "4,389 of 4,387 runs from cache".
        if use_native:
            files_seen = (diag.get("files_seen") or {}).get("total", 0)
            print(
                f"\nran in {time.time() - t0:.1f} s "
                f"({diag.get('cache_hits', 0)} of "
                f"{_counted_noun(files_seen, 'session file')} from cache)",
                file=sys.stderr,
            )
        else:
            print(
                f"\nran in {time.time() - t0:.1f} s "
                f"({_counted_noun(len(ocp_records), 'attempt')} from "
                f"{len(ocp_paths)} OCP document(s))",
                file=sys.stderr,
            )
    return 0




def _add_prices(sub) -> None:
    p = sub.add_parser("prices", help="show the active price table")
    p.add_argument("--prices", default=None, help="price table to use instead of the packaged one")
    p.set_defaults(func=prices_verb)


def _add_scan(sub) -> None:
    p = sub.add_parser(
        "scan",
        help="explain the implicit scan used by graph and analyze",
        description="Scanning is implicit in loopmath graph and loopmath analyze.",
    )
    p.set_defaults(func=_cli_support.scan_verb)


def _add_graph(sub) -> None:
    p = sub.add_parser("graph", help="emit the workflow graph of one or more workspaces")
    p.add_argument("--workspace", action="append", default=[], help="native workspace name to include (repeatable; required unless --ocp is used)")
    p.add_argument("--ocp", action="append", default=[], metavar="FILE", help="read an OCP document (v0.2 or later) as a graph source (repeatable)")
    p.add_argument("--format", type=_cli_support._graph_format, choices=GRAPH_FORMATS, default="ocp", help="ocp (OCP v0.3, default), json (internal dagr_graph), dot (Graphviz), run (herdr-dagr contract v3 run file), html (self-contained interactive HTML viewer)")
    p.add_argument("--out", default=None, help="write here instead of stdout (must be inside the git worktree of the current directory; parent directories are created)")
    p.add_argument("--privacy", choices=("metadata_only", "full"), default="metadata_only", help="OCP privacy profile (default metadata_only)")
    p.add_argument("--logs", default=None, help="parse this path instead of the default log roots")
    p.add_argument(
        "--since",
        type=float,
        default=DEFAULT_SINCE_DAYS,
        help=f"only read log files modified in the last N days (default {DEFAULT_SINCE_DAYS:g})",
    )
    p.add_argument("--all", action="store_true", help="read every log file, ignoring --since")
    p.add_argument("--limit", type=int, default=None, help="stop after N files per harness")
    p.add_argument("--no-cache", action="store_true", help="reparse everything, ignoring the cache")
    p.add_argument("--prices", default=None, help="price table to use instead of the packaged one")
    p.add_argument("--quiet", action="store_true", help="suppress progress output on stderr; exclusion counters always print")
    p.set_defaults(func=graph_verb)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def adapt_verb(args: argparse.Namespace) -> int:
    """Run one registered ADE adapter and serialize its OCP v0.2 document."""
    import json
    from pathlib import Path

    from .adapters import Selection, UnknownAdapterError, lookup

    out_path = None
    if args.out:
        out_path, err = _resolve_out(args.out)
        if err:
            print(err, file=sys.stderr)
            return 2

    try:
        adapter_type = lookup(args.name)
    except UnknownAdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    selection = Selection(
        stores=tuple(Path(value).expanduser() for value in args.store),
        session_ids=tuple(args.session),
        workspaces=tuple(args.workspace),
        since=args.since,
        until=args.until,
        limit=args.limit,
    )
    from .adapter_host import default_adapter_services

    adapter = adapter_type()
    adapter.configure_services(default_adapter_services())
    doc = adapter.emit(selection)
    if not isinstance(doc, dict):
        raise TypeError(f"adapter {args.name!r} emit() must return a dict")
    from .ocp import version_at_least

    if not version_at_least(doc, "0.2"):
        raise ValueError(f"adapter {args.name!r} emitted OCP {doc.get('ocp')!r}, expected '0.2' or later")
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"

    if out_path is None:
        sys.stdout.write(text)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {out_path} (OCP v0.2)", file=sys.stderr)
    return 0


def _add_adapt(sub) -> None:
    p = sub.add_parser("adapt", help="emit OCP v0.2 from an ADE store (loopmath ocp migrate converts it to v0.3)")
    p.add_argument("name", metavar="NAME", help="registered adapter name")
    p.add_argument("--out", default=None, help="write here instead of stdout (must be inside the git worktree of the current directory; parent directories are created)")
    p.add_argument("--store", action="append", default=[], metavar="PATH", help="use this store path instead of discovery (repeatable)")
    p.add_argument("--session", action="append", default=[], metavar="ID", help="include this session id (repeatable; default: all)")
    p.add_argument("--workspace", action="append", default=[], metavar="WORKSPACE", help="include this workspace (repeatable; default: all)")
    p.add_argument("--since", default=None, metavar="RFC3339", help="include sessions at or after this timestamp")
    p.add_argument("--until", default=None, metavar="RFC3339", help="include sessions at or before this timestamp")
    p.add_argument("--limit", type=_positive_int, default=None, metavar="N", help="emit at most N selected sessions")
    p.set_defaults(func=adapt_verb)


def validate_prices_verb(args: argparse.Namespace) -> int:
    """Check a price table and say what is wrong with it (SPEC section 8, item 2).

    Exit status is the whole point of the verb: 0 when the table is valid
    (warnings and all), 1 when it carries a hard error. Warnings print and do
    not fail, because a stale or todo-heavy table still prices runs and a
    check that refuses to run on one would just get skipped.
    """
    from . import price as price_mod

    result = price_mod.validate_prices(args.prices)
    stream = sys.stdout if result["ok"] else sys.stderr
    for line in price_mod.validation_lines(result):
        print(line, file=stream)
    return 0 if result["ok"] else 1


def _add_analyze(sub) -> None:
    p = sub.add_parser("analyze", help="cost per accepted run from your local harness logs")
    p.add_argument("--logs", default=None, help="parse this path instead of the default log roots")
    p.add_argument("--ocp", action="append", default=[], metavar="FILE", help="analyze an OCP v0.2 document (repeatable; use with --logs to combine sources)")
    p.add_argument(
        "--since",
        type=float,
        default=DEFAULT_SINCE_DAYS,
        help=f"only read log files modified in the last N days (default {DEFAULT_SINCE_DAYS:g})",
    )
    p.add_argument("--all", action="store_true", help="read every log file, ignoring --since")
    p.add_argument("--limit", type=int, default=None, help="stop after N files per harness")
    p.add_argument("--no-cache", action="store_true", help="reparse everything, ignoring the cache")
    p.add_argument("--prices", default=None, help="price table to use instead of the packaged one")
    p.add_argument("--min-n", type=int, default=5, help="minimum runs per configuration")
    p.add_argument("--boot", type=int, default=1000, help="resampling draws for the bands")
    p.add_argument("--seed", type=int, default=20260831, help="resampling seed (recorded)")
    p.add_argument("--ci", type=float, default=0.80, help="confidence band width (default 0.80)")
    p.add_argument("--tokens", action="store_true", help="rank by output tokens instead of dollars")
    p.add_argument("--grading", action="store_true", help="print which rule graded what")
    p.add_argument("--quiet", action="store_true", help="suppress progress and timing on stderr")
    p.set_defaults(func=analyze)


def _add_validate_prices(sub) -> None:
    p = sub.add_parser(
        "validate-prices", help="check a price table's schema, staleness and todo flags"
    )
    p.add_argument("--prices", default=None, help="price table to check instead of the packaged one")
    p.set_defaults(func=validate_prices_verb)


def _add_fit(sub) -> None:
    from .research_defaults import E1A_OBSERVE, SEED

    research = sub.add_parser("research", help="E1 research verbs: fit, transfer-test (needs the [bayes] extra)")
    rsub = research.add_subparsers(dest="research_command", required=True)
    for name, fn, helptext in (
        ("fit", fit_verb, "fit the E1 cost model on the sweep data"),
        ("transfer-test", transfer_test_verb, "score held-out cells from a fit"),
    ):
        p = rsub.add_parser(name, help=helptext)
        p.add_argument("--sweep-dir", default=None, help="sweep run records directory (default: LOOPMATH_SWEEP_DIR, then research.sweep_dir in config)")
        p.add_argument("--observe", default=E1A_OBSERVE, help="observed cells, model:effort,effort;...")
        p.add_argument("--holdout", default="rest", help="held-out cells, or 'rest'")
        p.add_argument("--reveal", default=None, help="extra observed cells within a task group")
        p.add_argument("--draws", type=int, default=500, help="NUTS draws per chain (default 500)")
        p.add_argument("--tune", type=int, default=500, help="NUTS tuning steps per chain (default 500)")
        p.add_argument("--chains", type=int, default=2, help="MCMC chains (default 2)")
        p.add_argument("--seed", type=int, default=SEED, help="random seed")
        if name == "fit":
            p.add_argument("--out", default=None, help="write the fit here")
        else:
            p.add_argument("--ci", type=float, default=0.80, help="central interval for held-out coverage (default 0.80)")
        p.set_defaults(func=fn)


def verify_receipts_verb(args: argparse.Namespace) -> int:
    """Verify an E2 receipt ledger and print one machine-friendly result."""
    from .receipts import ReceiptViolation, verify_receipts

    try:
        result = verify_receipts(args.ledger, args.frozen)
    except ReceiptViolation as exc:
        print(f"FAIL: {exc}")
        return 1
    print(result.message())
    return 0


def _add_verify_receipts(sub) -> None:
    p = sub.add_parser("verify-receipts", help="verify an E2 receipt ledger")
    p.add_argument(
        "ledger",
        nargs="?",
        default="e2-receipts.jsonl",
        help="receipt ledger (default: e2-receipts.jsonl)",
    )
    p.add_argument(
        "--frozen",
        default=None,
        help="checksum file (default: FROZEN.txt beside the ledger)",
    )
    p.set_defaults(func=verify_receipts_verb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loopmath", description="loopmath: which agent workflow to run next, priced from your own logs"
    )
    parser.add_argument("--version", action="version", version=f"loopmath {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_analyze(sub)
    _add_scan(sub)
    _add_validate_prices(sub)
    _add_prices(sub)
    _add_graph(sub)
    _add_adapt(sub)
    _add_verify_receipts(sub)

    from . import cli_registry as _registry

    _registry.register(sub)

    if _importlib_util.find_spec("matplotlib") is not None:
        # The E0 walkdown verb draws figures, so running it needs matplotlib.
        # SPEC section 2's light-core rule says `analyze` runs on numpy and
        # pandas, so matplotlib lives in the `e0` extra and this verb is simply
        # absent without it. It registers from e0/verb.py without importing the
        # walkdown, so no other command pays for pandas and matplotlib.
        from .e0 import verb as _e0_verb

        _e0_verb.register(sub)

    try:
        _add_fit(sub)
    except ImportError:
        # fit.py needs the modelling extra only to RUN, not to import; if it
        # cannot be imported at all the two verbs are simply absent and
        # `loopmath --help` still works. Never crash the whole CLI over an extra.
        pass

    sub.metavar = _registry.command_metavar(sub)  # hidden commands stay callable, off the usage line

    raw_args = sys.argv[1:] if argv is None else argv
    if raw_args and raw_args[0] == "view":
        parser.error(
            "loopmath view is unavailable; use 'loopmath graph --format dot' with "
            "Graphviz, or 'loopmath graph --format run' with the separate "
            "herdr-dagr viewer"
        )
    args = parser.parse_args(raw_args)
    if args.command == "graph" and not args.workspace and not args.ocp:
        parser.error("loopmath graph requires --workspace or --ocp")
    from .ingest.base import use_home

    use_home(getattr(args, "home", None))  # the caches follow --home as the store does
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
