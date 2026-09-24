"""Implementation helpers for the ``loopmath graph`` CLI verb."""

from __future__ import annotations

import argparse
import sys
import time

from .cli_support import _pipeline_counters, _scan_snapshot_id


GRAPH_FORMATS = ("ocp", "json", "dot", "run", "html")
# Graph.meta keys that are totals, not exclusions: printed in the headline, not the list.
_GRAPH_META_TOTALS = (
    "n_nodes",
    "n_artifacts",
    "n_artifacts_consumed",
    "usd_total",
    "ingest_files_seen",
    "ingest_records",
    "grading_records_graded",
    "grading_records_total",
)


def _worktree_root():
    """The git top level of the current directory (no network: `rev-parse`
    reads `.git` only), or None when the current directory is not inside a
    git worktree or git is not available."""
    import os
    import subprocess
    from pathlib import Path

    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return Path(r.stdout.strip()).resolve()


def _resolve_out(out: str):
    """`--out` resolved against the worktree root (the git top level of the
    current directory); anything outside it is refused, and so is any `--out`
    when the current directory is not inside a git worktree (EXTRACTOR-SPEC
    section 0, rule 3: no writes outside the worktree). Returns (path, None)
    or (None, error message); nothing is written on refusal."""
    from pathlib import Path

    root = _worktree_root()
    target = Path(out).expanduser().resolve()
    if root is None:
        return None, f"error: --out {out} refused: {Path.cwd()} is not inside a git worktree (git rev-parse --show-toplevel failed), so there is no worktree to write into; run from inside one or drop --out"
    try:
        target.relative_to(root)
    except ValueError:
        return None, f"error: --out {out} resolves to {target}, outside the worktree {root}; the extractor writes only inside it"
    return target, None


def _empty_graph_error(records: list[dict], workspaces: list[str]) -> str:
    """Explain why parsed native records produced no final graph nodes."""
    from collections import Counter

    wanted = set(workspaces)
    requested = ", ".join(repr(item) for item in workspaces)
    matched = [record for record in records if record.get("workspace") in wanted]
    if matched:
        return (
            f"error: graph is empty: {len(matched)} of {len(records)} parsed records "
            f"matched workspace selector(s) {requested}, but none had both a run_id "
            "and a session_path; each graph record needs both fields"
        )
    labels = Counter(
        str(record["workspace"]) for record in records if record.get("workspace")
    )
    if not labels:
        return (
            f"error: graph is empty: none of {len(records)} parsed records matched "
            f"workspace selector(s) {requested}; no parsed record has a workspace label"
        )
    ordered = sorted(labels.items())
    shown = ", ".join(f"{label} ({count})" for label, count in ordered[:8])
    if len(ordered) > 8:
        shown += f", and {len(ordered) - 8} more"
    return (
        f"error: graph is empty: none of {len(records)} parsed records matched workspace "
        f"selector(s) {requested}; available parsed workspace labels: {shown}. "
        "Use one of those labels with --workspace"
    )


def graph_verb(args: argparse.Namespace) -> int:
    """discover -> parse -> grade -> price -> extract the workflow graph -> emit.

    `--format ocp` is the extractor's native output (OCP v0.3, spec section 5);
    `json` is the internal `dagr_graph` form; `dot` is Graphviz; `run` is the
    herdr-dagr contract v3 run file built from the OCP document; `html` is the
    self-contained interactive visualizer. Everything the
    pipeline excluded (files skipped, cut by `--limit` or the time window,
    records ungraded or unpriced) and everything the extractor could not place
    is counted in the graph's meta, which every format carries, and printed on
    stderr (EXTRACTOR-SPEC section 0, rule 1). `--quiet` silences progress
    only; the counters always print.
    """
    import json

    from . import grade as grade_mod
    from . import ingest, price as price_mod
    from .graph import extract, sanitize, to_dot, to_ocp
    from .graph.ocp import EXT_KEY, RUN_EXT_KEY
    from .ingest.ocp import OCPError, from_ocp, merge_graphs, read_document

    t0 = time.time()
    since = None if args.all else args.since
    if args.logs is not None:
        since = None

    out_path = None
    if args.out:
        out_path, err = _resolve_out(args.out)
        if err:
            print(err, file=sys.stderr)
            return 2

    # The count rewrites itself with \r, so it only goes to a terminal; in a
    # log or a pipe it would run into the next line (dogfood, D87 item 4).
    live = not args.quiet and sys.stderr.isatty()

    def _progress(harness: str, i: int, n: int) -> None:
        if live and n:
            print(f"  parsing {harness}: {i}/{n}", end="\r", file=sys.stderr)

    graphs = []
    ocp_attempt_count = 0
    try:
        for path in args.ocp:
            document = read_document(path)
            graphs.append(from_ocp(document))
            attempts = document.get("attempts")
            if isinstance(attempts, list):
                ocp_attempt_count += len(attempts)
    except OCPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    records: list[dict] = []
    diag: dict = {}
    native_graph = None
    found = ()
    if args.workspace:
        found = ingest._discovery_manifest(
            ingest.discover(args.logs, since_days=since)
        )
        print(
            f"scan snapshot: {_scan_snapshot_id(found, limit=args.limit)}",
            file=sys.stderr,
        )
        records, diag = ingest.parse_all(
            args.logs,
            limit=args.limit,
            use_cache=not args.no_cache,
            progress=_progress,
            since_days=since,
            discovered=found,
        )
        if live:
            print(" " * 60, end="\r", file=sys.stderr)
        files_outside_window = 0
        if since is not None:
            # The time window drops files before parsing; count them like `analyze` does.
            in_window = len(found)
            every = sum(
                len(v) for v in ingest.discover(args.logs, since_days=None).values()
            )
            files_outside_window = max(0, every - in_window)
        records, coverage = grade_mod.grade_all(records)
        table = price_mod.load_prices(args.prices)
        records, price_warnings = price_mod.price_all(records, table)

        native_graph = extract(records, workspaces=list(args.workspace))
        native_graph.meta.update(
            _pipeline_counters(
                diag,
                coverage,
                price_warnings,
                since=since,
                limit=args.limit,
                files_outside_window=files_outside_window,
            )
        )
        if records and not native_graph.nodes:
            print(
                sanitize(_empty_graph_error(records, list(args.workspace))),
                file=sys.stderr,
            )
            return 1
        graphs.insert(0, native_graph)
    try:
        g = merge_graphs(graphs)
    except OCPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    emitter: dict = {}
    if args.format == "ocp":
        doc = to_ocp(g, producer=None, privacy=args.privacy)
        emitter = doc["ext"][EXT_KEY]["emitter"]
        text = json.dumps(doc, indent=1, sort_keys=False) + "\n"
    elif args.format == "run":
        from .graph.runfile import codex_thread_files, read_finish_markers, to_runfile

        doc = to_ocp(g, producer=None, privacy=args.privacy)
        # A session-level clean-finish marker in the harness log (read from the
        # last file of a resumed codex thread) plus the record's grading signals
        # decide done against settled_unverified; a session without either is
        # unknown, never clean.
        if native_graph is not None and any(
            n.harness == "codex" for n in native_graph.nodes
        ):
            thread_files = codex_thread_files(ingest._manifest_paths(found, "codex"))
        else:
            thread_files = {}
        finish = read_finish_markers(g, session_files=thread_files)
        signals = {
            str(r["run_id"]): r.get("_signals")
            for r in records
            if isinstance(r, dict) and r.get("run_id")
        }
        run_doc = to_runfile(g, ocp=doc, signals=signals, finish=finish)
        run_ext = run_doc["ext"][RUN_EXT_KEY]
        emitter = {
            **doc["ext"][EXT_KEY]["emitter"],
            **{f"run.{k}": v for k, v in run_ext["exporter"].items()},
        }
        for harness, slot in run_ext["harnesses_without_finish_marker"].items():
            emitter[f"run.sessions_without_finish_marker.{harness}"] = (
                f"{slot['sessions']} ({slot['reason']})"
            )
        text = json.dumps(run_doc, indent=1, sort_keys=False, ensure_ascii=False) + "\n"
    elif args.format == "html":
        from .graph.html_render import to_html

        text = to_html(g)
    elif args.format == "json":
        text = json.dumps(g.to_dict(), indent=1) + "\n"
    else:
        text = to_dot(g)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)

    ws = (
        ", ".join(args.workspace)
        if args.workspace
        else ", ".join(str(path) for path in args.ocp)
    )
    window = (
        "every log file"
        if since is None
        else f"log files modified in the last {since:g} days"
    )
    m = g.meta
    native_meta = native_graph.meta if native_graph is not None else {}

    def say(line: str) -> None:
        # Every summary line passes the emitter's vocabulary choke point (spec section 0, rule 4).
        print(sanitize(line), file=sys.stderr)

    def n(key: str) -> str:
        # A number the stage did not report prints as unknown, never as 0.
        v = m.get(key)
        return "unknown" if v is None else str(v)

    source_count = len(records) + ocp_attempt_count
    source_note = (
        f"{len(args.ocp)} OCP document(s)" if not args.workspace else window
    )
    say(
        f"graph of {ws}: {n('n_nodes')} nodes, {len(g.edges)} edges, "
        f"{n('n_artifacts')} artifacts from {source_count} records ({source_note})"
    )
    if native_graph is not None:

        def nn(key: str) -> str:
            value = native_meta.get(key)
            return "unknown" if value is None else str(value)

        say(
            f"  read {nn('ingest_files_seen')} session files: "
            f"{nn('ingest_files_skipped')} skipped, "
            f"{nn('ingest_files_omitted_by_limit')} omitted by --limit, "
            f"{nn('ingest_files_outside_window')} outside the time window"
        )
        say(
            f"  graded {nn('grading_records_graded')} of "
            f"{nn('grading_records_total')} records: {nn('grading_ungraded')} "
            f"ungraded, {nn('grading_synthetic_excluded')} zero-token synthetic "
            "sessions excluded"
        )
        say(
            f"  priced: {nn('pricing_unpriced_runs')} records unpriced, "
            f"{nn('pricing_todo_priced_runs')} priced from placeholder rates"
        )
    if args.ocp:
        say(
            f"  imported {len(args.ocp)} OCP document(s); "
            "costs and evidence tiers kept as supplied"
        )
    for key in sorted(m):
        value = m[key]
        if key.endswith("_reason"):
            continue
        if value is None and f"{key}_reason" in m:
            say(f"  {key}: unknown ({m[f'{key}_reason']})")
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value
            and key not in _GRAPH_META_TOTALS
        ):
            say(f"  {key}: {value}")
    for key in sorted(emitter):
        # The run exporter is a full honesty ledger: print every one of its
        # counters, including zero, so a collision class is never silent.
        if emitter[key] or (args.format == "run" and key.startswith("run.")):
            say(f"  emitter.{key}: {emitter[key]}")
    if out_path is not None:
        say(f"wrote {out_path} ({args.format}) in {time.time() - t0:.1f} s")
    return 0
