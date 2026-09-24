"""Handlers for `loopmath share` and `loopmath prior import-shared` (spec 02 section 4, spec 03 section 7).

Owner: lane 08.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

from ..belief import outcome
from ..output import EXIT_NOT_FOUND, EXIT_OK, EXIT_USER, emit_json, fail, home
from ..store.ids import parse_since
from .export import SCHEMA, build_share, dumps, load_salt, store_docs, store_org, write_share
from .import_ import check_share, import_share, ocp_problems, read_share

IMPORT_SCHEMA = "loopmath.prior.import-shared/1"


def share(args: argparse.Namespace) -> int:
    """Write the shareable part of the store to `--out`, or print it with `--preview`."""
    if not args.preview and not args.out:
        return fail("share needs --out FILE, or --preview to print what would leave")
    root = home(args.home)
    if not root.is_dir():  # a mistyped home must not become a new store with its own salt
        return fail(f"no loopmath store at {root}: pass --home PATH or set LOOPMATH_HOME "
                    "(loopmath onboard creates one)", EXIT_NOT_FOUND)
    now = _dt.datetime.now().astimezone()
    try:
        since = parse_since(args.since, now)
    except ValueError as exc:  # lane 7's one --since reader (D89, D109): `3m` is ambiguous and exits 2
        return fail(str(exc), getattr(exc, "exit_code", EXIT_USER))

    try:
        obj, skipped = build_share(store_docs(root), salt=load_salt(root), org=store_org(root),
                                   evidence_fn=outcome.outcome_evidence, since=since, now=now)
    except (OSError, ValueError) as exc:
        return fail(f"cannot read the store at {root}: {exc}")

    if args.preview:
        # Exactly the object --out would write, indented; nothing is written.
        sys.stdout.write(dumps(obj, indent=1) + "\n")
        return EXIT_OK

    try:
        out = write_share(obj, Path(args.out).expanduser())
    except OSError as exc:
        return fail(f"cannot write {args.out}: {exc}")
    summary = {"out": str(out), "org_hash": obj["org_hash"], "created_at": obj["created_at"],
               "loopmath_version": obj["loopmath_version"], "n_runs": len(obj["runs"]), "skipped": skipped}
    if args.json:
        emit_json(SCHEMA, summary)
        return EXIT_OK
    print(f"wrote {len(obj['runs'])} runs to {out} ({SCHEMA}, organization {obj['org_hash']})")
    if skipped:
        print("left out: " + ", ".join(f"{n} {why}" for why, n in sorted(skipped.items())))
    print("It holds task types, features, workflows, settings, tokens, dollars and outcomes;")
    print("repos, tasks and runs appear only as salted hashes. No titles, paths, commands, session ids or commits.")
    print(f"Review it before you send it: gunzip -c {out}   (or: loopmath share --preview)")
    return EXIT_OK


def import_shared(args: argparse.Namespace) -> int:
    """Check a shared file and add its runs to the store's priors as their own organization group."""
    path = Path(args.file).expanduser()
    if not path.is_file():
        return fail(f"no such file: {path}", EXIT_NOT_FOUND)
    try:
        obj = read_share(path)
    except (OSError, ValueError) as exc:
        return fail(f"{path}: {exc}")
    problems = check_share(obj) or ocp_problems(obj)
    if problems:
        print(f"error: {path} is not an importable {SCHEMA} file:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return EXIT_USER
    try:
        result = import_share(obj, home(args.home))
    except (OSError, ValueError) as exc:
        return fail(str(exc))
    if args.json:
        emit_json(IMPORT_SCHEMA, result)
        return EXIT_OK
    print(f"imported {len(obj['runs'])} runs as organization {result['org']}: {result['added']} new, "
          f"{result['updated']} updated, {result['unchanged']} unchanged; {result['total']} in the group")
    print(f"stored in {result['path']}. Run `loopmath fit` to use them.")
    return EXIT_OK
