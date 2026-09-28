"""Handlers for `loopmath share` and `loopmath prior import-shared` (spec 02 section 4, spec 03 section 7).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

from ..belief import outcome
from ..output import EXIT_NOT_FOUND, EXIT_OK, EXIT_USER, emit_json, fail, home
from ..store.ids import parse_since
from .export import SCHEMA, build_share, dumps, is_plain, load_salt, org_hash, salt_path, store_docs, store_org, write_share
from .import_ import ORG_PREFIX, check_share, import_share, ocp_problems, priors_dir, priors_path, read_share

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
    except ValueError as exc:  # lane 7's one --since reader: `3m` is ambiguous and exits 2
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
    look = f"less {out}" if is_plain(out) else f"gunzip -c {out}"
    print(f"Review it before you send it: {look}   (or: loopmath share --preview)")
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
    root = home(args.home)
    if obj["org_hash"] == _own_org_hash(root):
        return fail(f"{path} was written by `loopmath share` from this store (organization {obj['org_hash']}); "
                    "its runs are already yours, so importing it would count each of them twice. Nothing imported")
    try:
        result = import_share(obj, root)
    except (OSError, ValueError) as exc:
        return fail(str(exc))
    if args.json:
        emit_json(IMPORT_SCHEMA, result)
        return EXIT_OK
    print(f"imported {len(obj['runs'])} runs as organization {result['org']}: {result['added']} new, "
          f"{result['updated']} updated, {result['unchanged']} unchanged; {result['total']} in the group")
    print(f"stored in {result['path']}. Run `loopmath fit` to use them; `loopmath prior remove-shared "
          f"{result['org']}` takes them out again.")
    return EXIT_OK


def _own_org_hash(root: Path) -> str | None:
    """The organization hash this store's own share files carry, or None before its first share
    (reading the salt must not create one)."""
    if not salt_path(root).is_file():
        return None
    try:
        return org_hash(load_salt(root), store_org(root))
    except (OSError, ValueError):
        return None


def remove_shared(args: argparse.Namespace) -> int:
    """Take an imported organization out of the store's priors: `prior remove-shared ORG`."""
    root = home(args.home)
    org = args.org[len(ORG_PREFIX):] if args.org.startswith(ORG_PREFIX) else args.org
    imported = sorted(p.name[len("shared-"):-len(".json.gz")] for p in priors_dir(root).glob("shared-*.json.gz"))
    path = priors_path(root, org) if org in imported else None
    if path is None:
        have = ", ".join(ORG_PREFIX + o for o in imported) if imported else "none"
        return fail(f"no imported organization {args.org} in {root}; imported: {have}", EXIT_NOT_FOUND)
    try:
        runs = len(read_share(path).get("runs") or [])
    except (OSError, ValueError):
        runs = None
    path.unlink()
    result = {"org": ORG_PREFIX + org, "path": str(path), "runs": runs}
    if args.json:
        emit_json("loopmath.prior.remove-shared/1", result)
        return EXIT_OK
    count = "its runs" if runs is None else f"its {runs} runs"
    print(f"removed organization {ORG_PREFIX + org} and {count} from the priors. Run `loopmath fit` to refit without them.")
    return EXIT_OK
