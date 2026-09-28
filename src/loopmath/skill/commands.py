"""Handlers for `loopmath doctor` and `loopmath skill install|uninstall|show`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import __version__
from ..output import EXIT_OK, EXIT_USER, emit_json, fail, home
from . import doctor as _doctor
from . import install as _install

_MARK = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}


def doctor(args: argparse.Namespace) -> int:
    report = _doctor.run_checks(home(args.home), progress=lambda line: print(line, file=sys.stderr, flush=True))
    if args.json:
        emit_json("loopmath.doctor/1", {"version": __version__, **report})
    else:
        print(f"loopmath {__version__} doctor, store {report['home']}")
        for check in report["checks"]:
            print(f"{_MARK[check['status']]}  {check['name']:<7} {check['summary']}")
        if report["cache"]["bytes"]:
            from ..onboard.history import size_text

            print(f"doctor kept the log parse cache ({size_text(report['cache']['bytes'])} in "
                  f"{report['cache']['path']}) so the next read is fast; it changed nothing else in the store")
    return EXIT_OK if report["ok"] else EXIT_USER


# What a dry run says for each action install would take.
_WOULD = {"created": "to write", "updated": "to update", "replaced": "to replace", "appended": "to add",
          "removed": "to remove", "unchanged": "unchanged", "kept": "kept"}


def _count(entries: list[dict], dry_run: bool = False) -> str:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["action"]] = counts.get(e["action"], 0) + 1
    return ", ".join(f"{n} {_WOULD[action] if dry_run else action}" for action, n in counts.items())


def _results(verb: str, results: list[dict], as_json: bool) -> None:
    if as_json:
        emit_json("loopmath.skill/2", {"verb": verb, "results": results})
        return
    dry = any(r.get("dry_run") for r in results)
    if dry:
        print("dry run: nothing written. loopmath skill install (same options, without --dry-run) would do this:")
    for r in results:
        form = " (AGENTS.md block)" if r["method"] == "agents_block" else ""
        head = f"{r['target']} {r['scope']}{form}"
        actions = {e["action"] for e in r["skills"] + r["reference"]}
        if verb == "uninstall" and actions <= {"absent"}:
            print(f"{head}: not installed in {r['root']}, nothing to remove")
        elif verb == "uninstall" and actions <= {"removed", "absent"}:
            n = sum(e["action"] == "removed" for e in r["skills"])
            print(f"{head}: removed {n} skill{'s' if n != 1 else ''} and the reference.md beside each from {r['root']}")
        else:
            print(f"{head}: {len(r['skills'])} skills in {r['root']}: {_count(r['skills'], dry)}")
        if dry:
            print(f"  and in each skill folder a copy of the shared reference.md ({_count(r['reference'], dry)}), "
                  f"and {r['manifest']}, the list of the files install wrote")
        for e in r["removed"]:
            print(f"  {'to remove' if dry else 'removed'} {e['path']}")
        block, where = r.get("block"), r.get("block_file") or r.get("agents_md")
        if block == "removed":
            print(f"  {'to remove' if dry else 'removed'} the loopmath block from {where}")
        elif block and block != "absent":
            print(f"  AGENTS.md block {_WOULD[block] if dry else block} in {where}")
        for note in r.get("notes") or []:
            print(f"  note: {note}")


def _where(args: argparse.Namespace) -> Path | None:
    if args.dir and args.scope != "project":
        raise ValueError("--dir is the project folder for --scope project")
    return Path(args.dir).expanduser().resolve() if args.dir else None


def install(args: argparse.Namespace) -> int:
    try:
        results = _install.install(args.target, args.scope, cwd=_where(args), dry_run=getattr(args, "dry_run", False))
    except (OSError, ValueError) as exc:
        return fail(str(exc))
    _results("install", results, args.json)
    return EXIT_OK


def uninstall(args: argparse.Namespace) -> int:
    try:
        results = _install.uninstall(args.target, args.scope, cwd=_where(args))
    except (OSError, ValueError) as exc:
        return fail(str(exc))
    _results("uninstall", results, args.json)
    return EXIT_OK


def show(args: argparse.Namespace) -> int:
    name = getattr(args, "name", None) or "loopmath"
    try:
        sys.stdout.write(_install.skill_text(name))
    except KeyError:
        return fail(f"no skill {name!r}; one of {', '.join(_install.SKILLS)}, or reference")
    return EXIT_OK
