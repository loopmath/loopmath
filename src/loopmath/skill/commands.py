"""Handlers for `loopmath doctor` and `loopmath skill install|uninstall|show`.
(`loopmath guide` is guide.py's, which imports nothing else.)
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


def _results(verb: str, results: list[dict], as_json: bool) -> None:
    if as_json:
        emit_json("loopmath.skill/3", {"verb": verb, "results": results})
        return
    dry = any(r.get("dry_run") for r in results)
    if dry:
        print("dry run: nothing written. loopmath skill install (same options, without --dry-run) would do this:")
    for r in results:
        form = " (AGENTS.md block)" if r["method"] == "agents_block" else ""
        head = f"{r['target']} {r['scope']}{form}"
        [skill] = r["skills"]
        if verb == "uninstall" and skill["action"] == "absent" and not r["removed"]:
            print(f"{head}: not installed in {r['root']}, nothing to remove")
            continue
        print(f"{head}:")
        if skill["action"] != "absent":
            print(f"  {_WOULD[skill['action']] if dry else skill['action']} {skill['path']}, the loopmath skill")
        action = r.get("manifest_action")
        if dry and action:
            print(f"  {_WOULD[action]} {r['manifest']}, the list of the files install wrote")
        for e in r["removed"]:
            print(f"  {'to remove' if dry else 'removed'} {e['path']}")
        if any(e["path"].endswith(f"/{name}/SKILL.md") for e in r["removed"] for name in _install.RETIRED_SKILLS):
            print("  (0.3.4 installs one skill; the job skills' text comes from `loopmath guide` now)")
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
        topic = "reference" if name == "reference" else _install.RETIRED_SKILLS.get(name)
        if topic:
            return fail(f"0.3.4 has one skill, loopmath; {name} is `loopmath guide {topic}` now")
        return fail(f"no skill {name!r}; the one skill is loopmath, and `loopmath guide` lists the job guides")
    return EXIT_OK
