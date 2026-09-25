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
    return EXIT_OK if report["ok"] else EXIT_USER


def _count(entries: list[dict]) -> str:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["action"]] = counts.get(e["action"], 0) + 1
    return ", ".join(f"{n} {action}" for action, n in counts.items())


def _results(verb: str, results: list[dict], as_json: bool) -> None:
    if as_json:
        emit_json("loopmath.skill/2", {"verb": verb, "results": results})
        return
    for r in results:
        form = " (AGENTS.md block)" if r["method"] == "agents_block" else ""
        print(f"{r['target']} {r['scope']}{form}: {len(r['skills'])} skills in {r['root']}: {_count(r['skills'])}")
        for e in r["removed"]:
            print(f"  removed {e['path']}")
        if r.get("block"):
            print(f"  AGENTS.md block {r['block']} in {r['agents_md']}")
        for note in r.get("notes") or []:
            print(f"  note: {note}")


def _where(args: argparse.Namespace) -> Path | None:
    if args.dir and args.scope != "project":
        raise ValueError("--dir is the project folder for --scope project")
    return Path(args.dir).expanduser().resolve() if args.dir else None


def install(args: argparse.Namespace) -> int:
    try:
        results = _install.install(args.target, args.scope, cwd=_where(args))
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
