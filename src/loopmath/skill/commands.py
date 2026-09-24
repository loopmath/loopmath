"""Handlers for `loopmath doctor` and `loopmath skill install|uninstall|show`.

Owner: lane 09. Spec: design/0.1/02-commands.md section 5, 07-skill.md.
"""

from __future__ import annotations

import argparse
import sys

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


def _results(verb: str, results: list[dict], as_json: bool) -> None:
    if as_json:
        emit_json("loopmath.skill/1", {"verb": verb, "results": results})
        return
    for r in results:
        line = f"{r['target']} {r['scope']}: {r['action']} {r['path']}"
        if r.get("block"):
            line += f"; AGENTS.md block {r['block']} in {r['agents_md']}"
        print(line)


def install(args: argparse.Namespace) -> int:
    try:
        results = _install.install(args.target, args.scope)
    except (OSError, ValueError) as exc:
        return fail(str(exc))
    _results("install", results, args.json)
    return EXIT_OK


def uninstall(args: argparse.Namespace) -> int:
    try:
        results = _install.uninstall(args.target, args.scope)
    except (OSError, ValueError) as exc:
        return fail(str(exc))
    _results("uninstall", results, args.json)
    return EXIT_OK


def show(args: argparse.Namespace) -> int:
    sys.stdout.write(_install.skill_text())
    return EXIT_OK
