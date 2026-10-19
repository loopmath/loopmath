"""`loopmath adapt harness2 PATH` and `loopmath adapt harness-runs FILE` (docs/adapters.md, Run importers).

Both read rows, turn each into one OCP v0.3 run, and write `<run id>.ocp.json` files into `--out DIR` (or JSON
lines to stdout). Nothing is written until every row has made a run that passes strict OCP validation, so
a refusal leaves no partial output. `loopmath run import DIR --finish` then stores the runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..ocp.emit import validate_strict
from .ocp import NS, to_ocp
from .rows import SCHEMA, experiment_problem, read_numbered, task_type_problem

IMPORTERS = ("harness2", "harness-runs")
SUMMARY_SCHEMA = "loopmath.adapt.runs/1"


def _mtime(path: Path) -> str | None:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(int(stamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _where(row: dict[str, Any]) -> str:
    tool = row.get("tool") if isinstance(row.get("tool"), dict) else {}
    return str(tool.get("run_id") or row.get("id") or row.get("task_id"))


def _fail(args: argparse.Namespace, name: str, message: str, code: int, lines: tuple[str, ...] = (),
          **more: Any) -> int:
    """A refusal: `error: <message>` (and `lines`) on stderr, and under --json one JSON object on stdout."""
    print("\n".join([f"error: {message}", *lines]), file=sys.stderr)
    if getattr(args, "json", False):
        print(json.dumps({"schema": SUMMARY_SCHEMA, "ok": False, "importer": name, "error": message, "exit": code,
                          **more}, indent=1, ensure_ascii=False))
    return code


def _experiments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("experiment"):
            counts[row["experiment"]] = counts.get(row["experiment"], 0) + 1
    return [{"experiment": exp, "runs": n} for exp, n in sorted(counts.items())]


def _summary(name: str, fmt: str, rows: list[dict[str, Any]], docs: list[dict[str, Any]], out: Path | None,
             extra: dict[str, Any]) -> dict[str, Any]:
    arms: dict[str, int] = {}
    cost = {"recovered": 0, "opencode": 0, "priced": 0, "reported": 0, "unpriced": 0, "incomplete": 0, "none": 0}
    unpriced: dict[str | None, dict[str, Any]] = {}
    scored = failed = 0
    for row, doc in zip(rows, docs):
        run = doc["run"]
        arm = (run.get("labels") or {}).get("arm") or "none"
        arms[arm] = arms.get(arm, 0) + 1
        scored += any(s.get("value") is not None for s in run.get("signals") or [])
        att = doc["attempts"][0]
        failed += att.get("status") == "failed"
        marks = ((att.get("ext") or {}).get(NS) or {}).get("cost") or {}
        cost["recovered"] += bool(marks.get("recovered"))
        if marks.get("complete") is False:
            cost["incomplete"] += 1
        elif (att.get("cost") or {}).get("usd") is not None:
            source = marks.get("usd_source") or "reported"
            cost["priced" if source == "loopmath-prices" else "opencode" if source == "opencode" else "reported"] += 1
        elif marks.get("unpriced"):
            cost["unpriced"] += 1
            entry = unpriced.setdefault(row.get("model"), {"model": row.get("model"), "runs": 0,
                                                           "reason": marks["unpriced"]})
            entry["runs"] += 1
        else:
            cost["none"] += 1
    return {"schema": SUMMARY_SCHEMA, "ok": True, "importer": name, "format": fmt, "written": len(docs),
            "out": str(out) if out else None, "scored": scored, "unscored": len(docs) - scored, "failed": failed,
            "arms": dict(sorted(arms.items())), "cost": cost,
            "unpriced_models": sorted(unpriced.values(), key=lambda e: (-e["runs"], e["model"] or "")), **extra}


def _text(s: dict[str, Any]) -> str:
    where = f"to {s['out']}" if s["out"] else "to stdout"
    arms = ", ".join(f"{n} {a}" for a, n in s["arms"].items())
    c = s["cost"]
    lines = [f"wrote {s['written']} runs {where} ({s['format']}): {arms}",
             f"  scored {s['scored']}, unscored {s['unscored']}, failed {s['failed']}",
             f"  cost: {c['opencode']} from OpenCode, {c['priced']} priced from tokens, {c['reported']} reported, "
             f"{c['unpriced']} unpriced, {c['incomplete']} incomplete, {c['none']} none; {c['recovered']} recovered "
             f"from trajectories"]
    if s.get("unpriced_models"):
        named = ", ".join(f"{e['runs']} {e['model'] or 'with no model'} ({e['reason']})" for e in s["unpriced_models"])
        lines.append(f"  unpriced: {named}; loopmath prices shows the price table, and their dollars stay unknown")
    for exp in s.get("experiments") or []:
        if "bench" not in exp:
            lines.append(f"  experiment {exp['experiment']}: {exp['runs']} run{'s' if exp['runs'] != 1 else ''}")
            continue
        lines.append(f"  experiment {exp['experiment']}: {exp['bench']} {exp['namespace']}"
                     + ("" if exp.get("config") else " (no config.json)"))
    for skip in s.get("skipped") or []:
        lines.append(f"  skipped {skip['run_id']}: {skip['reason']}")
    for note in s.get("notes") or []:
        lines.append(f"  note: {note}")
    if s["out"]:
        lines.append(f"next: loopmath run import {s['out']} --finish")
    return "\n".join(lines)


def adapt_runs(args: argparse.Namespace) -> int:
    """The run importers' CLI. Exit 0 written, 1 refused (nothing written), 2 usage or a missing input."""
    name = args.name
    if not getattr(args, "path", None):
        return _fail(args, name, f"adapt {name} needs a PATH (the "
                     f"{'harness2 runs folder' if name == 'harness2' else 'rows file'})", 2)
    path = Path(args.path).expanduser()
    if not path.exists():
        return _fail(args, name, f"adapt {name}: {path}: no such {'folder' if name == 'harness2' else 'file'}", 2)
    if args.type is not None and (why := task_type_problem(args.type)):
        return _fail(args, name, f"adapt {name}: --type {why}", 2)
    if args.experiment is not None:
        why = experiment_problem(args.experiment)
        if why is None and name == "harness2":
            from ..adapters.harness2 import EXPERIMENT_LABEL

            if not EXPERIMENT_LABEL.match(args.experiment):
                why = f"{args.experiment!r}: letters, digits, '.', '_' or '-', at most 64"
        if why:
            return _fail(args, name, f"adapt {name}: --experiment {why}", 2)
    extra: dict[str, Any] = {"skipped": [], "notes": [], "experiments": []}
    try:
        if name == "harness2":
            from ..adapters import harness2 as H2

            got = H2.read(path, experiment=args.experiment)
            numbered, fmt, base_dir = [(None, r) for r in got.rows], got.format, None
            extra.update(skipped=got.skipped, notes=got.notes, experiments=got.experiments, counts=got.counts)
        else:
            numbered, fmt, base_dir = read_numbered(path), SCHEMA, path.parent
            if args.experiment:
                numbered = [(n, {**r, "experiment": r.get("experiment") or args.experiment}) for n, r in numbered]
            extra["experiments"] = _experiments([r for _, r in numbered])
        default = _mtime(path)
        docs = []
        for n, row in numbered:
            try:
                docs.append(to_ocp(row, base_dir=base_dir, accept_at=args.accept_at, observed_default=default,
                                   task_type=args.type, repo=args.repo))
            except (OSError, ValueError) as exc:
                raise ValueError(f"{path}:{n}: {exc}" if n is not None else f"row {_where(row)}: {exc}") from None
    except (OSError, ValueError) as exc:   # FormatError is a ValueError
        return _fail(args, name, f"adapt {name}: {exc}", 1)
    rows = [r for _, r in numbered]
    if not rows:
        if name == "harness-runs":
            return _fail(args, name, f"adapt {name}: {path} has no rows (it is empty or blank); nothing written", 1)
        skipped = extra["skipped"]
        return _fail(args, name, f"adapt {name}: no rollout under {path} makes a run"
                     + (f" ({len(skipped)} skipped, below)" if skipped else "") + "; nothing written", 1,
                     tuple(f"  skipped {k['run_id']}: {k['reason']}" for k in skipped), skipped=skipped)
    for row, doc in zip(rows, docs):
        errors = [f for f in validate_strict(doc) if f["level"] == "error"]
        if errors:
            first = errors[0]
            return _fail(args, name, f"adapt {name}: row {_where(row)} makes a run that is not valid OCP "
                         f"({len(errors)} error{'s' if len(errors) != 1 else ''}; {first['code']} {first['path']}: "
                         f"{first['message']}); nothing written", 1)
    ids = [d["run"]["id"] for d in docs]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        return _fail(args, name, f"adapt {name}: two rows make the same run ({', '.join(dup[:3])}); give each row "
                     "its own id", 1)
    out = Path(args.out).expanduser() if args.out else None
    if out is not None and out.exists() and not out.is_dir():
        return _fail(args, name, f"adapt {name}: --out {out} is a file; give a folder", 2)
    written = 0
    try:
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            for doc in docs:
                _write_atomic(out / f"{doc['run']['id']}.ocp.json", json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
                written += 1
        if args.rows:
            rows_path = Path(args.rows).expanduser()
            rows_path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(rows_path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    except OSError as exc:
        return _fail(args, name, f"adapt {name}: cannot write: {exc} ({written} of {len(docs)} run files written)", 1,
                     written=written)
    if out is None:
        for doc in docs:
            sys.stdout.write(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = _summary(name, fmt, rows, docs, out, extra)
    stream = sys.stdout if out is not None else sys.stderr
    print(json.dumps(summary, indent=1, ensure_ascii=False) if args.json else _text(summary), file=stream)
    return 0
