"""Handlers for `loopmath ocp validate|migrate` (spec 01 sections 3 and 4, spec 02 section 7).

Owner: lane 01.

- `ocp validate FILE...`: every rule for each file, plus the cross-file slate
  check (E194) over the set. Exit 0 when no file has an error, 1 when one
  does, 2 when a file is missing.
- `ocp migrate FILE... [--out DIR]`: v0.1, v0.2 or contract v3 to v0.3. With
  `--out` each result is written to DIR (never over an input); without it a
  single result goes to stdout. `--json` reports each file, and without
  `--out` carries the migrated documents. A result with a conformance error
  is reported and never written or printed (an existing output stays).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..output import EXIT_NOT_FOUND, EXIT_OK, EXIT_USER, emit_json, fail
from .conformance import validate_doc, validate_files
from .migrate import MigrationError, migrate_doc

FINDINGS_SHOWN = 8  # per file in the terminal summary; --json carries all


def _counts(findings: list[Any]) -> tuple[int, int]:
    errors = sum(f.level == "error" for f in findings)
    return errors, len(findings) - errors


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _tally(errors: int, warnings: int) -> str:
    return f"{_plural(errors, 'error')}, {_plural(warnings, 'warning')}"


def _finding_dicts(findings: list[Any]) -> list[dict[str, Any]]:
    return [f._asdict() for f in findings]


def _print_findings(findings: list[Any]) -> None:
    for f in findings[:FINDINGS_SHOWN]:
        print(f"  {f.level} {f.code} {f.path}: {f.message}")
    if len(findings) > FINDINGS_SHOWN:
        print(f"  ... {len(findings) - FINDINGS_SHOWN} more (use --json for all)")


def validate(args: argparse.Namespace) -> int:
    paths = [str(p) for p in args.files]
    missing = [p for p in paths if not Path(p).is_file()]
    results = validate_files(paths)
    rows, labels = [], []
    for path, findings in zip(paths, results):
        errors, warnings = _counts(findings)
        version, label = None, "not a file"
        if path not in missing:
            try:
                doc = json.loads(Path(path).read_text())
            except (OSError, ValueError):
                label = "not JSON"
            else:
                version = doc.get("ocp") if isinstance(doc, dict) else None
                label = "no ocp version" if version is None else f"ocp {version}"
        rows.append({"path": path, "ocp": version, "ok": errors == 0, "errors": errors, "warnings": warnings,
                     "findings": _finding_dicts(findings)})
        labels.append(label)
    ok = all(row["ok"] for row in rows)
    if args.json:
        emit_json("loopmath.ocp.validate/1", {"ok": ok, "files": rows})
    else:
        for row, label, findings in zip(rows, labels, results):
            verdict = "PASS" if row["ok"] else "FAIL"
            print(f"{verdict}  {row['path']}  ({label}; {_tally(row['errors'], row['warnings'])})")
            _print_findings(findings)
    if missing:
        return EXIT_NOT_FOUND
    return EXIT_OK if ok else EXIT_USER


def _print_rows(findings: list[dict[str, Any]], stream: Any) -> None:
    for f in findings[:FINDINGS_SHOWN]:
        print(f"  {f['level']} {f['code']} {f['path']}: {f['message']}", file=stream)
    if len(findings) > FINDINGS_SHOWN:
        print(f"  ... {len(findings) - FINDINGS_SHOWN} more (use --json for all)", file=stream)


def _out_name(path: Path) -> str:
    name = path.name
    if name.endswith(".ocp.json"):
        return name
    return (name[: -len(".json")] if name.endswith(".json") else name) + ".ocp.json"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def migrate(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.files]
    out_dir = Path(args.out).expanduser() if args.out else None
    if out_dir is None and len(paths) > 1 and not args.json:
        return fail("several files need --out DIR (or --json to get the migrated documents in one object)")
    inputs = {p.resolve() for p in paths}
    targets: dict[Path, Path] = {}
    if out_dir is not None:
        for path in paths:
            target = (out_dir / _out_name(path)).resolve()
            if target in inputs:
                return fail(f"--out would overwrite the input {path}; choose another folder")
            first = next((p for p, t in targets.items() if t == target), None)
            if first is not None:
                return fail(f"{first} and {path} would both be written to {target}; "
                            "migrate them into different --out folders")
            targets[path] = target

    rows, docs, missing = [], [], False
    for path in paths:
        row: dict[str, Any] = {"in": str(path), "out": None, "from": None, "ok": False}
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            missing = True
            row["error"] = "not found"
            rows.append(row)
            continue
        except (OSError, ValueError) as exc:
            row["error"] = f"cannot read JSON: {exc}"
            rows.append(row)
            continue
        row["from"] = doc.get("ocp", "contract") if isinstance(doc, dict) else None
        try:
            migrated = migrate_doc(doc)
        except (MigrationError, ValueError) as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        findings = validate_doc(migrated)
        errors, warnings = _counts(findings)
        row.update({"ok": errors == 0, "errors": errors, "warnings": warnings, "findings": _finding_dicts(findings)})
        rows.append(row)
        if errors:
            continue  # only conforming documents are OCP output
        text = json.dumps(migrated, indent=2, ensure_ascii=False) + "\n"
        if out_dir is not None:
            _write_atomic(targets[path], text)
            row["out"] = str(targets[path])
        elif args.json:
            row["doc"] = migrated
        docs.append(text)

    ok = all(row["ok"] for row in rows)
    if args.json:
        emit_json("loopmath.ocp.migrate/1", {"ok": ok, "files": rows})
    elif out_dir is None:
        if docs:
            sys.stdout.write(docs[0])
        for row in rows:
            if "error" in row:
                print(f"error: {row['in']}: {row['error']}", file=sys.stderr)
            elif row["errors"]:
                print(f"error: {row['in']}: the migrated document does not conform "
                      f"({_tally(row['errors'], row['warnings'])}); nothing written", file=sys.stderr)
                _print_rows(row["findings"], sys.stderr)
            elif row["warnings"]:
                print(f"{row['in']}: migrated from {row['from']} with {_plural(row['warnings'], 'warning')} "
                      "(run 'loopmath ocp validate' on the result)", file=sys.stderr)
    else:
        for row in rows:
            if "error" in row:
                print(f"FAIL  {row['in']}: {row['error']}")
                continue
            if not row["ok"]:
                print(f"FAIL  {row['in']}: the migrated document does not conform (from {row['from']}; "
                      f"{_tally(row['errors'], row['warnings'])}); nothing written")
                _print_rows(row["findings"], sys.stdout)
                continue
            print(f"PASS  {row['in']} -> {row['out']}  (from {row['from']}; "
                  f"{_tally(row['errors'], row['warnings'])})")
    if missing:
        return EXIT_NOT_FOUND
    return EXIT_OK if ok else EXIT_USER
