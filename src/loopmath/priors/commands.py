"""Handlers for `loopmath prior build|show` (lane 11; hidden from `--help`).

`prior build [--out DIR] [--sweep-dir P] [--e0-corpus P] [--rq1-dir P]`
rebuilds the shipped bundle from our sources. The inputs are read only; each
is its flag, else `LOOPMATH_SWEEP_DIR`, `LOOPMATH_E0_CORPUS` or
`LOOPMATH_PRIOR_RQ1`, and there is no default folder (D70).
`LOOPMATH_PRIOR_SOURCES` (comma list) limits the sources, and
`LOOPMATH_PRIOR_SALT` fixes the repo-hash salt for a reproducible build. A
missing input is an error, never a silently smaller bundle.

`prior show` prints the bundle's contents and provenance from its manifest.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..output import EXIT_NOT_FOUND, EXIT_OK, EXIT_USER, emit_json, fail


def _selected() -> list[str]:
    from .build import RUNNERS

    raw = os.environ.get("LOOPMATH_PRIOR_SOURCES", "")
    names = [s.strip() for s in raw.split(",") if s.strip()] or list(RUNNERS)
    unknown = [n for n in names if n not in RUNNERS]
    if unknown:
        raise ValueError(f"unknown prior source(s): {', '.join(unknown)}; known: {', '.join(RUNNERS)}")
    return names


def build(args: argparse.Namespace) -> int:
    from .build import BUNDLE_DIR, RUNNERS, BundleError, build_bundle
    from .registry import ENV_INPUTS, INPUT_FLAGS, SWEEP, E0, RQ1, MissingInput, input_path

    try:
        names = _selected()
    except ValueError as exc:
        return fail(str(exc))
    out_dir = Path(args.out).expanduser() if args.out else BUNDLE_DIR
    flags = {SWEEP: getattr(args, "sweep_dir", None), E0: getattr(args, "e0_corpus", None),
             RQ1: getattr(args, "rq1_dir", None)}
    results = []
    try:
        for name in names:
            try:
                path = input_path(name, flags.get(name))
            except MissingInput as exc:
                return fail(f"{exc}, or leave {name} out of LOOPMATH_PRIOR_SOURCES")
            if not path.exists():
                return fail(f"input for {name} not found at {path}; pass {INPUT_FLAGS[name]} PATH or set "
                            f"{ENV_INPUTS[name]}, or leave {name} out of LOOPMATH_PRIOR_SOURCES", EXIT_NOT_FOUND)
            print(f"converting {name} from {path}", file=sys.stderr)
            results.append(RUNNERS[name](path))
        manifest = build_bundle(out_dir, results, salt=os.environ.get("LOOPMATH_PRIOR_SALT") or None)
    except BundleError as exc:
        return fail(f"bundle not written: {exc}")
    if args.json:
        emit_json("loopmath.prior.build/1", {"out": str(out_dir), "manifest": manifest})
        return EXIT_OK
    print(f"wrote {out_dir} ({manifest['size_bytes'] / 1e6:.2f} MB of {manifest['size_cap_bytes'] / 1e6:.0f} MB)")
    for name, entry in manifest["sources"].items():
        v = entry["validation"]
        print(f"  {name:<13} {entry['runs']:>5} runs  {entry['bytes'] / 1e3:>7.0f} kB  "
              f"errors {v['errors']}, warnings {v['warnings']}")
    return EXIT_OK


def show(args: argparse.Namespace) -> int:
    from . import bundle_dir, manifest

    data = manifest()
    if not data.get("sources"):
        return fail(f"no prior bundle at {bundle_dir()}; run `loopmath prior build`", EXIT_NOT_FOUND)
    if args.json:
        emit_json("loopmath.prior.show/1", {"bundle": str(bundle_dir()), "manifest": data})
        return EXIT_OK
    print(f"Prior bundle: {sum(e['runs'] for e in data['sources'].values())} runs from "
          f"{len(data['sources'])} sources, {data['size_bytes'] / 1e6:.2f} MB, built {data['built_at']}.")
    print(f"OCP {data['ocp']}; config ids by {data['config_id_impl']}; tariff {data['tariff']['id']} "
          f"({data['tariff']['date']}); shared form: no titles, paths, commands or free text, "
          "private repo names hashed.")
    for name, e in data["sources"].items():
        s = e["summary"]
        types = ", ".join(f"{k} {v}" for k, v in s["types"].items())
        outcomes = ", ".join(f"{k} {v}" for k, v in s["outcomes"].items())
        print(f"\n{name}: {e['runs']} runs ({types}); {outcomes}")
        gaps = [f"{s[key]} {what}" for key, what in (("cost_unknown_attempts", "attempts with unknown usage"),
                                                      ("unpriced_attempts", "attempts with no price")) if s.get(key)]
        left_out = f" (not counting {' and '.join(gaps)})" if gaps else ""
        print(f"  {s['tokens']:,} tokens, ${s['usd']:,.2f}{left_out}; converter {e['converter']}; "
              f"{e['inputs'].get('files', '?')} input files, sha256 {str(e['inputs'].get('sha256', ''))[:12]}")
        by_model = sorted(s["attempts_by_model"].items(), key=lambda kv: -kv[1])
        top = f" (top 6 of {len(by_model)})" if len(by_model) > 6 else ""
        print(f"  attempts by model{top}: {', '.join(f'{k} {v}' for k, v in by_model[:6])}")
        print(f"  validation: {e['validation']['errors']} errors, {e['validation']['warnings']} warnings "
              f"({e['validation']['checker']})")
    return EXIT_OK if not data.get("problems") else EXIT_USER
