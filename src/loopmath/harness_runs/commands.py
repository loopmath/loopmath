"""`loopmath adapt harness2 PATH`, `loopmath adapt harness-runs FILE`, `loopmath adapt plugin-eval RESULT`,
`loopmath adapt openevolve PATH` and `loopmath adapt shinkaevolve PATH` (docs/adapters.md, Run importers;
docs/plugin-eval.md).

Each reads rows, turn each into one OCP v0.3 run, and write `<run id>.ocp.json` files into `--out DIR` (or JSON
lines to stdout). Nothing is written until every row has made a run that passes strict OCP validation, so
a refusal leaves no partial output. `loopmath run import DIR --finish` then stores the runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..ocp.emit import validate_strict
from .ocp import NS, to_ocp
from .rows import SCHEMA, experiment_problem, read_numbered, task_type_problem

IMPORTERS = ("harness2", "harness-runs", "plugin-eval", "openevolve", "shinkaevolve")
# The evolution importers' own flags, and which of them each takes (docs/adapters.md, OpenEvolve and ShinkaEvolve).
EVOLVE_FLAGS = {"task": "--task", "model": "--model", "metric": "--metric", "min_delta": "--min-delta",
                "prices": "--prices", "children_only": "--children-only"}
EVOLVE_TAKES = {"openevolve": ("task", "model", "metric", "min_delta", "prices"),
                "shinkaevolve": ("task", "min_delta", "children_only")}
SUMMARY_SCHEMA = "loopmath.adapt.runs/1"
INPUTS = {"harness2": ("harness2 runs folder", "folder"), "harness-runs": ("rows file", "file"),
          "plugin-eval": ("result file, or the results folder holding aggregate-result.json", "file or folder"),
          "openevolve": ("OpenEvolve output folder, checkpoint folder or evolution_trace.jsonl", "folder or file"),
          "shinkaevolve": ("ShinkaEvolve results folder or its programs.sqlite", "folder or file")}
# `--model` is shared with openevolve, so the evolution check below refuses it for the importers that take none.
PLUGIN_EVAL_ONLY = (("cost_from", "--cost-from"), ("outcome", "--outcome"), ("plugin", "--plugin"),
                    ("keep_text", "--keep-text"))


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
            cost["priced" if source in ("loopmath-prices", "price-table") else "opencode" if source == "opencode"
                 else "reported"] += 1
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
    if s.get("accounting"):
        a = s["accounting"]
        lines.append(f"  dollars: ${a['usd_total']:.6f} spent by the settled proposals, ${a['usd_attributed']:.6f} in "
                     f"rows, ${a['usd_unattributed']:.6f} in none; ${a['usd_embed_novelty']:.6f} on embeddings and "
                     f"novelty checks, not charged to mutators")
    for note in s.get("notes") or []:
        lines.append(f"  note: {note}")
    held = s.get("counts") or {}
    if held.get("records_withheld") or held.get("text_withheld"):
        lines.append(f"  privacy metadata_only: withheld {held.get('records_withheld', 0)} harness2 records and "
                     f"{held.get('text_withheld', 0)} error or judge texts, which hold text a model wrote; "
                     f"--privacy full keeps them")
    pe = s.get("plugin_eval")
    if pe is not None:
        lines[1:1] = _plugin_lines(pe)
        if pe["next"]:
            lines += ["next:", *pe["next"]]
        return "\n".join(lines)
    if s["out"]:
        lines.append(f"next: loopmath run import {s['out']} --finish")
    return "\n".join(lines)


def _counted(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _plugin_lines(pe: dict[str, Any]) -> list[str]:
    """The plugin eval block of the text summary: what was compared, and what the suite cost."""
    p = pe["plugin"]
    who = (f"plugin {p['name']}" + (f" {p['version']}" if p.get("version") else "")) if p else "plugin not named"
    claude = f", Claude Code {pe['claude_version']}" if pe["claude_version"] else ""
    runs = pe["runs"]
    if pe["compared"]:
        what = (f"{_counted(pe['cases'], 'case')}, {pe['compared']} compared: {_counted(pe['pairs'], 'pair')} "
                f"({runs['with']} with runs, {runs['without']} without)")
    else:
        what = f"{_counted(pe['cases'], 'case')}, one arm: {_counted(runs['with'], 'with run')}, nothing to compare"
    lines = [f"  {who}{claude}: {what}"]
    if pe["suite_cost_usd"] is not None:
        lines.append(f"  suite cost ${pe['suite_cost_usd']:.2f} (list price, with judge calls; not split over runs)")
    cf = pe["cost_from"]
    if cf is None:
        lines.append("  per-run cost unknown: plugin eval records the suite's cost only; --cost-from FILE gives "
                     "each run its dollars or tokens, from your own records, for cost per accepted result")
    else:
        lines.append(f"  per-run cost from --cost-from: {cf['runs_with_cost']} of "
                     f"{cf['runs_with_cost'] + cf['runs_without']} runs ({_counted(cf['lines'], 'line')})")
    return lines


def _plugin_next(pe: dict[str, Any], out: Path | None, rows: Path | None) -> tuple[list[str], dict[str, Any] | None]:
    """The commands after a plugin eval import, and the gate's cap: the cap from-runs picks for these pairs."""
    from ..compare.calibration import TABLE
    from ..compare.commands import auto_cap

    lines = [f"loopmath run import {shlex.quote(str(out))} --finish"] if out is not None else []
    if not pe["compared"]:
        return lines, None
    score = pe["outcome"] == "score"
    lines.append("loopmath check-change from-runs --base-label arm:without --cand-label arm:with"
                 + (" --rule score:plugin-eval" if score else ""))
    cap = auto_cap(pe["pairs"])
    runs = f" --runs {shlex.quote(str(rows))}" if rows is not None else ""
    lines.append(f"loopmath gate --base arm:without --cand arm:with{runs} --rule "
                 f"{'score:plugin-eval' if score else 'accept'} --r-max {cap['max_pairs']}")
    if score and pe["pairs"] > pe["compared"]:
        lines.append(f"# gate pairs runs, not cases: its answer is about these cases; check-change from-runs "
                     f"compares the {_counted(pe['compared'], 'case')}, one pair each")
    largest = max(int(c) for entry in TABLE.values() for by_cap in entry["thresholds"].values() for c in by_cap)
    if cap["max_pairs"] > largest:
        lines.append(f"# {_counted(pe['pairs'], 'pair')}: the cap is past the largest calibrated cap, {largest}, so "
                     f"no false adopt bound is stated and adopting needs the strictest calibrated threshold")
    return lines, cap


def evolve_price(path: str | Path) -> Any:
    """`adapt openevolve --prices FILE.toml`: the table is loaded and priced here, and handed to the reader as an
    `adapters.evolve.TablePrice`; the adapters never import loopmath's pricing."""
    from ..adapters.evolve import NeedsFlag, TablePrice
    from ..price import load_prices, price_run

    try:
        table = load_prices(path)
    except OSError as exc:
        raise NeedsFlag(f"--prices {path}: not readable ({exc.strerror or exc})") from None
    except Exception as exc:   # tomllib.TOMLDecodeError and load_prices' own ValueErrors
        raise NeedsFlag(f"--prices {path}: not a price table ({exc})") from None

    def usd(model: str, tokens: dict[str, int]) -> float | None:
        got = price_run({"model": model, "tokens": dict(tokens)}, table).get("usd")
        return None if got is None else float(got)

    return TablePrice(usd=usd, path=str(table.path), as_of=table.as_of)


def adapt_runs(args: argparse.Namespace) -> int:
    """The run importers' CLI. Exit 0 written, 1 refused (nothing written), 2 usage or a missing input."""
    name = args.name
    if name != "plugin-eval":
        for attr, flag in PLUGIN_EVAL_ONLY:
            if getattr(args, attr, None) not in (None, False):
                return _fail(args, name, f"adapt {name}: {flag} is for adapt plugin-eval only", 2)
    if not getattr(args, "path", None):
        return _fail(args, name, f"adapt {name} needs a PATH (the {INPUTS[name][0]})", 2)
    path = Path(args.path).expanduser()
    if not path.exists():
        return _fail(args, name, f"adapt {name}: {path}: no such {INPUTS[name][1]}", 2)
    if getattr(args, "privacy", None) == "full" and name != "harness2":
        why = ("its own text switch is --keep-text, which keeps prompts, grader names and other free text"
               if name == "plugin-eval" else
               "it copies no text the tool wrote: code, diffs, prompts and replies stay out of the runs"
               if name in EVOLVE_TAKES else
               "a rows file's text is the writer's; its privacy profile follows what the rows hold")
        return _fail(args, name, f"adapt {name}: --privacy full applies to adapt harness2 only ({why})", 2)
    if name == "plugin-eval":
        if args.accept_at is not None and args.outcome == "score":
            return _fail(args, name, f"adapt {name}: --accept-at sets the accept threshold; --outcome score keeps "
                         "each run's grade as its score, so give one or the other", 2)
        if args.accept_at is not None and not 0 < args.accept_at <= 1:
            return _fail(args, name, f"adapt {name}: --accept-at {args.accept_at:g}: a grade from 0 to 1, above 0", 2)
        if args.cost_from is not None and not Path(args.cost_from).expanduser().is_file():
            return _fail(args, name, f"adapt {name}: --cost-from {Path(args.cost_from).expanduser()}: no such file", 2)
    given = [k for k in EVOLVE_FLAGS if getattr(args, k, None) not in (None, False)]
    if name in EVOLVE_TAKES and args.accept_at is not None:
        return _fail(args, name, f"adapt {name}: --accept-at is not taken: a row's outcome is whether the call "
                     f"wrote a child that beat its parent (use --min-delta for a margin)", 2)
    if name == "shinkaevolve" and args.model is not None:
        return _fail(args, name, "adapt shinkaevolve: --model is not taken: ShinkaEvolve records the model of "
                     "every query", 2)
    takers = {k: [n for n in IMPORTERS if k in EVOLVE_TAKES.get(n, ()) or (k == "model" and n == "plugin-eval")]
              for k in EVOLVE_FLAGS}
    if extra := [k for k in given if name not in takers[k]]:
        return _fail(args, name, f"adapt {name}: " + "; ".join(
            f"{EVOLVE_FLAGS[k]} is for adapt {' and '.join(takers[k])} only" for k in extra), 2)
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

            privacy = getattr(args, "privacy", None) or "metadata_only"
            got = H2.read(path, experiment=args.experiment, privacy=privacy)
            numbered, fmt, base_dir = [(None, r) for r in got.rows], got.format, None
            extra.update(skipped=got.skipped, notes=got.notes, experiments=got.experiments, counts=got.counts,
                         privacy=privacy)
        elif name == "plugin-eval":
            from ..adapters import plugin_eval as PE

            got = PE.read(path, experiment=args.experiment, cost_from=args.cost_from, accept_at=args.accept_at,
                          outcome=args.outcome or "accept", model=args.model, plugin=args.plugin,
                          keep_text=args.keep_text)
            numbered, fmt, base_dir = [(None, r) for r in got.rows], got.format, None
            extra.update(notes=got.notes, experiments=got.experiments, counts=got.counts, plugin_eval=got.summary)
        elif name in EVOLVE_TAKES:
            from ..adapters.evolve import NeedsFlag

            kw = {k: getattr(args, k) for k in EVOLVE_TAKES[name] if getattr(args, k, None) is not None}
            try:
                if "prices" in kw:
                    kw["price"] = evolve_price(kw.pop("prices"))
                if name == "openevolve":
                    from ..adapters import openevolve as reader
                else:
                    from ..adapters import shinkaevolve as reader
                got = reader.read(path, experiment=args.experiment, task_type=args.type, repo=args.repo, **kw)
            except NeedsFlag as exc:
                return _fail(args, name, f"adapt {name}: {exc}", 2)
            numbered, fmt, base_dir = [(None, r) for r in got.rows], got.format, None
            extra.update(skipped=got.skipped, notes=got.notes, experiments=got.experiments, counts=got.counts)
            if got.accounting:
                extra["accounting"] = got.accounting
        else:
            numbered, fmt, base_dir = read_numbered(path), SCHEMA, path.parent
            if args.experiment:
                numbered = [(n, {**r, "experiment": r.get("experiment") or args.experiment}) for n, r in numbered]
            extra["experiments"] = _experiments([r for _, r in numbered])
        default = _mtime(path)
        accept_at = None if name == "plugin-eval" else args.accept_at   # the reader applied it to each grade
        unscored = {"accepted": True} if name == "plugin-eval" and (args.outcome or "accept") == "accept" else None
        docs = []
        for n, row in numbered:
            try:
                docs.append(to_ocp(row, base_dir=base_dir, accept_at=accept_at, observed_default=default,
                                   task_type=args.type, repo=args.repo, unscored_rule=unscored))
            except (OSError, ValueError) as exc:
                raise ValueError(f"{path}:{n}: {exc}" if n is not None else f"row {_where(row)}: {exc}") from None
    except (OSError, ValueError) as exc:   # FormatError is a ValueError
        return _fail(args, name, f"adapt {name}: {exc}", 1)
    rows = [r for _, r in numbered]
    if not rows:
        if name == "harness-runs":
            return _fail(args, name, f"adapt {name}: {path} has no rows (it is empty or blank); nothing written", 1)
        if name == "plugin-eval":
            return _fail(args, name, f"adapt {name}: {path} has no runs to import; nothing written", 1)
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
    if "plugin_eval" in extra:
        nxt, cap = _plugin_next(extra["plugin_eval"], out, Path(args.rows).expanduser() if args.rows else None)
        extra["plugin_eval"] = {**extra["plugin_eval"], "cap": cap, "next": nxt}
    summary = _summary(name, fmt, rows, docs, out, extra)
    stream = sys.stdout if out is not None else sys.stderr
    print(json.dumps(summary, indent=1, ensure_ascii=False) if args.json else _text(summary), file=stream)
    return 0
