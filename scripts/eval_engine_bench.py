#!/usr/bin/env python3
"""Benchmark and recovery harness for the engines of `loopmath eval fit`.

    eval_engine_bench.py time    --engines pymc,laplace --specs linear,size-target --out DIR
    eval_engine_bench.py recover --engine pymc --specs linear,levels-3 --studies 200 --out DIR
    eval_engine_bench.py stats   --out DIR [--engine pymc]
    eval_engine_bench.py compare --out DIR --engines pymc,laplace

Run it from the repository root with the project's Python, one thread per process and at low priority:

    PYTHONPATH=src LOOPMATH_NO_UPDATE_CHECK=1 OMP_NUM_THREADS=1 nice -n 10 python scripts/eval_engine_bench.py ...

`time` runs `loopmath eval fit` as its own process per spec and engine, as a user would, and keeps one
timing record per run: wall seconds, the engine's own seconds, CPU seconds of the process and its
children, convergence and the counts. `size-target` is the size target study (500 runs, 5 components,
4 judges, 2 repeats, 20,000 readings).

`recover` simulates N studies per spec and fits each in this process with `fit.fit_design`, then keeps
one record per study (truth, estimate, interval, converged, seconds; `tests/eval/recovery_stats.py`
describes it). A record that exists is kept, so a run that stops resumes where it stopped; a record made
with other options is refused, never overwritten (use a new DIR). A fit that raises is recorded with its
error and counts as a failed fit. Create DIR/STOP and the run stops before its next fit. `--workers N`
starts N processes, worker k taking the seeds with seed mod N = k; `--cpu-minutes M` stops starting fits
once this process and its children have used M CPU minutes; `--dry-run` prints the plan and an estimate
from records already made.

Budgets are totals. `--total-cores C` (default 2) is shared by the workers: each fit gets `--cores`
(default: C over the workers), and workers times cores may not pass C. `--cpu-minutes` and
`--max-fits` are split evenly between the workers, so the run as a whole stays inside them. The fits of
this process run with one thread per process (`OMP_NUM_THREADS=1`); the harness refuses to start a fit
otherwise.

`stats` reads the records of each engine and prints, per spec: coverage per parameter class with its
interval over independent studies, the arm means one by one, bias per class, the parameters whose
coverage is outside 85% to 95% by their own binomial interval, degenerate parameters, family false
alarms counted once per study, and the spec's part of the gate; then the engine gate item by item, each
`pass`, `fail` or `incomplete`. The gate's required specs are the whole catalog and the seven specs that
read a binary or levels component repeatedly, 200 studies each, whatever records exist; `--catalog`,
`--repeated`, `--studies`, `--catalog-studies` (studies per spec for the clean-fit items) and
`--coverage-specs` narrow it only with `--scope-note` saying who approved that. `compare` puts two engines side by side on the studies both fitted.

`time` checks before it starts that a fit cannot use more than `--cores` (default 2) cores: `eval fit`
runs its chains side by side, so without a `--cores` option in `eval fit` the chains may not exceed it.
A timing record keeps the cores measured (CPU over wall seconds) and the threads per process, and every
attempt counts: the gate reads failed and unconverged attempts as failures.

Engines. `pymc` runs `engine_pymc` through `fit.fit_design`. Any other name is passed on as
`fit_design(..., engine=NAME)` and `eval fit --engine NAME`; in a checkout without that option only
`pymc` runs, and the harness says so.

Layout under DIR:

    studies/SPEC-iN/sSEED/            the simulated study and its truth.json (N: inputs, or "all")
    records/ENGINE/SPEC-iN/sSEED.json one record per study
    timing/ENGINE/SPEC-iN-sSEED/      the fit file of the timed fit; timing records beside it, one per run
    stats-ENGINE.json, compare-A-B.json

Every study here is simulated.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "tests" / "eval", REPO / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import numpy as np  # noqa: E402

import recovery_stats as RS  # noqa: E402

DEFAULT_CHAINS = 2      # 2 cores per fit
TOTAL_CORES = 2         # cores shared by the workers of one run
OPTION_KEYS = ("inputs", "draws", "chains", "tune", "keep_draws", "family")


# ---------------------------------------------------------------------------------------------------
# Shared pieces


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def cpu_seconds() -> float:
    """User and system CPU of this process and of its children that have ended."""
    own, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


def label(spec: str, inputs: int | None) -> str:
    return f"{spec}-i{inputs if inputs is not None else 'all'}"


def study_folder(out: Path, spec: str, inputs: int | None, seed: int) -> Path:
    return out / "studies" / label(spec, inputs) / f"s{seed}"


def ensure_study(out: Path, spec: str, inputs: int | None, seed: int) -> Path:
    """The simulated study of a spec and seed, written once. It is written into a folder of its own and
    renamed into place, so two runs that need the same study at once do not mix files."""
    from loopmath.eval import simulate

    folder = study_folder(out, spec, inputs, seed)
    if (folder / simulate.TRUTH_FILE).exists():
        return folder
    sim = simulate.generate(RS.spec_for(spec, inputs), seed=seed)
    scratch = folder.with_name(f".{folder.name}.{os.getpid()}.new")
    simulate.write(sim, scratch)
    try:
        os.rename(scratch, folder)
    except OSError:
        if not (folder / simulate.TRUTH_FILE).exists():
            raise
    return folder


def fit_engine_kwargs(engine: str) -> dict:
    """How to ask `fit.fit_design` for an engine, or a clear stop when this checkout cannot."""
    from loopmath.eval import fit as FIT

    if "engine" in inspect.signature(FIT.fit_design).parameters:
        return {"engine": engine}
    if engine != "pymc":
        raise SystemExit(f"fit.fit_design in this checkout takes no engine argument: only pymc can run; "
                         f"asked for {engine}")
    return {}


def cli_flags() -> set[str]:
    """The options `loopmath eval fit` here takes (`--engine`, `--cores` are the ones the harness asks about)."""
    got = subprocess.run([sys.executable, "-m", "loopmath", "eval", "fit", "--help"], capture_output=True, text=True,
                         env=child_env(), timeout=300)
    return {word.strip(",[]") for word in got.stdout.split() if word.startswith("--")}


def child_env() -> dict:
    """The environment of a child process: this checkout first on the path, no update check, one thread."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("LOOPMATH_NO_UPDATE_CHECK", "1")
    env["OMP_NUM_THREADS"] = "1"
    return env


def one_thread() -> None:
    """Refuse to fit in this process unless it runs one thread per process."""
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("set OMP_NUM_THREADS=1 before fitting (one thread per process); found "
                         f"{os.environ.get('OMP_NUM_THREADS')!r}")


def csv(text: str) -> list[str]:
    return [word.strip() for word in text.split(",") if word.strip()]


def check_specs(names: list[str]) -> None:
    from loopmath.eval import specs

    known = set(specs.names()) | {RS.SIZE_TARGET}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit(f"unknown spec {', '.join(unknown)}; known: {', '.join(specs.names())}, {RS.SIZE_TARGET}")


# ---------------------------------------------------------------------------------------------------
# recover


def options_of(args: argparse.Namespace) -> dict:
    return {"inputs": args.inputs, "draws": args.draws, "chains": args.chains, "tune": args.tune,
            "keep_draws": args.keep_draws, "family": not args.no_family}


def record_path(out: Path, engine: str, spec: str, inputs: int | None, seed: int) -> Path:
    return out / "records" / engine / label(spec, inputs) / f"s{seed}.json"


def kept(path: Path, options: dict) -> bool:
    """True when the record exists with the same options; a record with other options, or of an older
    schema, stops the run."""
    if not path.exists():
        return False
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema") != RS.RECORD_SCHEMA:
        raise SystemExit(f"{path} is a record of schema {obj.get('schema')}, not {RS.RECORD_SCHEMA}: it has no "
                         "parameter inventory. Use a new --out folder: a record is never overwritten.")
    old = obj.get("options", {})
    same = {key: old.get(key) for key in OPTION_KEYS} == {key: options.get(key) for key in OPTION_KEYS}
    if not same:
        raise SystemExit(f"{path} was made with other options ({old}); this run asks for {options}. "
                         "Use a new --out folder: a record is never overwritten.")
    return True


def fit_study(out: Path, engine: str, spec: str, seed: int, args: argparse.Namespace) -> dict:
    """Simulate (once), fit and compare one study; the record, also written to its file."""
    from loopmath.eval import contracts as C
    from loopmath.eval import fit as FIT
    from loopmath.eval import model as M
    from loopmath.eval import simulate

    options = options_of(args)
    record = {"schema": RS.RECORD_SCHEMA, "spec": spec, "seed": seed, "engine": engine, "options": options,
              "cores": args.cores, "started": now(), "converged": False, "rows": [], "aggregates": {}, "family": None,
              "expected": {}, "degenerate": [], "no_truth": [], "excluded": {}, "seconds": None, "cpu_seconds": None}
    folder = ensure_study(out, spec, args.inputs, seed)
    truth = simulate.read_truth(folder / simulate.TRUTH_FILE)
    start, cpu = time.monotonic(), cpu_seconds()
    try:
        design = M.build(C.read_study(folder), C.read_document(folder / "rubric.json", C.SCHEMA_RUBRIC),
                         C.read_document(folder / "judges.json", C.SCHEMA_JUDGES), allow_simulated=True)
        record["build_seconds"] = time.monotonic() - start
        start, cpu = time.monotonic(), cpu_seconds()
        meta, draws = FIT.fit_design(design, seed=seed, draws=args.draws, chains=args.chains, tune=args.tune,
                                     cores=args.cores, keep_draws=args.keep_draws, **fit_engine_kwargs(engine))
        record["seconds"], record["cpu_seconds"] = time.monotonic() - start, cpu_seconds() - cpu
        record.update(converged=bool(meta["diagnostics"]["converged"]), diagnostics=meta["diagnostics"],
                      engine_meta=meta.get("engine"), counts=meta.get("counts"))
        record.update(RS.study_record(design, meta, draws, truth, family=not args.no_family))
    except Exception as exc:   # a failed fit is a result: it is recorded and counted
        record["seconds"], record["cpu_seconds"] = time.monotonic() - start, cpu_seconds() - cpu
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    record["finished"] = now()
    RS.write_json_atomic(record_path(out, engine, spec, args.inputs, seed), record)
    return record


def plan(args: argparse.Namespace) -> list[tuple[str, int]]:
    seeds = range(args.start, args.start + args.studies)
    k, n = args.shard
    return [(spec, seed) for spec in args.specs for seed in seeds if seed % n == k]


def estimate(out: Path, engine: str, spec: str, inputs: int | None) -> float | None:
    """Median CPU seconds per fit from records already made for this engine and spec."""
    folder = out / "records" / engine / label(spec, inputs)
    cpus = [r["cpu_seconds"] for r in RS.load_records(folder) if r.get("cpu_seconds")] if folder.exists() else []
    return float(np.median(cpus)) if cpus else None


def budgets(args: argparse.Namespace) -> list[dict]:
    """The limits of each worker, so that together they stay inside the run's totals: cores per fit
    (workers times cores at most `--total-cores`), CPU minutes and fits split evenly. A worker left with
    no fit is not started."""
    n = args.workers
    if n < 1:
        raise SystemExit("--workers must be at least 1")
    if n > args.total_cores:
        raise SystemExit(f"{n} workers need at least {n} cores; --total-cores is {args.total_cores}")
    cores = args.cores if args.cores is not None else args.total_cores // n
    if cores < 1 or n * cores > args.total_cores:
        raise SystemExit(f"{n} workers of {cores} cores each pass --total-cores {args.total_cores}")
    out = []
    for k in range(n):
        fits = None if args.max_fits is None else args.max_fits // n + (1 if k < args.max_fits % n else 0)
        if fits == 0:
            continue
        out.append({"shard": k, "cores": cores, "max_fits": fits,
                    "cpu_minutes": None if args.cpu_minutes is None else args.cpu_minutes / n})
    return out


def recover(args: argparse.Namespace) -> int:
    out = Path(args.out)
    check_specs(args.specs)
    limits = budgets(args)
    if not limits:
        print("nothing to fit: --max-fits is 0", flush=True)
        return 0
    if args.workers > 1 and args.shard == (0, 1) and not args.dry_run:
        return spawn_workers(args, limits)
    if args.cores is None:
        args.cores = limits[0]["cores"]
    fit_engine_kwargs(args.engine)
    options = options_of(args)
    todo = [(spec, seed) for spec, seed in plan(args)
            if not kept(record_path(out, args.engine, spec, args.inputs, seed), options)]
    total = len(plan(args))
    if args.dry_run:
        print(f"engine {args.engine}: {total} studies planned, {total - len(todo)} kept, {len(todo)} to fit")
        cost = 0.0
        for spec in args.specs:
            left = sum(1 for s, _ in todo if s == spec)
            per = estimate(out, args.engine, spec, args.inputs)
            cost += (per or 0.0) * left
            print(f"  {label(spec, args.inputs)}: {left} to fit, "
                  + (f"median {per:.0f} CPU s per fit from earlier records" if per else "no earlier record to estimate from"))
        print(f"estimated CPU minutes (specs with an estimate): {cost / 60:.1f}")
        return 0
    one_thread()
    print(f"limits: {args.cores} cores per fit, CPU minutes {args.cpu_minutes}, fits {args.max_fits}", flush=True)
    used = cpu_seconds()
    done = 0
    for spec, seed in todo:
        if (out / "STOP").exists():
            print(f"stopped: {out / 'STOP'} exists", flush=True)
            return 0
        if args.cpu_minutes is not None and (cpu_seconds() - used) / 60.0 >= args.cpu_minutes:
            print(f"stopped: {args.cpu_minutes} CPU minutes used", flush=True)
            return 0
        if args.max_fits is not None and done >= args.max_fits:
            print(f"stopped: {args.max_fits} fits made", flush=True)
            return 0
        record = fit_study(out, args.engine, spec, seed, args)
        done += 1
        rows = record["rows"]
        share = sum(row["hit"] for row in rows) / len(rows) if rows else float("nan")
        state = f"error {record['error'][:120]}" if record.get("error") else (
            "converged" if record["converged"] else "not converged")
        print(f"{record['finished']} {args.engine} {label(spec, args.inputs)} s{seed}: {record['seconds']:.1f} s, "
              f"{record['cpu_seconds']:.0f} CPU s, {state}, {len(rows)} rows, {share:.2f} of intervals hold the truth",
              flush=True)
    return 0


def worker_argv(args: argparse.Namespace, limit: dict) -> list[str]:
    """The command line of one worker, built from the parsed options and the worker's own limits."""
    argv = ["recover", "--out", str(args.out), "--specs", ",".join(args.specs), "--engine", args.engine,
            "--studies", str(args.studies), "--start", str(args.start), "--chains", str(args.chains),
            "--keep-draws", str(args.keep_draws), "--workers", "1", "--total-cores", str(limit["cores"]),
            "--cores", str(limit["cores"]), "--shard", f"{limit['shard']}/{args.workers}"]
    for flag, value in (("--inputs", args.inputs), ("--draws", args.draws), ("--tune", args.tune),
                        ("--max-fits", limit["max_fits"]), ("--cpu-minutes", limit["cpu_minutes"])):
        if value is not None:
            argv += [flag, str(value)]
    if args.no_family:
        argv.append("--no-family")
    return argv


def spawn_workers(args: argparse.Namespace, limits: list[dict]) -> int:
    """Start one worker per limit, each on one shard of the seeds with its share of the budgets, and wait."""
    procs = []
    for limit in limits:
        cmd = [sys.executable, str(Path(__file__).resolve()), *worker_argv(args, limit)]
        procs.append(subprocess.Popen(cmd, env=child_env()))
        print(f"worker {limit['shard']}: pid {procs[-1].pid}, {limit['cores']} cores per fit, CPU minutes "
              f"{limit['cpu_minutes']}, fits {limit['max_fits']}", flush=True)
    codes = [proc.wait() for proc in procs]
    return max(codes)


# ---------------------------------------------------------------------------------------------------
# time


def time_fits(args: argparse.Namespace) -> int:
    out = Path(args.out)
    check_specs(args.specs)
    flags = cli_flags()
    engine_flag = "--engine" in flags
    for engine in args.engines:
        if engine != "pymc" and not engine_flag:
            raise SystemExit(f"`loopmath eval fit` in this checkout has no --engine: {engine} cannot be timed yet")
    if args.cores < 1 or args.cores > RS.GATE_CORES:
        raise SystemExit(f"--cores {args.cores}: the gate times a fit on at most {RS.GATE_CORES:g} cores")
    args.pass_cores = "--cores" in flags
    if not args.pass_cores and args.chains > args.cores:
        raise SystemExit(f"`loopmath eval fit` here has no --cores and runs its {args.chains} chains side by side; "
                         f"use --chains at most --cores ({args.cores})")
    for spec in args.specs:
        folder = ensure_study(out, spec, args.inputs, args.seed)
        for engine in args.engines:
            for rep in range(args.repeat):
                if (out / "STOP").exists():
                    print(f"stopped: {out / 'STOP'} exists", flush=True)
                    return 0
                print(json.dumps(time_one(out, folder, engine, spec, args, engine_flag, rep)), flush=True)
    return 0


def time_one(out: Path, folder: Path, engine: str, spec: str, args: argparse.Namespace, engine_flag: bool,
             rep: int) -> dict:
    where = out / "timing" / engine / f"{label(spec, args.inputs)}-s{args.seed}"
    where.mkdir(parents=True, exist_ok=True)
    fit_json = where / "fit.json"
    argv = [sys.executable, "-m", "loopmath", "eval", "fit", str(folder), "--rubric", str(folder / "rubric.json"),
            "--judges", str(folder / "judges.json"), "--seed", str(args.seed), "--allow-simulated",
            "--chains", str(args.chains), "--out", str(fit_json)]
    if args.draws is not None:
        argv += ["--draws", str(args.draws)]
    if args.keep_draws is not None:
        argv += ["--keep-draws", str(args.keep_draws)]
    if engine_flag:
        argv += ["--engine", engine]
    if args.pass_cores:
        argv += ["--cores", str(args.cores)]
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    before = kids.ru_utime + kids.ru_stime
    started, start = now(), time.monotonic()
    got = subprocess.run(argv, capture_output=True, text=True, env=child_env())
    wall = time.monotonic() - start
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = kids.ru_utime + kids.ru_stime - before
    record = {"schema": RS.TIMING_SCHEMA, "spec": spec, "inputs": args.inputs, "seed": args.seed, "engine": engine,
              "repeat": rep, "full_size": spec == RS.SIZE_TARGET and args.inputs is None, "started": started,
              "wall_seconds": wall, "cpu_seconds": cpu, "cores_measured": cpu / wall if wall > 0 else None,
              "cores_limit": args.cores, "exit_code": got.returncode, "argv": argv[1:],
              "threads": child_env().get("OMP_NUM_THREADS"), "cpu_count": os.cpu_count(), "chains": args.chains,
              "draws": args.draws, "converged": None, "counts": None}
    fresh = fit_json.exists() and fit_json.stat().st_mtime >= time.time() - wall - 1.0
    if got.returncode == 0 and fresh:
        meta = json.loads(fit_json.read_text(encoding="utf-8"))
        diagnostics = meta["diagnostics"]
        counts = dict(meta.get("counts") or {})
        counts.update(dimensions=len(meta.get("dimensions") or []), judges=len(meta["index"].get("judges") or []))
        record.update(converged=diagnostics["converged"], engine_seconds=diagnostics.get("run_seconds"),
                      diagnostics=diagnostics, engine_meta=meta.get("engine"), counts=counts)
    else:
        record["stderr_tail"] = got.stderr[-2000:]
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    RS.write_json_atomic(where / f"timing-{stamp}-r{rep}.json", record)
    return {key: record.get(key) for key in ("engine", "spec", "inputs", "wall_seconds", "engine_seconds",
                                             "cpu_seconds", "cores_measured", "converged", "exit_code")}


def load_timings(out: Path, engine: str) -> list[dict]:
    """Every timing record of an engine, of any schema version: the gate counts every attempt, and an
    attempt of an older schema fails it rather than disappearing."""
    folder = out / "timing" / engine
    rows = []
    for path in sorted(folder.rglob("timing-*.json")) if folder.exists() else []:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if str(obj.get("schema", "")).startswith("loopmath.eval-engine-bench-timing/"):
            rows.append(obj)
    return rows


# ---------------------------------------------------------------------------------------------------
# stats and compare


def grouped(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for record in records:
        out.setdefault(label(record["spec"], record["options"].get("inputs")), []).append(record)
    return out


def f(x, digits: int = 3, signed: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def render_summary(name: str, s: dict) -> list[str]:
    secs, cpu = s["seconds"], s["cpu_seconds"]
    lines = [f"{s['engine']} {name}: {s['studies']} studies, {len(s['failed'])} failed, {len(s['unconverged'])} "
             f"not converged; fit seconds median {f(secs['median'], 1)} (max {f(secs['max'], 1)}), "
             f"CPU seconds median {f(cpu['median'], 1)}, total CPU minutes {cpu['total'] / 60:.1f}",
             f"  {'class':<26} {'studies':>7} {'rows':>6} {'coverage':>8}  {'95% interval':<15} {'place':<9} "
             f"{'bias':>9} {'se':>8} {'bias/sd':>8}"]
    for klass, c in s["classes"].items():
        ci = f"[{f(c['lo'], 2)}, {f(c['hi'], 2)}]"
        lines.append(f"  {klass:<26} {c['studies']:>7} {c['rows']:>6} {f(c['coverage']):>8}  {ci:<15} {c['place']:<9} "
                     f"{f(c['bias'], 4, True):>9} {f(c['bias_se'], 4):>8} {f(c['bias_over_sd'], 2, True):>8}")
    lines.append("  arm means and differences, one by one:")
    for p in s["arm_means"]:
        lines.append(f"    {p['name']:<44} {p['hits']}/{p['studies']} [{f(p['lo'], 2)}, {f(p['hi'], 2)}] bias "
                     f"{f(p['bias'], 4, True)} se {f(p['bias_se'], 4)}"
                     + ("" if p["unbiased"] is not False else "  <- biased"))
    for p in s["parameters_outside"]:
        lines.append(f"  outside the band by its own interval: {p['name']} {p['hits']}/{p['studies']} "
                     f"(adjusted [{f(p['lo_adj'], 2)}, {f(p['hi_adj'], 2)}])")
    for d in s["degenerate"][:20]:
        lines.append(f"  degenerate: s{d['seed']} {d['name']} ({d['why']})")
    if len(s["degenerate"]) > 20:
        lines.append(f"  degenerate: {len(s['degenerate']) - 20} more")
    fam = s["family"]
    if fam["studies_skipped"]:
        lines.append(f"  family: {fam['studies_skipped']} studies made without the family measurement")
    lines.append(f"  family: {fam['studies_with_a_null_contrast']} studies measured a contrast with no simulated "
                 f"effect, {fam['studies_flagged']} flagged (share {f(fam['share_of_studies'], 2)}, 95% "
                 f"[{f(fam['lo'], 2)}, {f(fam['hi'], 2)}], limit {fam['limit']}); per contrast "
                 f"{f(fam['share_of_contrasts'], 2)}; simulated effects {fam['simulated_effects']}, found "
                 f"{f(fam['found'], 2)}")
    lines.append(f"  gate for this spec: {s['gate']['state']}")
    for key, item in s["gate"]["items"].items():
        lines.append(f"    {key}: {item['state']} ({item['why']})")
    return lines


def stats(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if not (out / "records").is_dir():
        raise SystemExit(f"no records under {out}: run `recover` first")
    try:
        scope = RS.gate_scope(args.catalog, args.repeated, args.studies, args.coverage_specs, args.scope_note,
                              args.catalog_studies)
    except ValueError as exc:
        raise SystemExit(f"gate scope: {exc}")
    engines = csv(args.engine) if args.engine else sorted(p.name for p in (out / "records").iterdir() if p.is_dir())
    result = {}
    for engine in engines:
        old = RS.old_records(out / "records" / engine)
        if old:
            print(f"{engine}: {old} records of an older schema are not read (no parameter inventory)")
        groups = grouped(RS.load_records(out / "records" / engine))
        summaries = {name: RS.summarise(records, studies_needed=scope["studies"])
                     for name, records in sorted(groups.items())}
        full = {name.rsplit("-i", 1)[0]: s for name, s in summaries.items() if name.endswith("-iall")}
        timings = load_timings(out, engine)
        gate = RS.engine_gate(full, timings, scope)
        result[engine] = {"summaries": summaries, "gate": gate, "old_records": old,
                          "timings": [{k: t.get(k) for k in ("spec", "inputs", "wall_seconds", "engine_seconds",
                                                              "cpu_seconds", "cores_measured", "threads", "converged",
                                                              "exit_code", "started")}
                                      for t in timings]}
        for name, s in summaries.items():
            print("\n".join(render_summary(name, s)))
            print()
        for t in result[engine]["timings"]:
            print(f"timing {engine} {label(t['spec'], t['inputs'])}: wall {f(t['wall_seconds'], 1)} s, engine "
                  f"{f(t['engine_seconds'], 1)} s, CPU {f(t['cpu_seconds'], 1)} s, cores {f(t['cores_measured'], 2)}, "
                  f"converged {t['converged']}, exit {t['exit_code']}")
        print(f"engine gate for {engine}: {gate['state']} (catalog-size records only; reported here, decided by people)")
        if scope["narrowed"]:
            print(f"  narrowed scope: {scope['note']}; specs left out: {', '.join(scope['omitted']) or 'none'}")
        for key, item in gate["items"].items():
            print(f"  {key}: {item['state']}  " + json.dumps({k: v for k, v in item.items() if k != "state"}))
        RS.write_json_atomic(out / f"stats-{engine}.json", result[engine])
    return 0


def compare(args: argparse.Namespace) -> int:
    out = Path(args.out)
    a, b = args.engines[:2]
    ga, gb = grouped(RS.load_records(out / "records" / a)), grouped(RS.load_records(out / "records" / b))
    result = {}
    for name in sorted(set(ga) & set(gb)):
        got = RS.compare(ga[name], gb[name])
        result[name] = got
        print(f"{name}: {got['studies']} studies fitted by both; {a} median {f(got['seconds_a']['median'], 1)} s, "
              f"{b} median {f(got['seconds_b']['median'], 1)} s; converged {got['converged_a']} and {got['converged_b']}")
        print(f"  {'class':<16} {'rows':>5} {'median |gap|/sd':>15} {'max':>7} {'width ' + b + '/' + a:>16} "
              f"{'cover ' + a:>10} {'cover ' + b:>10}  worst")
        for klass, c in got["classes"].items():
            print(f"  {klass:<16} {c['rows']:>5} {f(c['median_abs_gap_sd'], 2):>15} {f(c['max_abs_gap_sd'], 2):>7} "
                  f"{f(c['median_width_ratio'], 2):>16} {f(c['coverage_a'], 2):>10} {f(c['coverage_b'], 2):>10}  "
                  f"{c['worst']}")
    RS.write_json_atomic(out / f"compare-{a}-{b}.json", result)
    return 0


# ---------------------------------------------------------------------------------------------------


def shard(text: str) -> tuple[int, int]:
    k, n = (int(x) for x in text.split("/"))
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError("--shard K/N needs 0 <= K < N")
    return k, n


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="eval_engine_bench.py", description=__doc__.split("\n\n")[0])
    sub = top.add_subparsers(dest="command", required=True)

    def fit_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", required=True, help="the folder for studies, records, timings and summaries")
        p.add_argument("--specs", type=csv, required=True, help="catalog names and/or size-target, comma separated")
        p.add_argument("--inputs", type=int, default=None, help="inputs per study (default: the spec's own)")
        p.add_argument("--draws", type=int, default=None, help="draws per chain (default: eval fit's own)")
        p.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
        p.add_argument("--keep-draws", type=int, default=1000)

    p = sub.add_parser("recover", help="fit N simulated studies per spec; one record per study")
    fit_options(p)
    p.add_argument("--engine", default="pymc")
    p.add_argument("--studies", type=int, default=2)
    p.add_argument("--start", type=int, default=0, help="first seed")
    p.add_argument("--tune", type=int, default=None)
    p.add_argument("--cores", type=int, default=None,
                   help="cores for the chains of one fit (default: --total-cores over --workers)")
    p.add_argument("--total-cores", type=int, default=TOTAL_CORES, help="cores for the whole run, all workers")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard", type=shard, default=(0, 1), help=argparse.SUPPRESS)
    p.add_argument("--cpu-minutes", type=float, default=None, help="for the whole run, split between the workers")
    p.add_argument("--max-fits", type=int, default=None, help="for the whole run, split between the workers")
    p.add_argument("--no-family", action="store_true", help="skip the family contrasts (eval judges) per study")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(run=recover)

    p = sub.add_parser("time", help="time `loopmath eval fit` per spec and engine, as its own process")
    fit_options(p)
    p.add_argument("--engines", type=csv, default=["pymc"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--cores", type=int, default=int(RS.GATE_CORES), help="cores one fit may use (default 2)")
    p.set_defaults(run=time_fits)

    p = sub.add_parser("stats", help="recovery statistics and the gate, from the records")
    p.add_argument("--out", required=True)
    p.add_argument("--engine", default=None, help="engines, comma separated (default: every engine with records)")
    p.add_argument("--catalog", type=csv, default=None, help="narrow the gate's catalog specs (needs --scope-note)")
    p.add_argument("--repeated", type=csv, default=None,
                   help="narrow the gate's repeated binary and levels specs (needs --scope-note)")
    p.add_argument("--studies", type=int, default=RS.GATE_STUDIES, help="studies per spec for coverage")
    p.add_argument("--catalog-studies", type=int, default=None,
                   help="studies per spec for the clean-fit items (default: --studies; fewer than 200 needs --scope-note)")
    p.add_argument("--coverage-specs", type=int, default=RS.GATE_SPECS, help="specs whose coverage must pass")
    p.add_argument("--scope-note", default=None, help="who approved a narrower scope, and why")
    p.set_defaults(run=stats)

    p = sub.add_parser("compare", help="two engines side by side on the studies both fitted")
    p.add_argument("--out", required=True)
    p.add_argument("--engines", type=csv, required=True)
    p.set_defaults(run=compare)
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.run(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
