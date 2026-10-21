"""OpenEvolve program databases as harness-runs rows (experimental): one `mutator` row per child program.

Written from reading OpenEvolve's code (https://github.com/algorithmicsuperintelligence/openevolve at 34427bae,
paths relative to `openevolve/`), never from running it; docs/adapters.md section "OpenEvolve and ShinkaEvolve"
lists the fields. What it reads:

- A database folder (`ProgramDatabase.save`, database.py:726-776): `programs/<id>.json`, one `Program` each
  (database.py:65-100), and `metadata.json` with `last_iteration` and `feature_stats`. Checkpoints are
  `<output>/checkpoints/checkpoint_<iteration>/` (controller.py:446-501); `database.db_path` keeps every admitted
  program the same way (database.py:506).
- An output folder (`<initial program dir>/openevolve_output`, controller.py:59-60): its newest checkpoint, plus
  `evolution_trace.jsonl` when the trace was on (evolution_trace.py:24-60; off by default, config.py:419-427).
- A trace file alone.

A child is a program with a parent (process_parallel.py:373-389). Its model and tokens are `metadata.token_usage`
(llm/openai.py:235-253, `model` from the model config's name), set only by OpenAI-compatible providers. Its score is
`combined_score`, else the mean of its numeric metrics that are not feature dimensions (utils/metrics_utils.py:72-118),
with the dimensions read from `metadata.json` `feature_stats` (database.py:771, 979-1026). With neither known, a
program with more than one metric is refused unless `--metric` names the score. The parent's score is the parent's
record, else the `parent_metrics` the child carries (an evicted parent).
Migrant copies (database.py:2114-2140) and the root are not rows. Failed iterations are never stored
(process_parallel.py:744-756), and past `population_size` the worst programs are evicted (database.py:1916-1990), so a
checkpoint keeps survivors; the notes say how many iterations ran and how many children were found.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .evolve import (FormatError, NeedsFlag, Result, TablePrice, check_label, experiment_id, experiments, improved,
                     iso, mutator_row, number, plural, task_of)

TOOL = "openevolve"
NS = "dev.loopmath.adapter.openevolve"
OUTPUT = "openevolve_output"
TRACE = "evolution_trace.jsonl"
CHECKPOINT = re.compile(r"^checkpoint_(\d+)$")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"{path}: not readable JSON ({exc})") from None


def _is_db(folder: Path) -> bool:
    return (folder / "programs").is_dir() and (folder / "metadata.json").is_file()


def _newest_checkpoint(output: Path) -> Path | None:
    found = []
    folder = output / "checkpoints"
    if folder.is_dir():
        for p in folder.iterdir():
            m = CHECKPOINT.match(p.name)
            if m and _is_db(p):
                found.append((int(m.group(1)), p))
    return max(found)[1] if found else None


def _layout(path: Path) -> tuple[Path | None, Path | None, Path | None]:
    """(database folder, trace file, output folder) for PATH."""
    if path.is_file():
        if path.suffix != ".jsonl":
            raise FormatError(f"{path}: not an OpenEvolve trace (a .jsonl file) or folder")
        return None, path, None
    if _is_db(path):
        output = path.parent.parent if path.parent.name == "checkpoints" else None
        trace = output / TRACE if output is not None and (output / TRACE).is_file() else None
        return path, trace, output
    db = _newest_checkpoint(path)
    trace = path / TRACE if (path / TRACE).is_file() else None
    if db is None and trace is None:
        raise FormatError(f"no OpenEvolve database under {path}: no programs/ with metadata.json, no "
                          f"checkpoints/checkpoint_<n>/, no {TRACE}")
    return db, trace, path


def _task_from(output: Path | None, db: Path | None, trace: Path | None) -> str:
    if output is not None:
        return output.parent.name if output.name == OUTPUT else output.name
    if trace is not None:
        return trace.parent.parent.name if trace.parent.name == OUTPUT else trace.parent.name
    assert db is not None
    return db.name


def _score(metrics: Any, metric: str | None, features: set[str] | None, who: str) -> float | None:
    """The fitness OpenEvolve ranks by (utils/metrics_utils.py:72-118), or METRIC. `features` is None when the
    feature dimensions are unknown (a trace alone, or no `feature_stats`)."""
    if not isinstance(metrics, dict):
        return None
    if metric is not None:
        if metric not in metrics:
            raise NeedsFlag(f"--metric {metric!r}: not among the metrics of {who} "
                             f"({', '.join(sorted(metrics)) or 'none'})")
        return number(metrics[metric])
    if "combined_score" in metrics:
        return number(metrics["combined_score"])
    numeric = {k: number(v) for k, v in metrics.items()}
    numeric = {k: v for k, v in numeric.items() if v is not None}
    if features is None and len(numeric) > 1:
        raise NeedsFlag(f"{who} has no combined_score and the feature dimensions are unknown, so OpenEvolve's "
                         f"fitness (the mean of the metrics that are not feature dimensions) cannot be rebuilt "
                         f"from {', '.join(sorted(numeric))}; give --metric NAME, or a checkpoint whose "
                         f"metadata.json has feature_stats")
    values = [v for k, v in numeric.items() if k not in (features or set())] or list(numeric.values())
    return sum(values) / len(values) if values else None


def _programs(db: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    meta = _load_json(db / "metadata.json")
    if not isinstance(meta, dict):
        raise FormatError(f"{db / 'metadata.json'}: not an OpenEvolve database metadata object")
    programs: dict[str, dict[str, Any]] = {}
    for f in sorted((db / "programs").glob("*.json")):
        p = _load_json(f)
        if not (isinstance(p, dict) and isinstance(p.get("id"), str) and "metrics" in p):
            raise FormatError(f"{f}: not an OpenEvolve program (no id and metrics)")
        programs[p["id"]] = p
    if not programs:
        raise FormatError(f"{db / 'programs'}: no programs")
    return programs, meta


def _trace(path: Path) -> list[dict[str, Any]]:
    lines = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise FormatError(f"{path}: not readable ({exc})") from None
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            raise FormatError(f"{path}:{n}: not JSON") from None
        if not (isinstance(x, dict) and isinstance(x.get("child_id"), str) and "child_metrics" in x):
            raise FormatError(f"{path}:{n}: not an OpenEvolve trace line (no child_id and child_metrics)")
        lines.append(x)
    return lines


def _child_from_program(p: dict[str, Any]) -> dict[str, Any]:
    md = p.get("metadata") if isinstance(p.get("metadata"), dict) else {}
    return {"id": p["id"], "parent_id": p.get("parent_id"), "generation": p.get("generation"),
            "iteration": p.get("iteration_found"), "timestamp": p.get("timestamp"), "metrics": p.get("metrics"),
            "parent_metrics": md.get("parent_metrics"), "island": md.get("island"),
            "token_usage": md.get("token_usage"), "seconds": None, "source": "checkpoint"}


def _child_from_trace(x: dict[str, Any]) -> dict[str, Any]:
    md = x.get("metadata") if isinstance(x.get("metadata"), dict) else {}
    return {"id": x["child_id"], "parent_id": x.get("parent_id"), "generation": x.get("generation"),
            "iteration": x.get("iteration"), "timestamp": x.get("timestamp"), "metrics": x.get("child_metrics"),
            "parent_metrics": x.get("parent_metrics"), "island": x.get("island_id"),
            "token_usage": md.get("token_usage"), "seconds": number(md.get("iteration_time")), "source": "trace"}


def _root_id(programs: dict[str, dict[str, Any]]) -> str | None:
    """The initial program (controller.py:275-300): no parent, generation 0, not a migrant. Eviction protects only
    the best and the newest program (database.py:1916-1990), so a later checkpoint may have lost it."""
    roots = sorted(pid for pid, p in programs.items()
                   if p.get("parent_id") is None and (p.get("generation") or 0) == 0
                   and not (isinstance(p.get("metadata"), dict) and p["metadata"].get("migrant")))
    if len(roots) > 1:   # OpenEvolve adds one; never let the sort order pick the experiment
        shown = ", ".join(roots[:3]) + (", ..." if len(roots) > 3 else "")
        raise NeedsFlag(f"{len(roots)} programs with no parent at generation 0 ({shown}), so the experiment id is "
                        f"ambiguous; give --experiment")
    return roots[0] if roots else None


def _priced(price: TablePrice | None, model: str | None, tokens: dict[str, int]) -> float | None:
    """Dollars for `tokens` at `model`'s rates in the caller's table, or None when it has no row for the model.
    OpenEvolve records no cache split, so every input token is priced as uncached input."""
    if price is None or not model:
        return None
    usd = price.usd(model, {"in": tokens.get("input", 0), "cache_read": 0, "cache_write": 0,
                            "out": tokens.get("output", 0)})
    return None if usd is None else round(float(usd), 6)


def read(path: str | Path, *, experiment: str | None = None, task: str | None = None, model: str | None = None,
         metric: str | None = None, min_delta: float = 0.0, price: TablePrice | None = None,
         task_type: str | None = None, repo: str | None = None) -> Result:
    """Every child under an OpenEvolve output folder, database folder or trace, as checked harness-runs rows.
    Reads only. `price` is a price table the caller loaded (`adapt openevolve --prices`), which prices the tokens
    of the models it lists. Raises FormatError for an input OpenEvolve did not write, and ValueError when children
    name no model and `model` is not given, when `metric` is missing from a program or needed and not given, or
    when the root program is absent or ambiguous and `experiment` is not given."""
    from ..harness_runs.rows import check_row

    check_label(experiment)
    path = Path(path)
    db, trace_path, output = _layout(path)
    programs, meta = _programs(db) if db is not None else ({}, {})
    lines = _trace(trace_path) if trace_path is not None else []
    stats = meta.get("feature_stats")
    features = set(stats) if isinstance(stats, dict) and stats else None
    fmt = "openevolve-checkpoint+trace" if db is not None and lines else \
        "openevolve-checkpoint" if db is not None else "openevolve-trace"
    result = Result(rows=[], format=fmt)
    children: dict[str, dict[str, Any]] = {}
    counts = {"programs": len(programs), "children": 0, "roots": 0, "migrants": 0, "from_trace": 0,
              "no_parent_score": 0, "priced_by_table": 0}
    for pid, p in sorted(programs.items()):
        md = p.get("metadata") if isinstance(p.get("metadata"), dict) else {}
        if p.get("parent_id") is None:
            counts["roots"] += 1
            result.skipped.append({"run_id": pid, "reason": "root"})
        elif md.get("migrant"):
            counts["migrants"] += 1
            result.skipped.append({"run_id": pid, "reason": "migrant"})
        else:
            children[pid] = _child_from_program(p)
    for x in lines:
        if x["child_id"] not in children and x["child_id"] not in programs:
            children[x["child_id"]] = _child_from_trace(x)
            counts["from_trace"] += 1
        elif x["child_id"] in children and children[x["child_id"]]["seconds"] is None:
            children[x["child_id"]]["seconds"] = number((x.get("metadata") or {}).get("iteration_time"))
    nameless = [c["id"] for c in children.values()
                if not (isinstance(c["token_usage"], dict) and isinstance(c["token_usage"].get("model"), str)
                        and c["token_usage"]["model"])]
    if nameless and model is None:
        raise NeedsFlag(f"{plural(len(nameless), 'child', 'children')} of {len(children)} name no model "
                         f"(OpenEvolve records the model only with OpenAI-compatible providers' token usage); "
                         f"give --model NAME, the model that wrote them")
    if experiment is not None:
        exp = experiment
    elif (root := _root_id(programs)) is not None:
        exp = experiment_id("oe", root)
    else:
        why = "a trace holds no root" if db is None else f"{db.name} no longer holds it (evicted)"
        raise NeedsFlag(f"the experiment id comes from the root program, and {why}; give --experiment with the "
                         f"id an earlier export of this run printed (or any label of your own, used every time), "
                         f"so the same child keeps the same id")
    task_id = task or _task_from(output, db, trace_path)
    task_obj = task_of(task_id, task_type, repo)
    metric_name = metric or "combined_score"
    for cid in sorted(children, key=lambda k: (number(children[k]["iteration"]) or 0.0, k)):
        c = children[cid]
        child_score = _score(c["metrics"], metric, features, f"child {cid}")
        parent = programs.get(c["parent_id"]) if c["parent_id"] else None
        if parent is not None:
            parent_score = _score(parent.get("metrics"), metric, features, f"parent {c['parent_id']}")
        else:
            parent_score = _score(c["parent_metrics"], metric, features, f"the parent metrics of {cid}")
        if child_score is None or parent_score is None:
            counts["no_parent_score"] += 1
            result.skipped.append({"run_id": cid, "reason": "no parent score" if child_score is not None
                                   else "no score"})
            continue
        ok, delta = improved(child_score, parent_score, min_delta)
        usage = c["token_usage"] if isinstance(c["token_usage"], dict) else None
        named = usage.get("model") if usage and isinstance(usage.get("model"), str) and usage["model"] else None
        cost: dict[str, Any]
        if usage is not None and isinstance(usage.get("prompt_tokens"), int):
            cost = {"tokens": {"input": max(0, usage["prompt_tokens"]),
                               "output": max(0, int(usage.get("completion_tokens") or 0))}}
            usd = _priced(price, named or model, cost["tokens"])
            if usd is not None:
                cost.update(usd=usd, usd_source="price-table")
                counts["priced_by_table"] += 1
        else:
            cost = {"complete": False, "incomplete": "OpenEvolve recorded no token usage for this child"}
        end = number(c["timestamp"])
        start = end - c["seconds"] if end is not None and c["seconds"] else end
        note = {"unit": "stored child", "parent_id": c["parent_id"], "generation": c["generation"],
                "iteration": c["iteration"],
                "island": c["island"], "source": c["source"], "model_given": None if named else True}
        row = mutator_row(tool=TOOL, ns=NS, experiment=exp, child_id=cid, task_id=task_id, task=task_obj,
                          model=named or model, effort=None, accepted=ok, delta=delta, parent_score=parent_score,
                          child_score=child_score, metric=metric_name, cost=cost, status="done",
                          started=iso(start), ended=iso(end) if c["seconds"] else None, stage_started=iso(start),
                          stage_ended=iso(end) if c["seconds"] else None,
                          stage=f"iteration {c['iteration']}" if c["iteration"] is not None else "child", note=note)
        problems = check_row(row)
        if problems:
            raise FormatError(f"OpenEvolve child {cid} makes a bad row: {'; '.join(problems)}")
        result.rows.append(row)
    counts["children"] = len(result.rows)
    result.counts = counts
    result.experiments = experiments(result.rows)
    last = meta.get("last_iteration") if isinstance(meta.get("last_iteration"), int) else None
    where = f"{db.name}" if db is not None else trace_path.name if trace_path is not None else str(path)
    if last is not None:
        result.notes.append(
            f"{where}: {plural(last, 'iteration')} ran and {plural(len(children), 'child', 'children')} are in the "
            f"database; failed iterations are never stored (process_parallel.py:744-756) and past population_size "
            f"the worst programs are evicted (database.py:1916-1990), so these rows lean to survivors; keep the "
            f"evolution trace (evolution_trace.enabled) or set database.db_path to keep every admitted child")
    if price is not None:
        result.notes.append(f"--prices {price.path} (as of {price.as_of}) priced "
                            f"{plural(counts['priced_by_table'], 'child', 'children')}; the rest are priced from "
                            f"loopmath's own table where it has the model, else left unpriced")
    if model is not None and nameless:
        result.notes.append(f"--model {model} filled {plural(len(nameless), 'child', 'children')} that named none; "
                            f"their cost is unknown")
    return result
