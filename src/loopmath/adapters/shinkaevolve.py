"""ShinkaEvolve results folders as harness-runs rows (experimental): one `mutator` row per patch query.

Written from reading ShinkaEvolve's code (https://github.com/SakanaAI/ShinkaEvolve at 8adc053a, paths relative to
`shinka/`), never from running it; docs/adapters.md section "OpenEvolve and ShinkaEvolve" lists the fields.

- Every patch query writes `<results>/gen_<g>/attempts/novelty_<n>/resample_<r>/patch_<p>/metadata.json`
  (core/async_runner.py:3050-3125), applied or not, with the reply's `llm_model` and `llm_cost` when there was a
  reply. Each such query is one row: its own model and its own dollars. Text beside it (the reply, the patch) is
  never read.
- The query whose attempt numbers are the child's (`meta_patch_data`, core/async_runner.py:3779-3797) and that
  applied wrote the child. Its row carries the child's outcome: improved means correct and above the parent's
  `combined_score`. Every other query of the generation did not improve.
- The proposal's stored `model_name` and `reasoning` are a draw made once per proposal (core/async_runner.py:3670);
  each query draws its own model (llm/llm.py:619-645), so the stored draw is never used as a model or an effort.
- One generation is one proposal: its folder is created exclusively (core/async_runner.py:2604-2634). A generation
  is settled when it has a program row or a failed-proposal event (`attempt_log`, core/async_runner.py:3307-3398);
  only settled generations from 1 on give rows. A generation with a program is a child generation, whatever events
  it also has, as in Shinka's own count (core/async_runner.py:6048-6069).
- Each query row also carries its generation's proposal dollars (`api_costs`) and how many of its queries had no
  reply, so `recommend --role` can say how much its chances leave out. Both are fixed once the generation settles.
- Island and spawned copies (database/islands.py:603-660, 788-836) and the root are not children. A copy of a
  root keeps its null parent: the initial program's island copies (islands.py:603-660, dbase.py:976-980) and a
  spawned island's root (islands.py:974-1000). So the root that names the experiment is the one program with no
  parent that is not a copy, and copies added later never change a row's id; with none or several, the
  experiment id needs `experiment`.

Without the query records (a database copied alone), the reader refuses unless `children_only`, which gives one row
per evaluated child from the applied reply (`llm_result`: its model, dollars, tokens and effort). The unit is then
the evaluated child, not the query.

`programs.sqlite` is opened read only and immutable when no `-wal` file is beside it; with one (a run still going),
a temporary copy is read instead, so the original is never touched. Only the columns needed are selected: never
`code`, `code_diff` or `text_feedback`.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import tempfile
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evolve import (FormatError, NeedsFlag, Result, check_label, experiment_id, experiments, improved,
                     mutator_row, number, plural, task_of)

TOOL = "shinkaevolve"
NS = "dev.loopmath.adapter.shinkaevolve"
DB = "programs.sqlite"
RECORD = re.compile(r"^gen_(\d+)/attempts/novelty_(\d+)/resample_(\d+)/patch_(\d+)/metadata\.json$")


def _layout(path: Path) -> tuple[Path, Path]:
    """(results folder, database) for PATH: the results folder or its `programs.sqlite`."""
    if path.is_dir():
        if (path / DB).is_file():
            return path, path / DB
        raise FormatError(f"no ShinkaEvolve database in {path}: no {DB}")
    if not path.is_file():
        raise FormatError(f"{path}: no such file or folder")
    return path.parent, path


def _connect(db: Path, tmp: Path) -> tuple[sqlite3.Connection, bool]:
    """A read-only connection to `db`, or to a copy of it when a `-wal` file is present. (connection, copied)"""
    wal = db.with_name(db.name + "-wal")
    if wal.exists():
        for suffix in ("", "-wal"):
            src = db.with_name(db.name + suffix)
            if src.exists():
                shutil.copy2(src, tmp / (DB + suffix))
        return sqlite3.connect(tmp / DB), True
    uri = "file:" + urllib.parse.quote(str(db.resolve())) + "?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True), False


def _json(text: Any) -> dict[str, Any]:
    if not isinstance(text, str) or not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _effort(kwargs: Any) -> str | None:
    """`reasoning.effort`, `thinking_level` or `thinking_budget` (llm/kwargs.py:117-175) as an effort label."""
    if not isinstance(kwargs, dict):
        return None
    reasoning = kwargs.get("reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str) and reasoning["effort"]:
        return reasoning["effort"]
    if isinstance(kwargs.get("thinking_level"), str) and kwargs["thinking_level"]:
        return kwargs["thinking_level"]
    budget = kwargs.get("thinking_budget")
    if isinstance(budget, int) and not isinstance(budget, bool):
        return f"budget-{budget}"
    return None


def _task_from(results: Path) -> str:
    """The folder holding the results folder (default `results_<time>`, core/async_runner.py:282-286)."""
    return results.parent.name if results.name.startswith("results_") and results.parent.name else results.name


def _local_iso(text: Any) -> str | None:
    """A query record's `timestamp`, naive local time (core/async_runner.py:3113), as RFC 3339 UTC, read in this
    machine's zone."""
    if not isinstance(text, str):
        return None
    try:
        t = datetime.fromisoformat(text)
    except ValueError:
        return None
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _records(results: Path) -> list[dict[str, Any]]:
    """Every query record under `results`, with its attempt numbers from its path."""
    out = []
    for f in sorted(results.glob("gen_*/attempts/novelty_*/resample_*/patch_*/metadata.json")):
        m = RECORD.match(f.relative_to(results).as_posix())
        if not m:
            continue
        md = _json(f.read_text(encoding="utf-8", errors="replace"))
        g, n, r, p = (int(x) for x in m.groups())
        model = md.get("llm_model") if isinstance(md.get("llm_model"), str) and md["llm_model"] else None
        out.append({"generation": g, "novelty": n, "resample": r, "patch": p, "model": model,
                    "usd": number(md.get("llm_cost")), "success": md.get("success") is True,
                    "at": _local_iso(md.get("timestamp"))})
    return out


def _read_db(db: Path) -> tuple[list[tuple], list[tuple], bool]:
    with tempfile.TemporaryDirectory(prefix="loopmath-shinka-") as tmp:
        try:
            con, copied = _connect(db, Path(tmp))
        except sqlite3.Error as exc:
            raise FormatError(f"{db}: not readable as SQLite ({exc})") from None
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "programs" not in tables:
                raise FormatError(f"{db}: not a ShinkaEvolve database (no programs table)")
            programs = con.execute("SELECT id, parent_id, generation, combined_score, correct, metadata "
                                   "FROM programs").fetchall()
            events = con.execute("SELECT id, generation, details, created_at FROM attempt_log WHERE status = 'failed' "
                                 "ORDER BY created_at, id").fetchall() if "attempt_log" in tables else []
        except sqlite3.Error as exc:
            raise FormatError(f"{db}: not a ShinkaEvolve database ({exc})") from None
        finally:
            con.close()
    return programs, events, copied


def read(path: str | Path, *, experiment: str | None = None, task: str | None = None, min_delta: float = 0.0,
         children_only: bool = False, task_type: str | None = None, repo: str | None = None) -> Result:
    """Every answered patch query of a ShinkaEvolve run (or, with `children_only`, every evaluated child) as checked
    harness-runs rows. Reads only. Raises FormatError for an input ShinkaEvolve did not write, and ValueError when
    the query records are missing and `children_only` is not set, or the root is missing and `experiment` is not
    given."""
    from ..harness_runs.rows import check_row

    check_label(experiment)
    results, db = _layout(Path(path))
    programs, events, copied = _read_db(db)
    by_id: dict[str, dict[str, Any]] = {}
    children: dict[int, list[dict[str, Any]]] = {}
    roots: list[str] = []
    result = Result(rows=[], format="shinkaevolve-children" if children_only else "shinkaevolve-queries")
    counts = {"programs": len(programs), "copies": 0, "generations": 0, "failed_generations": 0,
              "failure_events": 0, "unsettled_generations": 0, "unsettled_queries": 0, "generation_0_queries": 0,
              "queries": 0, "no_reply": 0, "writer_mismatch": 0, "children_without_query": 0,
              "children_without_reply": 0}
    for pid, parent_id, generation, score, correct, meta_text in programs:
        md = _json(meta_text)
        prog = {"id": pid, "parent_id": parent_id, "generation": _int(generation), "score": number(score),
                "correct": bool(correct), "md": md}
        by_id[pid] = prog
        if md.get("_is_island_copy") or md.get("_spawned_island"):   # first: a copy of a root has no parent either
            counts["copies"] += 1
            result.skipped.append({"run_id": pid, "reason": "island_copy"})
        elif parent_id is None:
            roots.append(pid)
            result.skipped.append({"run_id": pid, "reason": "root"})
        elif prog["generation"] is not None and prog["generation"] >= 1:
            children.setdefault(prog["generation"], []).append(prog)
    failure: dict[int, dict[str, Any]] = {}
    for _eid, generation, details_text, _created in events:
        d = _json(details_text)
        if d.get("node_kind") == "failed_proposal" and _int(generation) is not None and generation >= 1:
            counts["failure_events"] += 1
            failure[generation] = d               # the last event of a generation
    settled = set(children) | set(failure)
    counts["generations"] = len(settled)
    counts["failed_generations"] = len(set(failure) - set(children))

    if experiment is not None:
        exp = experiment
    elif len(roots) == 1:
        exp = experiment_id("shinka", roots[0])
    elif not roots:
        raise NeedsFlag(f"{db}: no root program (a program with no parent that is not an island or spawned copy), "
                        f"so the experiment id is unknown; give --experiment")
    else:
        shown = ", ".join(sorted(roots)[:3]) + (", ..." if len(roots) > 3 else "")
        raise NeedsFlag(f"{db}: {len(roots)} programs with no parent that are not island or spawned copies ({shown}), "
                        f"so the experiment id is ambiguous; give --experiment")
    task_id = task or _task_from(results)
    task_obj = task_of(task_id, task_type, repo)

    def outcome(child: dict[str, Any]) -> tuple[bool, float | None, float | None, float | None]:
        parent = by_id.get(child["parent_id"])
        parent_score = parent["score"] if parent else None
        if not child["correct"]:
            return False, None, parent_score, None
        ok, delta = improved(child["score"], parent_score, min_delta)
        return ok, delta, parent_score, child["score"]

    def add(row: dict[str, Any], what: str) -> None:
        problems = check_row(row)
        if problems:
            raise FormatError(f"ShinkaEvolve {what} makes a bad row: {'; '.join(problems)}")
        result.rows.append(row)

    usd_side = 0.0
    gen_usd: dict[int, float] = {}
    for g in sorted(settled):
        md = children[g][0]["md"] if g in children else failure[g]
        gen_usd[g] = number(md.get("api_costs")) or 0.0
        usd_side += (number(md.get("embed_cost")) or 0.0) + (number(md.get("novelty_cost")) or 0.0)
    usd_total = sum(gen_usd.values())

    if children_only:
        for g in sorted(children):
            for child in sorted(children[g], key=lambda c: c["id"]):
                llm = child["md"].get("llm_result") if isinstance(child["md"].get("llm_result"), dict) else None
                model = llm.get("model_name") if llm and isinstance(llm.get("model_name"), str) else None
                if not model:
                    counts["children_without_reply"] += 1
                    result.skipped.append({"run_id": child["id"], "reason": "no reply record"})
                    continue
                ok, delta, parent_score, child_score = outcome(child)
                usd = number(llm.get("cost"))
                cost: dict[str, Any] = ({"usd": max(0.0, usd), "usd_source": TOOL} if usd is not None else
                                        {"complete": False, "incomplete": "the reply recorded no cost"})
                tin, tout = _int(llm.get("input_tokens")), _int(llm.get("output_tokens"))
                if tin is not None and tout is not None and tin >= 0 and tout >= 0:
                    cost["tokens"] = {"input": tin, "output": tout}
                note = {"unit": "evaluated child", "generation": g, "parent_id": child["parent_id"],
                        "correct": child["correct"], "patch_attempt": _int(child["md"].get("patch_attempt"))}
                add(mutator_row(tool=TOOL, ns=NS, experiment=exp, child_id=child["id"], task_id=task_id,
                                task=task_obj, model=model, effort=_effort(llm.get("kwargs")), accepted=ok,
                                delta=delta, parent_score=parent_score, child_score=child_score,
                                metric="combined_score", cost=cost, status="done", started=None, ended=None,
                                stage_started=None, stage_ended=None, stage=f"gen {g}", note=note),
                    f"child {child['id']}")
        attributed = sum(r["stages"][0]["cost"].get("usd") or 0.0 for r in result.rows)
        result.notes.append(
            f"--children-only: one row per evaluated child, so each model's chance is per evaluated child, not per "
            f"query; {plural(counts['failed_generations'], 'failed proposal')} and the queries that wrote no child "
            f"are not rows, and their dollars are unattributed")
    else:
        records = _records(results)
        if not records:
            raise NeedsFlag(f"{results}: no per-query records (gen_<g>/attempts/.../metadata.json), so the model "
                             f"and dollars of each query are unknown; read the whole results folder, or give "
                             f"--children-only for one row per evaluated child (the applied reply's model and cost)")
        wrote: set[str] = set()
        no_reply = Counter(rec["generation"] for rec in records if rec["model"] is None)
        for rec in sorted(records, key=lambda x: (x["generation"], x["novelty"], x["resample"], x["patch"])):
            g = rec["generation"]
            if g == 0:
                counts["generation_0_queries"] += 1
                continue
            if g not in settled:
                counts["unsettled_queries"] += 1
                continue
            if rec["model"] is None:
                counts["no_reply"] += 1
                continue
            counts["queries"] += 1
            child = next((c for c in children.get(g, []) if rec["success"]
                          and (_int(c["md"].get("novelty_attempt")), _int(c["md"].get("resample_attempt")),
                               _int(c["md"].get("patch_attempt"))) == (rec["novelty"], rec["resample"], rec["patch"])),
                         None)
            if child is not None:
                wrote.add(child["id"])
                ok, delta, parent_score, child_score = outcome(child)
                llm = child["md"].get("llm_result") if isinstance(child["md"].get("llm_result"), dict) else {}
                if isinstance(llm.get("model_name"), str) and llm["model_name"] != rec["model"]:
                    counts["writer_mismatch"] += 1
            else:
                ok, delta, parent_score, child_score = False, None, None, None
            cost = ({"usd": max(0.0, rec["usd"]), "usd_source": TOOL} if rec["usd"] is not None else
                    {"complete": False, "incomplete": "the query record has no llm_cost"})
            rid = f"gen-{g}/n{rec['novelty']}r{rec['resample']}p{rec['patch']}"
            note = {"unit": "query", "generation": g, "novelty_attempt": rec["novelty"],
                    "resample_attempt": rec["resample"], "patch_attempt": rec["patch"], "applied": rec["success"],
                    "wrote_child": child is not None, "child_id": child["id"] if child else None,
                    "parent_id": child["parent_id"] if child else None,
                    "correct": child["correct"] if child else None,
                    "failed_proposal": True if g not in children else None,
                    "generation_usd": round(gen_usd[g], 6), "generation_no_reply": no_reply[g]}
            add(mutator_row(tool=TOOL, ns=NS, experiment=exp, child_id=rid, task_id=task_id, task=task_obj,
                            model=rec["model"], effort=None, accepted=ok, delta=delta, parent_score=parent_score,
                            child_score=child_score, metric="combined_score", cost=cost,
                            status="done" if rec["success"] else "failed", started=None, ended=rec["at"],
                            stage_started=None, stage_ended=rec["at"], stage=f"gen {g}", note=note), f"query {rid}")
        counts["children_without_query"] = sum(1 for g in children for c in children[g] if c["id"] not in wrote)
        attributed = sum(r["stages"][0]["cost"].get("usd") or 0.0 for r in result.rows)
        if counts["children_without_query"]:
            result.notes.append(f"{plural(counts['children_without_query'], 'child', 'children')} have no record of "
                                f"the query that wrote them, so their outcome is in no row")
        if counts["writer_mismatch"]:
            result.notes.append(f"{plural(counts['writer_mismatch'], 'query', 'queries')} name another model than "
                                f"the reply stored on the child they wrote; the query's own record is used")
        result.notes.append("query record times are naive local time, read in this machine's zone")
        result.notes.append(
            f"each model's chance is per answered query in a settled generation: "
            f"{plural(counts['no_reply'], 'query', 'queries')} with no reply and "
            f"{plural(counts['unsettled_queries'], 'query', 'queries')} in unsettled generations are outside it, "
            f"and ${usd_total - attributed:.6f} of the settled proposals' dollars is in no row")
    counts["unsettled_generations"] = len(_unsettled(results, settled))
    if counts["unsettled_generations"]:
        result.notes.append(f"{plural(counts['unsettled_generations'], 'unsettled generation')} (no program and no "
                            f"failure event: still running, or stopped mid-proposal) are not rows")
    if copied:
        result.notes.append(f"{db.name}-wal was present (the run may still be going), so a copy was read")
    if counts["failure_events"] > counts["failed_generations"]:
        result.notes.append(f"{plural(counts['failure_events'], 'failure event')} cover "
                            f"{plural(counts['failed_generations'], 'failed generation')}; each generation is one "
                            f"proposal (core/async_runner.py:2604-2634), so each counts once")
    result.accounting = {"usd_total": round(usd_total, 6), "usd_attributed": round(attributed, 6),
                         "usd_unattributed": round(usd_total - attributed, 6),
                         "usd_embed_novelty": round(usd_side, 6)}
    result.counts = counts
    result.experiments = experiments(result.rows)
    return result


def _unsettled(results: Path, settled: set[int]) -> set[int]:
    """Generations from 1 on with a folder of query records but no program row and no failure event."""
    out = set()
    for f in results.glob("gen_*/attempts"):
        m = re.match(r"^gen_(\d+)$", f.parent.name)
        if m and int(m.group(1)) >= 1 and int(m.group(1)) not in settled:
            out.add(int(m.group(1)))
    return out
