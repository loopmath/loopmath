"""Synthetic ShinkaEvolve results folders, written from reading ShinkaEvolve's code (never from running it).

ShinkaEvolve is read only: no test imports it, installs it or runs it. Paths and lines below are relative to
`shinka/` at commit 8adc053a2ce4511ad2ac310e004c530a73fb974a of https://github.com/SakanaAI/ShinkaEvolve
(Apache 2.0). README.md in this folder gathers them. The code, diffs, feedback, patch names and message text inside
are loopmath's own marker strings, so a test can check that none of them reaches a row or a run file.

`create(path)` makes `programs.sqlite` with the two tables the exporter reads, `insert_program(...)` adds one
program as `add` does, `log_attempt(...)` adds one attempt_log row, `query_record(...)` writes one patch query's
`metadata.json`, and `build(...)` writes a small run whose expected rows are known by hand.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SECRET = "SHFIXTURE-PRIVATE-TEXT"
T0 = 1_777_000_000.0   # epoch seconds (database/dbase.py:167; core/async_runner.py:4172)

# database/dbase.py:423-449 (programs) and 507-523 (attempt_log), column for column.
PROGRAMS_DDL = """CREATE TABLE IF NOT EXISTS programs (
    id TEXT PRIMARY KEY, code TEXT NOT NULL, language TEXT NOT NULL, parent_id TEXT,
    archive_inspiration_ids TEXT, top_k_inspiration_ids TEXT, generation INTEGER NOT NULL,
    timestamp REAL NOT NULL, code_diff TEXT, combined_score REAL, public_metrics TEXT, private_metrics TEXT,
    text_feedback TEXT, complexity REAL, embedding TEXT, embedding_pca_2d TEXT, embedding_pca_3d TEXT,
    embedding_cluster_id INTEGER, correct BOOLEAN DEFAULT 0, children_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT, migration_history TEXT, island_idx INTEGER, system_prompt_id TEXT)"""
ATTEMPT_DDL = """CREATE TABLE IF NOT EXISTS attempt_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, generation INTEGER NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
    details TEXT, created_at REAL NOT NULL)"""


def connect(path: Path) -> sqlite3.Connection:
    """A writer connection in WAL mode, as the database opens itself (database/dbase.py:411)."""
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def create(path: Path) -> Path:
    """`<results_dir>/programs.sqlite` (core/async_runner.py:1192) with its tables (database/dbase.py:420-523)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(connect(path)) as con, con:
        con.execute(PROGRAMS_DDL)
        con.execute(ATTEMPT_DDL)
    return path


def llm_result(model: str, *, input_tokens: int, output_tokens: int, cost: float,
               effort: str | None = None) -> dict[str, Any]:
    """`QueryResult.to_dict()` (llm/providers/result.py:41-59). `kwargs` are the provider call's own kwargs
    (llm/providers/openai.py:207-209), drawn inside `query` (llm/llm.py:619-645), so their `reasoning` effort is the
    one this reply ran with (llm/kwargs.py:143-159). Message text is loopmath's marker."""
    kwargs: dict[str, Any] = {"temperature": 0.5, "max_output_tokens": 4096}
    if effort is not None:
        kwargs["reasoning"] = {"effort": effort, "summary": "auto"}
    return {"content": f"{SECRET} patch reply", "msg": f"{SECRET} user msg", "system_msg": f"{SECRET} system",
            "new_msg_history": [{"role": "user", "content": f"{SECRET} history"}], "model_name": model,
            "kwargs": kwargs, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "thinking_tokens": 0, "cost": cost, "input_cost": cost * 0.4, "output_cost": cost * 0.6,
            "thought": f"{SECRET} thought", "model_posteriors": None, "num_tool_calls": 0, "num_total_queries": 1}


def child_metadata(*, drawn_model: str, drawn_effort: str | None, result: dict[str, Any] | None, api_costs: float,
                   novelty_attempt: int = 1, resample_attempt: int = 1, patch_attempt: int = 1, start: float,
                   end: float) -> dict[str, Any]:
    """A completed child's metadata: `meta_patch_data` (core/async_runner.py:3779-3797) spread with `llm_kwargs`
    drawn once per proposal (3670; llm/kwargs.py:103-175), then the costs, logs and pipeline times added at
    4179-4208 (core/pipeline_timing.py:30-54). The drawn `model_name` and `reasoning` are not the writer's: each
    query draws its own (llm/llm.py:619-645). `api_costs` sums the proposal's queries (2805-2817, 2969)."""
    md: dict[str, Any] = {
        "api_costs": api_costs, "patch_type": "diff", "patch_name": f"{SECRET} name",
        "patch_description": f"{SECRET} description", "num_applied": 1, "error_attempt": None,
        "diff_summary": {"added": 3, "deleted": 1}, "novelty_attempt": novelty_attempt,
        "resample_attempt": resample_attempt, "patch_attempt": patch_attempt, "system_prompt_id": None,
        "model_name": drawn_model, "temperature": 0.5, "max_output_tokens": 4096,
        "llm_result": result,
        "embed_cost": 0.0001, "novelty_cost": 0.0, "stdout_log": f"{SECRET} stdout", "stderr_log": "",
        "results_missing": False, "safe_processing": True,
        "pipeline_started_at": start, "sampling_started_at": start, "sampling_finished_at": end,
        "evaluation_started_at": end, "evaluation_finished_at": end + 30.0,
        "postprocess_started_at": end + 30.0, "postprocess_finished_at": end + 31.0,
    }
    if drawn_effort is not None:
        md["reasoning"] = {"effort": drawn_effort}
    return md


def insert_program(con: sqlite3.Connection, pid: str, *, parent: str | None, generation: int, score: float,
                   correct: bool, metadata: dict[str, Any], island: int = 0, timestamp: float | None = None) -> None:
    """One `programs` row as `add` inserts it (database/dbase.py:866-899); JSON columns are `json.dumps`."""
    con.execute(
        "INSERT INTO programs (id, code, language, parent_id, archive_inspiration_ids, top_k_inspiration_ids, "
        "generation, timestamp, code_diff, combined_score, public_metrics, private_metrics, text_feedback, "
        "complexity, embedding, correct, children_count, metadata, island_idx) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (pid, f"def run():  # {SECRET} code {pid}", "python", parent, "[]", "[]", generation,
         T0 + generation * 60.0 + 50.0 if timestamp is None else timestamp, f"--- {SECRET} diff {pid}", score,
         json.dumps({"combined_score": score}), json.dumps({}), f"{SECRET} feedback", 1.0, "[]", int(correct), 0,
         json.dumps(metadata), island))


def log_attempt(con: sqlite3.Connection, *, generation: int, stage: str, status: str, details: dict[str, Any],
                created_at: float) -> None:
    """One `attempt_log` row (`record_attempt_event`, database/dbase.py:713-735; details sorted JSON)."""
    con.execute("INSERT INTO attempt_log (generation, stage, status, details, created_at) VALUES (?, ?, ?, ?, ?)",
                (generation, stage, status, json.dumps(details, sort_keys=True), created_at))


def failed_proposal(*, parent: str, drawn_model: str, api_costs: float, start: float, end: float) -> dict[str, Any]:
    """`details` of a terminal failed proposal (core/async_runner.py:3307-3398). `model_name` is the proposal's
    draw, not a writer."""
    return {"node_kind": "failed_proposal", "language": "python", "failure_stage": "proposal",
            "failure_class": "patch_apply_failed", "failure_reason": f"{SECRET} reason", "parent_id": parent,
            "archive_inspiration_ids": [], "top_k_inspiration_ids": [], "code_diff_available": False,
            "patch_type": "diff", "patch_name": None, "patch_description": None, "system_prompt_id": None,
            "model_name": drawn_model, "api_costs": api_costs, "embed_cost": 0.0, "novelty_cost": 0.0,
            "novelty_attempt": None, "resample_attempt": None, "patch_attempt": None, "max_similarity": None,
            "failure_json_path": "/srv/results/gen_4/failure.json", "pipeline_started_at": start,
            "sampling_started_at": start, "sampling_finished_at": end, "evaluation_started_at": end,
            "evaluation_finished_at": end, "postprocess_started_at": end, "postprocess_finished_at": end,
            "sampling_worker_id": 0, "evaluation_worker_id": None}


def query_record(results: Path, *, generation: int, novelty: int = 1, resample: int = 1, patch: int = 1,
                 model: str | None, cost: float | None, success: bool, at: float) -> Path:
    """One patch query's record, `_save_patch_attempt_async` (core/async_runner.py:3050-3125):
    `<results>/gen_<g>/attempts/novelty_<n>/resample_<r>/patch_<p>/metadata.json` (FOLDER_PREFIX "gen",
    core/config.py:16). `llm_cost` and `llm_model` are written only when there was a response (3116-3118). The
    `timestamp` is `datetime.now().isoformat()`, naive local time (3113). The reply and patch text files beside
    it carry loopmath's marker."""
    folder = (results / f"gen_{generation}" / "attempts" / f"novelty_{novelty}" / f"resample_{resample}"
              / f"patch_{patch}")
    folder.mkdir(parents=True, exist_ok=True)
    md: dict[str, Any] = {
        "generation": generation, "novelty_attempt": novelty, "resample_attempt": resample, "patch_attempt": patch,
        "success": success, "num_applied": 1 if success else 0, "patch_name": f"{SECRET} name",
        "patch_description": f"{SECRET} description", "error_msg": None if success else f"{SECRET} error",
        "timestamp": datetime.fromtimestamp(at).isoformat(),
    }
    if model is not None:
        md["llm_cost"] = cost
        md["llm_model"] = model
        (folder / "llm_response.txt").write_text(f"{SECRET} reply", encoding="utf-8")
    if success:
        (folder / "patch.txt").write_text(f"{SECRET} patch", encoding="utf-8")
    path = folder / "metadata.json"
    path.write_text(json.dumps(md, indent=2), encoding="utf-8")
    return path


@dataclass
class Expect:
    """One expected call row, keyed `gen-<g>/n<n>r<r>p<p>`."""
    model: str
    usd: float
    improved: bool
    delta: float | None
    applied: bool                      # the stage status: done when the patch applied, failed when not
    child: str | None = None           # the program this query wrote


@dataclass
class ChildExpect:
    """One expected `--children-only` row, keyed by program id."""
    model: str
    usd: float
    tokens: tuple[int, int]
    effort: str | None
    improved: bool
    delta: float | None


@dataclass
class Built:
    results_dir: Path
    db: Path
    expect: dict[str, Expect] = field(default_factory=dict)
    children: dict[str, ChildExpect] = field(default_factory=dict)
    usd_total: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)


def spawned(source: str, *, island: int, strategy: str = "best") -> dict[str, Any]:
    """The metadata `_copy_program_to_island` adds to a spawned copy (database/islands.py:811-817)."""
    return {"_spawned_island": True, "_spawned_from_program_id": source, "_spawn_island_idx": island,
            "_spawn_strategy": strategy}


def build(root: Path, *, task: str = "circle_packing") -> Built:
    """A run with every case the exporter must handle. Each generation is one proposal: its folder is created
    exclusively, and a generation whose folder exists is not proposed again (core/async_runner.py:2604-2634).

    - gen 0: root s0 (0.50), and its own query record. Not a row.
    - gen 1: s1 (s0) 0.60, correct. One query by gpt-5-mini wrote it: improved +0.10. The proposal's draw said
      o4-mini with effort low; the reply ran gpt-5-mini at effort high.
    - gen 2: s2 (s0) 0.45. Query 1 by o4-mini did not apply; query 2 by gemini-2.5-flash wrote it: not improved,
      delta -0.05. The draw said gpt-5-mini.
    - gen 3: s3 (s1) 0.90 but incorrect. Its query by gpt-5-mini did not improve and has no delta.
    - gen 4: a failed proposal: three queries that did not apply (flash, flash, o4-mini) and one with no reply.
    - gen 5: s6 (s1) 0.70. The first query (flash) applied but lost the novelty check; the second novelty attempt
      (gpt-5-mini) wrote s6: improved +0.10.
    - gen 6: in flight: a query record and no program or failure event. Not settled, so not rows.
    - gen 7: a failed proposal logged twice (one query by flash). The generation counts once.
    - gen 8: s8 (s1) 0.65, and a failure event for the same generation. The program wins: Shinka's own count of
      failed generations leaves out generations with a program (core/async_runner.py:6048-6069). Its query by
      flash wrote s8: improved +0.05; a second query by o4-mini did not apply.
    - c4: the island copy of the root s0, with no parent (islands.py:603-660). c5: a spawned island's root, a copy
      of s1 with no parent, and c5a: a copy of s6 under it (islands.py:788-836, 974-1000). Not rows, and not the
      root: their ids sort before s0, and the experiment id still comes from s0.
    """
    results = root / task / "results_20260501_120000"   # default `results_<%Y%m%d_%H%M%S>` (core/async_runner.py:282-286)
    db = create(results / "programs.sqlite")
    b = Built(results_dir=results, db=db)
    t = T0

    def at(g: int, k: int = 0) -> float:
        return t + g * 60.0 + k * 5.0

    q = query_record
    q(results, generation=0, model="gpt-5-mini", cost=0.001, success=True, at=at(0))
    q(results, generation=1, model="gpt-5-mini", cost=0.004, success=True, at=at(1))
    q(results, generation=2, patch=1, model="o4-mini", cost=0.001, success=False, at=at(2))
    q(results, generation=2, patch=2, model="gemini-2.5-flash", cost=0.002, success=True, at=at(2, 1))
    q(results, generation=3, model="gpt-5-mini", cost=0.005, success=True, at=at(3))
    q(results, generation=4, patch=1, model="gemini-2.5-flash", cost=0.001, success=False, at=at(4))
    q(results, generation=4, patch=2, model="gemini-2.5-flash", cost=0.001, success=False, at=at(4, 1))
    q(results, generation=4, patch=3, model="o4-mini", cost=0.0015, success=False, at=at(4, 2))
    q(results, generation=4, resample=2, patch=1, model=None, cost=None, success=False, at=at(4, 3))
    q(results, generation=5, novelty=1, model="gemini-2.5-flash", cost=0.002, success=True, at=at(5))
    q(results, generation=5, novelty=2, model="gpt-5-mini", cost=0.003, success=True, at=at(5, 1))
    q(results, generation=6, model="o4-mini", cost=0.002, success=True, at=at(6))
    q(results, generation=7, model="gemini-2.5-flash", cost=0.001, success=False, at=at(7))
    q(results, generation=8, patch=1, model="o4-mini", cost=0.001, success=False, at=at(8))
    q(results, generation=8, patch=2, model="gemini-2.5-flash", cost=0.002, success=True, at=at(8, 1))

    with contextlib.closing(connect(db)) as con, con:
        insert_program(con, "s0", parent=None, generation=0, score=0.50, correct=True, metadata={})
        insert_program(con, "s1", parent="s0", generation=1, score=0.60, correct=True, metadata=child_metadata(
            drawn_model="o4-mini", drawn_effort="low", api_costs=0.004, start=at(1), end=at(1, 2),
            result=llm_result("gpt-5-mini", input_tokens=2000, output_tokens=300, cost=0.004, effort="high")))
        insert_program(con, "s2", parent="s0", generation=2, score=0.45, correct=True, island=1,
                       metadata=child_metadata(
                           drawn_model="gpt-5-mini", drawn_effort="high", api_costs=0.003, patch_attempt=2,
                           start=at(2), end=at(2, 2),
                           result=llm_result("gemini-2.5-flash", input_tokens=2500, output_tokens=400, cost=0.002)))
        insert_program(con, "s3", parent="s1", generation=3, score=0.90, correct=False, metadata=child_metadata(
            drawn_model="gpt-5-mini", drawn_effort=None, api_costs=0.005, start=at(3), end=at(3, 2),
            result=llm_result("gpt-5-mini", input_tokens=2100, output_tokens=350, cost=0.005)))
        # Copies keep a null parent when they copy a root, and their ids (uuid4 in Shinka) can sort before the root's.
        insert_program(con, "c4", parent=None, generation=0, score=0.50, correct=True, island=1,   # islands.py:603-660
                       metadata={"_is_island_copy": True, "_original_program_id": "s0"})          # dbase.py:976-980
        s1_md = json.loads(con.execute("SELECT metadata FROM programs WHERE id='s1'").fetchone()[0])
        insert_program(con, "c5", parent=None, generation=1, score=0.60, correct=True, island=2,   # islands.py:974-1000
                       metadata={**s1_md, **spawned("s1", island=2)})                             # islands.py:788-836
        log_attempt(con, generation=4, stage="proposal", status="failed", created_at=at(4, 4),
                    details=failed_proposal(parent="s1", drawn_model="gpt-5-mini", api_costs=0.0035,
                                            start=at(4), end=at(4, 4)))
        insert_program(con, "s6", parent="s1", generation=5, score=0.70, correct=True, metadata=child_metadata(
            drawn_model="gemini-2.5-flash", drawn_effort=None, api_costs=0.005, novelty_attempt=2, start=at(5),
            end=at(5, 2), result=llm_result("gpt-5-mini", input_tokens=1900, output_tokens=280, cost=0.003)))
        s6_md = json.loads(con.execute("SELECT metadata FROM programs WHERE id='s6'").fetchone()[0])
        insert_program(con, "c5a", parent="c5", generation=5, score=0.70, correct=True, island=2,  # a subtree child
                       metadata={**s6_md, **spawned("s6", island=2), "_spawned_as_child": True})  # islands.py:817
        for k in (1, 2):
            log_attempt(con, generation=7, stage="proposal", status="failed", created_at=at(7, k),
                        details=failed_proposal(parent="s0", drawn_model="o4-mini", api_costs=0.001,
                                                start=at(7), end=at(7, k)))
        log_attempt(con, generation=8, stage="evaluation_submit", status="failed", created_at=at(8, 2),
                    details=failed_proposal(parent="s1", drawn_model="o4-mini", api_costs=0.003,
                                            start=at(8), end=at(8, 2)))
        insert_program(con, "s8", parent="s1", generation=8, score=0.65, correct=True, metadata=child_metadata(
            drawn_model="o4-mini", drawn_effort=None, api_costs=0.003, patch_attempt=2, start=at(8),
            end=at(8, 2), result=llm_result("gemini-2.5-flash", input_tokens=2300, output_tokens=310, cost=0.002)))

    E = Expect
    b.expect = {
        "gen-1/n1r1p1": E("gpt-5-mini", 0.004, True, 0.10, True, child="s1"),
        "gen-2/n1r1p1": E("o4-mini", 0.001, False, None, False),
        "gen-2/n1r1p2": E("gemini-2.5-flash", 0.002, False, -0.05, True, child="s2"),
        "gen-3/n1r1p1": E("gpt-5-mini", 0.005, False, None, True, child="s3"),
        "gen-4/n1r1p1": E("gemini-2.5-flash", 0.001, False, None, False),
        "gen-4/n1r1p2": E("gemini-2.5-flash", 0.001, False, None, False),
        "gen-4/n1r1p3": E("o4-mini", 0.0015, False, None, False),
        "gen-5/n1r1p1": E("gemini-2.5-flash", 0.002, False, None, True),
        "gen-5/n2r1p1": E("gpt-5-mini", 0.003, True, 0.10, True, child="s6"),
        "gen-7/n1r1p1": E("gemini-2.5-flash", 0.001, False, None, False),
        "gen-8/n1r1p1": E("o4-mini", 0.001, False, None, False),
        "gen-8/n1r1p2": E("gemini-2.5-flash", 0.002, True, 0.05, True, child="s8"),
    }
    C = ChildExpect
    b.children = {
        "s1": C("gpt-5-mini", 0.004, (2000, 300), "high", True, 0.10),
        "s2": C("gemini-2.5-flash", 0.002, (2500, 400), None, False, -0.05),
        "s3": C("gpt-5-mini", 0.005, (2100, 350), None, False, None),
        "s6": C("gpt-5-mini", 0.003, (1900, 280), None, True, 0.10),
        "s8": C("gemini-2.5-flash", 0.002, (2300, 310), None, True, 0.05),
    }
    # One proposal per settled generation: the program's api_costs, else the failure event's (gen 7 once).
    b.usd_total = 0.004 + 0.003 + 0.005 + 0.0035 + 0.005 + 0.001 + 0.003
    b.counts = {"generations": 7, "failed_generations": 2, "unsettled_generations": 1, "no_reply": 1,
                "copies": 3, "generation_0_queries": 1, "unsettled_queries": 1}
    return b
