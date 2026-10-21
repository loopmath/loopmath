"""Synthetic OpenEvolve program databases, written from reading OpenEvolve's code (never from running it).

OpenEvolve is read only: no test imports it, installs it or runs it. Paths and lines below are relative to
`openevolve/` at commit 34427bae6239063d46be0d38e0fabe11279d05aa of
https://github.com/algorithmicsuperintelligence/openevolve (Apache 2.0). README.md in this folder gathers them.
The code, prompts and change notes inside the programs are loopmath's own marker strings, so a test can check that
none of them reaches a row or a run file.

`program(...)` makes one `Program` record as `asdict` writes it, `write_db(...)` writes a database folder as
`ProgramDatabase.save` does, `write_trace(...)` writes evolution trace lines, and `build(...)` writes a small run
whose expected rows are known by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Text markers: present in every free-text field, never allowed in a row or a run file.
SECRET = "OEFIXTURE-PRIVATE-TEXT"
T0 = 1_777_000_000.0   # epoch seconds (database.py:79 `timestamp: float = field(default_factory=time.time)`)


def program(pid: str, *, parent: str | None, generation: int, iteration: int, metrics: dict[str, float],
            metadata: dict[str, Any] | None = None, timestamp: float | None = None) -> dict[str, Any]:
    """One `Program` as `asdict` writes it (database.py:65-100, 102-104)."""
    return {
        "id": pid,                                            # database.py:69
        "code": f"def solve():  # {SECRET} code {pid}",       # database.py:70
        "changes_description": f"{SECRET} changes {pid}",     # database.py:71
        "language": "python",                                 # database.py:74
        "parent_id": parent,                                  # database.py:77
        "generation": generation,                             # database.py:78
        "timestamp": T0 + iteration * 60.0 if timestamp is None else timestamp,   # database.py:79
        "iteration_found": iteration,                         # database.py:80
        "metrics": dict(metrics),                             # database.py:83
        "complexity": 0.0,                                    # database.py:86
        "diversity": 0.0,                                     # database.py:87
        "metadata": dict(metadata or {}),                     # database.py:90
        "prompts": None,                                      # database.py:93
        "artifacts_json": None,                               # database.py:96
        "artifact_dir": None,                                 # database.py:97
        "embedding": None,                                    # database.py:100
    }


def usage(model: str, prompt: int, completion: int) -> dict[str, Any]:
    """`last_usage` of the OpenAI-compatible client (llm/openai.py:235-253); `model` is `model_cfg.name` (54)."""
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "model": model}


def child_metadata(parent: dict[str, Any], *, island: int, token_usage: dict[str, Any] | None) -> dict[str, Any]:
    """A child's metadata (process_parallel.py:373-389): changes, the parent's metrics, island, token usage (217)."""
    return {"changes": f"{SECRET} diff summary", "parent_metrics": dict(parent["metrics"]), "island": island,
            "token_usage": token_usage}


def migrant(original: dict[str, Any], new_id: str, *, island: int) -> dict[str, Any]:
    """A migrant copy (database.py:2114-2140): new id, parent = the original, same generation and metrics, the
    original's metadata plus island and `migrant: True` (so it carries the original's token usage)."""
    copy = program(new_id, parent=original["id"], generation=original["generation"],
                   iteration=original["iteration_found"], metrics=original["metrics"],
                   metadata={**original["metadata"], "island": island, "migrant": True},
                   timestamp=original["timestamp"] + 1.0)
    return copy


def write_db(folder: Path, programs: list[dict[str, Any]], *, last_iteration: int, islands: int = 2,
             prompts: dict[str, dict[str, Any]] | None = None, features: tuple[str, ...] = ("complexity", "diversity"),
             ) -> Path:
    """A database folder as `save` writes it (database.py:726-776): `programs/<id>.json` per program
    (`_save_program`, 942-977, with `prompts` when `log_prompts`), and `metadata.json` (760-774). `feature_stats`
    is keyed by each configured feature dimension (database.py:771, 979-1026, 2429-2445; default dimensions
    complexity and diversity, config.py:343-350)."""
    progs = folder / "programs"
    progs.mkdir(parents=True, exist_ok=True)
    for p in programs:
        d = dict(p)
        if prompts and p["id"] in prompts:
            d["prompts"] = prompts[p["id"]]
        (progs / f"{p['id']}.json").write_text(json.dumps(d), encoding="utf-8")
    best = max(programs, key=lambda p: next(iter(p["metrics"].values()), 0.0)
               if "combined_score" not in p["metrics"] else p["metrics"]["combined_score"])["id"]
    meta = {
        "island_feature_maps": [{} for _ in range(islands)],
        "islands": [[p["id"] for p in programs if p["metadata"].get("island", 0) == i] for i in range(islands)],
        "archive": [best],
        "best_program_id": best,
        "island_best_programs": [best] * islands,
        "last_iteration": last_iteration,
        "current_island": 0,
        "island_generations": [0] * islands,
        "last_migration_generation": 0,
        "feature_stats": {f: {"min": 0.0, "max": 1.0, "values": []} for f in features},
    }
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


def checkpoint_dir(output_dir: Path, iteration: int) -> Path:
    """`<output_dir>/checkpoints/checkpoint_<iteration>` (controller.py:446-501)."""
    return output_dir / "checkpoints" / f"checkpoint_{iteration}"


def trace_line(parent: dict[str, Any], child: dict[str, Any], *, island: int) -> dict[str, Any]:
    """One `EvolutionTrace` line (evolution_trace.py:24-60), as logged for an admitted child whose parent is still
    present (process_parallel.py:776-802). `to_dict` drops None fields (evolution_trace.py:47-49); the prompt and
    the response are in by default (config.py:425 `include_prompts: bool = True`)."""
    pm, cm = parent["metrics"], child["metrics"]
    return {
        "iteration": child["iteration_found"], "timestamp": child["timestamp"], "parent_id": parent["id"],
        "child_id": child["id"], "parent_metrics": pm, "child_metrics": cm,
        "prompt": {"system": f"{SECRET} trace system", "user": f"{SECRET} trace user"},
        "llm_response": f"{SECRET} trace response",
        "improvement_delta": {k: cm[k] - pm[k] for k in cm if k in pm},
        "island_id": island, "generation": child["generation"],
        "metadata": {"iteration_time": 41.0, "changes": f"{SECRET} trace changes",
                     "token_usage": child["metadata"].get("token_usage")},
    }


def write_trace(path: Path, lines: list[dict[str, Any]]) -> Path:
    """`evolution_trace.jsonl` (config.py:419-430 format jsonl; controller.py:151-170 output path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return path


@dataclass
class Expect:
    """What the exporter should make of one child."""
    child: str
    model: str | None
    improved: bool
    delta: float
    tokens: tuple[int, int] | None    # (input, output)


@dataclass
class Built:
    output_dir: Path            # `<initial program dir>/openevolve_output` (controller.py:59-60)
    db: Path                    # the newest checkpoint folder
    expect: dict[str, Expect] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)   # program id: why it is not a row
    last_iteration: int = 0
    older: Path | None = None   # checkpoint_5: r0, c1, c2


VARIANTS = frozenset({
    "no-usage",     # one child from a provider with no last_usage (claude_code, copilot_cli, codex_app_server)
    "root-evicted", # the newest checkpoint lost the root: only the best and the new program are protected
                    # (database.py:1916-1990); checkpoint_5 still has it
    "trace",        # also write evolution_trace.jsonl in the output folder
    "prompts",      # log_prompts on: `prompts` saved with each child (database.py:971-973)
})


def build(root: Path, *, task: str = "circle_packing", variants: frozenset[str] | set[str] = frozenset()) -> Built:
    """A run with a root, five children, one migrant, and one child whose parent was evicted.

    root r0 0.50; c1 (r0, gpt-5-mini) 0.55 improved; c2 (r0, gemini-2.5-flash) 0.40 not; c3 (c1, flash) 0.55
    equal, so not; c4 (parent e9, evicted; parent_metrics 0.30, gpt-5-mini) 0.35 improved; m1 migrant of c1.
    With "no-usage", c5 (c2, no token usage) 0.45 improved."""
    unknown = set(variants) - VARIANTS
    assert not unknown, unknown
    output_dir = root / task / "openevolve_output"
    r0 = program("r0", parent=None, generation=0, iteration=0, metrics={"combined_score": 0.50})  # controller.py:275-300
    c1 = program("c1", parent="r0", generation=1, iteration=1, metrics={"combined_score": 0.55},
                 metadata=child_metadata(r0, island=0, token_usage=usage("gpt-5-mini", 1000, 200)))
    c2 = program("c2", parent="r0", generation=1, iteration=2, metrics={"combined_score": 0.40},
                 metadata=child_metadata(r0, island=1, token_usage=usage("gemini-2.5-flash", 1100, 150)))
    c3 = program("c3", parent="c1", generation=2, iteration=3, metrics={"combined_score": 0.55},
                 metadata=child_metadata(c1, island=0, token_usage=usage("gemini-2.5-flash", 1200, 180)))
    e9 = program("e9", parent="r0", generation=1, iteration=0, metrics={"combined_score": 0.30})  # evicted, not saved
    c4 = program("c4", parent="e9", generation=2, iteration=4, metrics={"combined_score": 0.35},
                 metadata=child_metadata(e9, island=1, token_usage=usage("gpt-5-mini", 900, 210)))
    m1 = migrant(c1, "m1", island=1)
    progs = [r0, c1, c2, c3, c4, m1]
    b = Built(output_dir=output_dir, db=output_dir)
    b.expect = {
        "c1": Expect("c1", "gpt-5-mini", True, 0.05, (1000, 200)),
        "c2": Expect("c2", "gemini-2.5-flash", False, -0.10, (1100, 150)),
        "c3": Expect("c3", "gemini-2.5-flash", False, 0.0, (1200, 180)),
        "c4": Expect("c4", "gpt-5-mini", True, 0.05, (900, 210)),
    }
    b.skipped = {"r0": "root", "m1": "migrant"}
    if "no-usage" in variants:
        c5 = program("c5", parent="c2", generation=2, iteration=5, metrics={"combined_score": 0.45},
                     metadata=child_metadata(c2, island=1, token_usage=None))
        progs.append(c5)
        b.expect["c5"] = Expect("c5", None, True, 0.05, None)
    last = 10   # iterations ran past the children found: failed iterations and evictions (process_parallel.py:744-756)
    prompts = None
    if "prompts" in variants:
        prompts = {p["id"]: {"diff_user": {"system": f"{SECRET} system", "user": f"{SECRET} user",
                                           "responses": [f"{SECRET} response"]}}
                   for p in progs if p["parent_id"]}
    newest = [p for p in progs if p["id"] != "r0"] if "root-evicted" in variants else progs
    b.db = write_db(checkpoint_dir(output_dir, last), newest, last_iteration=last, prompts=prompts)
    b.older = write_db(checkpoint_dir(output_dir, 5), [r0, c1, c2], last_iteration=5)   # older; read on request
    b.last_iteration = last
    if "trace" in variants:
        by = {p["id"]: p for p in progs}
        lines = [trace_line(by[c["parent_id"]], c, island=c["metadata"]["island"])
                 for c in progs if c["parent_id"] in by and not c["metadata"].get("migrant")]
        write_trace(output_dir / "evolution_trace.jsonl", lines)
    return b


def build_features(root: Path, *, with_stats: bool = True) -> Path:
    """A database with no `combined_score`: fitness is the mean of the metrics that are not feature dimensions
    (utils/metrics_utils.py:72-118). `spread` is a feature dimension. r0 (fitness 0.5, spread 50); p1 (r0; 1.0, 100);
    c1 (p1; 2.0, 0). By OpenEvolve's fitness c1 beats p1 by +1.0 and p1 beats r0 by +0.5; an average over all
    metrics would say c1 fell by 49.5. Without `with_stats`, metadata.json names no feature dimensions."""
    r0 = program("r0", parent=None, generation=0, iteration=0, metrics={"fitness": 0.5, "spread": 50.0})
    p1 = program("p1", parent="r0", generation=1, iteration=1, metrics={"fitness": 1.0, "spread": 100.0},
                 metadata=child_metadata(r0, island=0, token_usage=usage("gpt-5-mini", 800, 100)))
    c1 = program("c1", parent="p1", generation=2, iteration=2, metrics={"fitness": 2.0, "spread": 0.0},
                 metadata=child_metadata(p1, island=0, token_usage=usage("gpt-5-mini", 900, 120)))
    folder = root / "features" / "openevolve_output"
    return write_db(checkpoint_dir(folder, 2), [r0, p1, c1], last_iteration=2,
                    features=("spread",) if with_stats else ())


def price_table(path: Path, rates: dict[str, tuple[float, float]]) -> Path:
    """A price table in the packaged format (src/loopmath/prices.toml; loopmath.price.load_prices): $/Mtok."""
    lines = ['as_of = "2026-10-20"', ""]
    for model, (inp, out) in rates.items():
        lines += [f'["{model}"]', f"input = {inp}", f"cache_read = {inp / 10}", f"cache_write = {inp}",
                  f"output = {out}", 'source = "test table"', ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
