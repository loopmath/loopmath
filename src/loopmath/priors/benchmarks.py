"""Published benchmark results (lane 11): `priors/benchmarks.toml` (D36 layout, D50 coverage).

The file has `[[benchmark]]` tables (`id`, `title`, `metric`, `kind` success,
cost or tokens (D54), `reference` model, `reference_effort`, `harness`, `url`, `note`) and
`[[result]]` tables (`benchmark`, `model`, `effort`, `value`, `harness`, `date`,
`url`, `note`). Lane 5's `belief.priors.benchmark_factors` turns them into prior
factors on version nodes (spec 04 section 5); it reads the first result of the
reference model as the reference, so that row is the one at `reference_effort`.

This module loads the file and checks it: every value is a published number
with its own source line (D50), success values are fractions, the reference
model has a result, and only the 12 current models appear. A model with no
result anywhere is listed by `missing_models`, never filled in.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from ..ingest.base import EFFORTS

BENCHMARKS_TOML = Path(__file__).resolve().parent / "benchmarks.toml"

# D50: the current models in the packaged price table (with lane 2's additions).
MODELS = (
    "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5", "claude-sonnet-5", "haiku-4.5",
    "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
)
KINDS = ("success", "cost", "tokens")  # D36, D54
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BENCH_KEYS = ("id", "title", "metric", "kind", "reference", "harness", "url")
_RESULT_KEYS = ("benchmark", "model", "value", "harness", "date", "url")


def load_benchmarks(path: str | Path | None = None) -> dict:
    """`{"benchmark": [...], "result": [...]}` from the TOML file (the packaged one by default)."""
    data = tomllib.loads(Path(path or BENCHMARKS_TOML).read_text(encoding="utf-8"))
    return {"benchmark": list(data.get("benchmark") or []), "result": list(data.get("result") or [])}


def check_benchmarks(data: dict) -> list[str]:
    """Problems with a loaded file (empty when it is right)."""
    problems: list[str] = []
    benches: dict[str, dict] = {}
    for i, b in enumerate(data.get("benchmark") or []):
        where = f"benchmark {b.get('id') or i}"
        problems += [f"{where}: no {k}" for k in _BENCH_KEYS if not b.get(k)]
        if b.get("kind") and b["kind"] not in KINDS:
            problems.append(f"{where}: kind {b['kind']!r} is not one of {', '.join(KINDS)}")
        if b.get("id") in benches:
            problems.append(f"{where}: duplicate id")
        benches[str(b.get("id"))] = b
    seen: set[tuple] = set()
    first_ref: dict[str, dict] = {}
    for i, r in enumerate(data.get("result") or []):
        where = f"result {i + 1} ({r.get('benchmark')}, {r.get('model')}, {r.get('effort')})"
        problems += [f"{where}: no {k}" for k in _RESULT_KEYS if r.get(k) in (None, "")]
        bench = benches.get(str(r.get("benchmark")))
        if bench is None:
            problems.append(f"{where}: unknown benchmark")
            continue
        if r.get("model") not in MODELS:
            problems.append(f"{where}: model is not one of the 12 current models (D50)")
        if r.get("effort") is not None and r["effort"] not in EFFORTS:
            problems.append(f"{where}: effort {r['effort']!r} is not one of {sorted(EFFORTS)}")
        value = r.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append(f"{where}: value is not a number")
        elif bench.get("kind") == "success" and not 0 <= value <= 1:
            problems.append(f"{where}: a success value is a fraction in [0, 1]")
        elif bench.get("kind") in ("cost", "tokens") and not value > 0:
            problems.append(f"{where}: a {bench['kind']} value is positive")
        if r.get("date") and not _DATE.match(str(r["date"])):
            problems.append(f"{where}: date is not YYYY-MM-DD")
        if r.get("url") and not str(r["url"]).startswith("https://"):
            problems.append(f"{where}: url is not https")
        key = (r.get("benchmark"), r.get("model"), r.get("effort"))
        if key in seen:
            problems.append(f"{where}: duplicate result")
        seen.add(key)
        if r.get("model") == bench.get("reference"):
            first_ref.setdefault(str(r.get("benchmark")), r)
    for bid, b in benches.items():
        ref = first_ref.get(bid)
        if ref is None:
            problems.append(f"benchmark {bid}: no result for its reference model {b.get('reference')}")
        elif b.get("reference_effort") and ref.get("effort") != b["reference_effort"]:
            problems.append(f"benchmark {bid}: the first {b.get('reference')} result is at effort "
                            f"{ref.get('effort')!r}, not the reference effort {b['reference_effort']!r}")
    return problems


def coverage(data: dict) -> dict[str, list[str]]:
    """Model to the benchmark ids that have a result for it, for the 12 current models."""
    out: dict[str, list[str]] = {m: [] for m in MODELS}
    for r in data.get("result") or []:
        ids = out.get(r.get("model"))
        if ids is not None and r.get("benchmark") not in ids:
            ids.append(r["benchmark"])
    return out


def missing_models(data: dict) -> list[str]:
    """Current models with no published result on any benchmark in the file (listed in the report, D50)."""
    return [m for m, ids in coverage(data).items() if not ids]
