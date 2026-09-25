"""Prior bundle and benchmark prior factors (spec 04 section 5).

- The shipped bundle (lane 11) enters the fit as data, each run under its `source` node
  (`sweep`, `e0`, `rq1`, `repo_history`). `bundle_docs()` reads it through lane 11's
  `loopmath.priors.bundle_docs()` (D37); `bundle_entries()` pairs each run with its manifest
  source, so a store run with a shipped label stays the user's (D118 N2).
- Shared imports (lane 8) enter under their `shared:<org>` source through
  `share.import_.shared_runs(home)` (D22).
- Benchmark priors (`priors/benchmarks.toml`, layout D36) become Gaussian prior factors on
  version nodes: success head, the logit gap to the benchmark's reference model with variance
  `1 / (w p (1 - p))`; cost head, the log cost ratio with variance `1 / w`; tokens head
  (`kind = "tokens"`, published total tokens for the evaluation, D54), the log token ratio
  with variance `1 / w`. `w` is config `benchmark_prior_weight` (default 5): one benchmark
  result counts like about 5 runs.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .design import setting_terms
from .forest import canonical_model_id

BENCHMARK_PRIOR_WEIGHT = 5.0
PACKAGE_PRIORS = Path(__file__).resolve().parent.parent / "priors"


def bundle_docs(bundle_dir: Path | None = None, without: tuple[str, ...] = ()) -> Iterator[dict]:
    """OCP v0.3 run documents of the shipped prior bundle, or of `bundle_dir` (lane 11's reader, D37)."""
    from .. import priors as lane11

    yield from lane11.bundle_docs(without=tuple(without), directory=bundle_dir)


def bundle_entries(bundle_dir: Path | None = None) -> Iterator[tuple[str, dict]]:
    """(manifest source, document) for every shipped run: the fit's source for it (spec 04 section 1)."""
    from .. import priors as lane11

    yield from lane11.bundle_entries(directory=bundle_dir)


def run_id_of(doc: dict) -> str:
    from .. import priors as lane11

    return lane11.run_id_of(doc)


def bundle_sources(bundle_dir: Path | None = None) -> list[str]:
    """Source names in the shipped prior bundle, or in `bundle_dir` (lane 11's manifest).

    Read from `manifest()`: the package attribute `sources` turns into the `priors.sources`
    submodule once anything imports it, so calling it fails in a full run.
    """
    from .. import priors as lane11

    return list(lane11.manifest(bundle_dir).get("sources") or {})


def shared_docs(home: Path) -> Iterator[dict]:
    """Runs imported from other organizations (lane 8's `shared_runs`, D22)."""
    from ..share.import_ import shared_runs

    yield from shared_runs(home)


@dataclass
class FactorSpec:
    """One Gaussian pseudo-observation: sum(value * b[node]) ~ N(mean, var) on one head."""

    head: str  # "success" | "cost"
    terms: list[tuple[str, str | None, float]]
    mean: float
    var: float
    note: str


def benchmark_path() -> Path:
    return PACKAGE_PRIORS / "benchmarks.toml"


def _model_terms(model: str, effort: str | None, sign: float) -> list[tuple[str, str | None, float]]:
    terms = [t for t in setting_terms("unknown", canonical_model_id(model), effort or "default", sign, False)
             if t[0].split(":", 1)[0] in ("provider", "family", "model")]
    if effort:
        terms += [t for t in setting_terms("unknown", canonical_model_id(model), effort, sign, False)
                  if t[0].split(":", 1)[0] in ("effort", "family_effort")]
    return terms


def benchmark_factors(path: Path | None = None, *, weight: float = BENCHMARK_PRIOR_WEIGHT) -> list[FactorSpec]:
    """Prior factors from `benchmarks.toml` (D36 layout). A missing file gives no factors."""
    path = path or benchmark_path()
    if not path.is_file():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    benches = {str(b.get("id")): b for b in data.get("benchmark") or [] if b.get("id")}
    results = [r for r in data.get("result") or [] if r.get("benchmark") in benches and r.get("model")]
    out: list[FactorSpec] = []
    for bid, bench in benches.items():
        kind = str(bench.get("kind") or "success")
        ref = canonical_model_id(str(bench.get("reference") or ""))
        rows = [r for r in results if r["benchmark"] == bid]
        ref_rows = [r for r in rows if canonical_model_id(str(r["model"])) == ref]
        if not ref_rows:
            continue
        ref_value = float(ref_rows[0]["value"])
        for r in rows:
            model = canonical_model_id(str(r["model"]))
            if model == ref:
                continue
            value = float(r["value"])
            terms = _model_terms(model, r.get("effort"), 1.0) + _model_terms(ref, ref_rows[0].get("effort"), -1.0)
            terms = _cancel(terms)
            if not terms:
                continue
            note = f"{bid}: {model} {value} against {ref} {ref_value} ({r.get('url', '')}, {r.get('date', '')})"
            if kind == "success":
                p = min(max(value, 0.01), 0.99)
                p_ref = min(max(ref_value, 0.01), 0.99)
                gap = math.log(p / (1 - p)) - math.log(p_ref / (1 - p_ref))
                out.append(FactorSpec("success", terms, gap, 1.0 / (weight * p * (1 - p)), note))
            elif kind in ("cost", "tokens") and value > 0 and ref_value > 0:  # D54: tokens feed the tokens head only
                out.append(FactorSpec(kind, terms, math.log(value / ref_value), 1.0 / weight, note))
    return out


def _cancel(terms: list[tuple[str, str | None, float]]) -> list[tuple[str, str | None, float]]:
    """Sum values per node and drop the nodes that cancel (shared provider or family)."""
    acc: dict[str, list] = {}
    for node, parent, value in terms:
        if node in acc:
            acc[node][1] += value
        else:
            acc[node] = [parent, value]
    return [(n, p, v) for n, (p, v) in acc.items() if abs(v) > 1e-12]
