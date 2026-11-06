"""Prior bundle and benchmark prior factors (spec 04 section 5).

- The shipped bundle enters the fit as data, each run under its `source` node
  (`sweep`, `e0`, `rq1`, `repo_history`). `bundle_docs()` reads it through
  `loopmath.priors.bundle_docs()`; `bundle_entries()` pairs each run with its manifest
  source, so a store run with a shipped label stays the user's.
- Shared imports enter under their `shared:<org>` source through
  `share.import_.shared_runs(home)`.
- Benchmark priors (`priors/benchmarks.toml`, layout D36) become Gaussian prior factors on
  version nodes: success head, the logit gap to the benchmark's reference model with variance
  `1 / (w p (1 - p))`; cost head, the log cost ratio with variance `1 / w`; tokens head
  (`kind = "tokens"`, published total tokens for the evaluation), the log token ratio
  with variance `1 / w`. `w` is one model's share of the benchmark's weight `w_b` (0.2.3):
  the `weight` of its `[[benchmark]]` table (default 5, about 5 runs), shared by
  the model's k results on that benchmark, so each gets `w_b / k` and a model with five
  efforts pulls as hard in total as a model with one. Config `benchmark_prior_weight`, when
  set, is one `w_b` for every benchmark; 0 turns the factors off.
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
    """OCP v0.3 run documents of the shipped prior bundle, or of `bundle_dir`, read by `loopmath.priors`."""
    from .. import priors as bundle

    yield from bundle.bundle_docs(without=tuple(without), directory=bundle_dir)


def bundle_entries(bundle_dir: Path | None = None) -> Iterator[tuple[str, dict]]:
    """(manifest source, document) for every shipped run: the fit's source for it (spec 04 section 1)."""
    from .. import priors as bundle

    yield from bundle.bundle_entries(directory=bundle_dir)


def run_id_of(doc: dict) -> str:
    from .. import priors as bundle

    return bundle.run_id_of(doc)


def bundle_sources(bundle_dir: Path | None = None) -> list[str]:
    """Source names in the shipped prior bundle, or in `bundle_dir` (from its manifest).

    Read from `manifest()`: the package attribute `sources` turns into the `priors.sources`
    submodule once anything imports it, so calling it fails in a full run.
    """
    from .. import priors as bundle

    return list(bundle.manifest(bundle_dir).get("sources") or {})


def shared_docs(home: Path) -> Iterator[dict]:
    """Runs imported from other organizations (`share.import_.shared_runs`)."""
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


def benchmark_weights(path: Path | None = None, *, weight: float | None = None) -> dict[str, float]:
    """Each benchmark's weight `w_b`: `weight` for every benchmark when it is given (config
    `benchmark_prior_weight`), else the benchmark's own `weight` field, else 5. Empty for a missing file."""
    path = path or benchmark_path()
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {str(b["id"]): _weight_of(b, weight) for b in data.get("benchmark") or [] if b.get("id")}


def _weight_of(bench: dict, weight: float | None) -> float:
    if weight is not None:
        return float(weight)
    own = bench.get("weight")
    return float(own) if isinstance(own, (int, float)) and not isinstance(own, bool) else BENCHMARK_PRIOR_WEIGHT


def benchmark_factors(path: Path | None = None, *, weight: float | None = None) -> list[FactorSpec]:
    """Prior factors from `benchmarks.toml`. A missing file gives no factors.

    `weight` (config `benchmark_prior_weight`) overrides every benchmark's own weight; None reads
    each benchmark's `weight`. A model's k results on one benchmark share its weight: each factor
    has weight `w_b / k`. A benchmark with weight 0 or less gives no factors.
    """
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
        w_b = _weight_of(bench, weight)
        if w_b <= 0:
            continue
        rows = [r for r in results if r["benchmark"] == bid]
        ref_rows = [r for r in rows if canonical_model_id(str(r["model"])) == ref]
        if not ref_rows:
            continue
        ref_value = float(ref_rows[0]["value"])
        by_model: dict[str, list[tuple[str, list, float, dict]]] = {}  # model -> (head, terms, mean, row) per result
        for r in rows:
            model = canonical_model_id(str(r["model"]))
            if model == ref:
                continue
            value = float(r["value"])
            terms = _model_terms(model, r.get("effort"), 1.0) + _model_terms(ref, ref_rows[0].get("effort"), -1.0)
            terms = _cancel(terms)
            if not terms:
                continue
            if kind == "success":
                p = min(max(value, 0.01), 0.99)
                p_ref = min(max(ref_value, 0.01), 0.99)
                gap = math.log(p / (1 - p)) - math.log(p_ref / (1 - p_ref))
                by_model.setdefault(model, []).append(("success", terms, gap, r))
            elif kind in ("cost", "tokens") and value > 0 and ref_value > 0:  # Tokens feed the tokens head only
                by_model.setdefault(model, []).append((kind, terms, math.log(value / ref_value), r))
        for model, items in by_model.items():
            w = w_b / len(items)  # one budget per model and benchmark
            share = f"weight {w_b:g} over {len(items)}" if len(items) > 1 else f"weight {w_b:g}"
            for head, terms, mean, r in items:
                note = (f"{bid}: {model} {float(r['value'])} against {ref} {ref_value} ({share}; "
                        f"{r.get('url', '')}, {r.get('date', '')})")
                if head == "success":
                    p = min(max(float(r["value"]), 0.01), 0.99)
                    out.append(FactorSpec("success", terms, mean, 1.0 / (w * p * (1 - p)), note))
                else:
                    out.append(FactorSpec(head, terms, mean, 1.0 / w, note))
    return out


BENCHMARK_RUNS = 100  # a success benchmark's runs when neither its table (`runs`) nor RUNS_BY_BENCHMARK names them
# Task runs behind each published value (tasks x 3 repeats, from each benchmark's `metric`); benchmarks.toml is
# generated, so the counts live here until its generator writes `runs`
RUNS_BY_BENCHMARK = {"aa-terminal-bench-4.0": 198, "aa-terminal-bench-2.1": 267}


def benchmark_offset_factors(path: Path | None = None, *, alpha: float = 1.0, gap_scale: float = 1.0,
                             transfer_sd: float = 0.6, effort_sd: float = 0.0,
                             weight: float | None = None) -> list[FactorSpec]:
    """Success factors by offset and scale (`level.py`), one per model and effort, no reference model.

    A result p of model m at effort e on benchmark b, on its `runs` n, reads
    `S_{m,e} + o_b + t_{b,m} ~ N(c logit(p~), c^2 / (n p~ (1 - p~)) / alpha)`, p~ = (n p + 1/2) / (n + 1):
    `S` the setting's provider, family, model, effort and family x effort nodes, `o_b` the benchmark's free offset
    `bench:<b>`, `t_{b,m}` the model's transfer error `btx:<b>|<m>` with prior variance (c transfer_sd)^2 / alpha
    (`fit.make_prior`), shared by the model's efforts, `c` the gap scale; with `effort_sd` > 0 also the effort's
    own transfer error `btxe:<b>|<m>|<e>`, variance (c effort_sd)^2 / alpha. alpha 0 gives no factors; benchmarks of
    weight 0 give none, as in `benchmark_factors`, and so does `weight` 0 (`benchmark_prior_weight`, which overrides
    each benchmark's own); a weight is not read otherwise. A value that is not a finite number gives no factor, and
    one outside 0 to 1 is read as its bound (the reference path clamps to 0.01 to 0.99)."""
    path = path or benchmark_path()
    if alpha <= 0 or not path.is_file():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: list[FactorSpec] = []
    for bench in data.get("benchmark") or []:
        bid = str(bench.get("id") or "")
        if not bid or str(bench.get("kind") or "success") != "success" or _weight_of(bench, weight) <= 0:
            continue
        own = bench.get("runs")
        n = (float(own) if isinstance(own, (int, float)) and not isinstance(own, bool) and own > 0
             else float(RUNS_BY_BENCHMARK.get(bid, BENCHMARK_RUNS)))
        for r in data.get("result") or []:
            if r.get("benchmark") != bid or not r.get("model"):
                continue
            value = r.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue  # an edited table: no factor rather than no fit
            model = canonical_model_id(str(r["model"]))
            p = (min(max(float(value), 0.0), 1.0) * n + 0.5) / (n + 1)  # a rate outside [0, 1] as its bound
            terms = _cancel(_model_terms(model, r.get("effort"), 1.0)) + [(f"bench:{bid}", None, 1.0)]
            if transfer_sd > 0:
                terms.append((f"btx:{bid}|{model}", None, 1.0))
            if effort_sd > 0:
                terms.append((f"btxe:{bid}|{model}|{r.get('effort') or 'default'}", None, 1.0))
            out.append(FactorSpec("success", terms, gap_scale * math.log(p / (1 - p)),
                                  gap_scale ** 2 / (n * p * (1 - p)) / alpha,
                                  f"{bid}: {model} {r.get('effort') or 'default'} {float(r['value'])} of {n:g} runs "
                                  f"(offset; {r.get('url', '')}, {r.get('date', '')})"))
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
