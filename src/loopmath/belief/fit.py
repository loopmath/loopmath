"""Fit every head from the store and the prior bundle; write fits/<fit>/.

Spec 04 section 4. `fit()` reads finished OCP v0.3 runs (the store, the shipped bundle,
shared imports), builds the rows of every head, sets the level scales by empirical Bayes,
and writes `fits/<fit>.partial/{state.npz, meta.json, design.json}`, renamed to
`fits/<fit>/` when complete; `fits/latest` then points at it. A non-blocking `flock` on
`fits/.fit.lock` stops two fits running together (lane 7's trigger has its own job lock on
`fits/.lock`, so the two never wait on each other).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import time
import tomllib
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from scipy import sparse

from ..taskmodel import FeatureConfigError, FeatureSet
from . import design as design_rows
from . import priors as prior_data
from .design import (DESIGN_VERSION, Term, Unusable, cost_row, gate_row, parse_run, run_row, structure,
                     task_features, task_terms)
from .features import FeatureTally, add_horizon, drop_features, horizon_nodes, horizon_specs
from .block import leaf_split
from .forest import Forest, default_scale, fixed_sd, scale_group
from .pricing import PRICE_NODE, PRICE_PRIOR_SD, STREAMS, fit_offsets, price_term
from .gaussian import N_DRAWS, Factors, GaussianHead, HeadFit, LogisticHead, Prior
from .state import SCORE_CLAMP, inverse_transform, transform  # noqa: F401  (re-exported)

FIT_LOCK = ".fit.lock"
KEEP_FITS = 5  # The newest fits kept, besides the ones a receipt or stored recommendation names
TASK_LEVELS = ("org", "type", "repo", "subtype", "task")
MIN_SCORE_RUNS = 3  # spec 04 section 2: a score head needs at least 3 measured runs
HEAD_KIND = {"cost": ("gaussian", "cost"), "tokens": ("gaussian", "cost"),
             "gate": ("logistic", "logit"), "success": ("logistic", "logit")}


class FitBusy(RuntimeError):
    """Another fit holds the fit lock."""


class UnknownSource(ValueError):
    """`--without` named a source that matches no run; no fit was written."""


class NothingToFit(ValueError):
    """No head has a row, so no fit was written and `fits/latest` did not move."""


class UnknownFit(LookupError):
    """`--fit ID` named no complete fit in the store."""


FIXED_SOURCES = ("benchmark", "shared", "user")

SHIPPED_COPY = "shipped copy of a stored run"

REASON_NOTES = {
    "attempt without usable cost": "crashed or no usage reported",
    "attempt outside the workflow": "a model attempt at no piece of the run's workflow",
    "duplicate run id": "the same run found twice",
    SHIPPED_COPY: "your store has the same run id; your copy is used",
    "no configuration": "the run records no workflow and settings",
    "workflow ref not resolved": "a workflow this version does not know",
    "workflow has no pieces": "an empty workflow",
    "piece without a setting": "a piece with no harness, model or effort",
}


def reason_note(reason: str) -> str | None:
    """Plain words for a dropped reason in meta.json, for the terminal summary."""
    if reason in REASON_NOTES:
        return REASON_NOTES[reason]
    if reason.startswith("without "):
        return "left out by --without"
    if reason.startswith("unreadable: "):
        return "not a readable OCP v0.3 run"
    if reason.startswith("score ") and reason.endswith(" runs"):
        return "too few runs for a score head"
    if reason.startswith("score "):
        return "a value outside the score's scale"
    return None


def dropped_text(dropped: dict[str, int], top: int = 4) -> str:
    """`dropped N: reason (n: plain words); ...`, largest first."""
    parts = []
    for reason, n in sorted(dropped.items(), key=lambda kv: -kv[1])[:top]:
        note = reason_note(reason)
        parts.append(f"{reason} ({n}: {note})" if note else f"{reason} ({n})")
    return f"dropped {sum(dropped.values())}: " + "; ".join(parts)


def unknown_sources(without: Iterable[str], seen: Iterable[str] = (),
                    bundle_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """(names in `without` that match no source, the known names).

    Known: the bundle's sources, `benchmark`, `user`, `shared` (every `shared:<org>`), and any
    source in `seen`. A name is fine when it is known or matches a seen source.
    """
    seen = set(seen)
    known = sorted(set(prior_data.bundle_sources(bundle_dir)) | set(FIXED_SOURCES) | seen)
    unknown = [w for w in without if w not in known and not any(_excluded(src, (w,)) for src in seen)]
    return unknown, known


def unknown_source_message(unknown: list[str], known: list[str]) -> str:
    return f"unknown source {', '.join(unknown)}; known: {', '.join(known)}"


def _nothing_to_fit(no_prior: bool, without: tuple[str, ...], dropped: dict[str, int]) -> str:
    if no_prior:
        why = "--no-prior leaves only your finished runs, and none is usable yet"
    elif without:
        why = "--without " + " ".join(without) + " left no usable runs"
    else:
        why = "no usable runs in your store or in the prior bundle (see `loopmath doctor`)"
    real = {k: v for k, v in dropped.items() if not k.startswith("without ")}
    return (f"nothing to fit: {why}" + (f"; {dropped_text(real)}" if real else "")
            + "; no fit written, fits/latest unchanged")


# ---------------------------------------------------------------- reading documents

def store_docs(home: Path) -> Iterator[dict]:
    """Finished run documents in the store, through lane 7's reader (it decides what is finished)."""
    from ..store.home import Store

    yield from Store(home).finished_docs()


def _read_config(home: Path) -> dict:
    path = Path(home) / "config.toml"
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


# ---------------------------------------------------------------- rows

@dataclass
class HeadRows:
    name: str
    engine: str  # "gaussian" | "logistic"
    kind: str  # "cost" | "logit" | "score"
    rows: list[list[Term]] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    w: list[float] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    not_repriced: int = 0  # cost rows on recorded dollars, not the current tariff
    prices: dict = field(default_factory=dict)  # cost head: the fit's price offsets (pricing.fit_offsets)

    def add(self, terms, y, w, run, source, ttype):
        self.rows.append(terms)
        self.y.append(y)
        self.w.append(w)
        self.runs.append(run)
        self.sources.append(source)
        self.types.append(ttype)


def _excluded(source: str, without: tuple[str, ...]) -> bool:
    for w in without:
        if source == w or (w == "shared" and source.startswith("shared:")):
            return True
    return False


def collect_rows(docs: Iterable[dict | tuple[str | None, dict]], *, without: tuple[str, ...] = (),
                 now: datetime | None = None, features: FeatureSet | None = None):
    """Rows of every head from run documents, plus counts of what was used and dropped.

    `docs` yields documents, or `(origin, document)` pairs from `fit()`: `user` for the store
    (read first), the manifest source for a shipped run, None to read the document's label
    (spec 04 section 1). A shipped run whose id is in the store is left out as
    `shipped copy of a stored run`, and a shipped source named in `without` is counted
    without being parsed.

    Returns (heads, score_info, dropped, runs_by_source, config_support, checks,
    shipped_overlap, user_labels, tally); `checks` counts the non-model check attempts by check name,
    which are neither evidence nor dropped; `shipped_overlap` counts, per shipped source,
    the user's runs that are also in it; `user_labels` counts the user's runs by the source
    their document names, when that is not `user`. Features are read with `features` (the
    built-ins by default); only admitted values keep their `feature:` terms, and `tally`
    (features.FeatureTally) holds the support, the admitted values and the recorded horizons.
    """
    now = now or datetime.now().astimezone()
    features = features or FeatureSet()
    tally = FeatureTally(features)
    heads = {name: HeadRows(name, *HEAD_KIND[name]) for name in HEAD_KIND}
    score_raw: dict[str, list] = defaultdict(list)
    score_meta: dict[str, Counter] = defaultdict(Counter)
    score_info: dict[str, dict] = {}
    dropped: Counter = Counter()
    checks: Counter = Counter()
    runs_by_source: Counter = Counter()
    config_support: dict[str, Counter] = defaultdict(Counter)
    seen: set[str] = set()
    stored: set[str] = set()  # run ids of the user's store, whose copy wins
    overlap: Counter = Counter()
    labels: Counter = Counter()
    model_runs: dict[str, set[str]] = defaultdict(set)  # the price offset's inputs (spec 04 section 2)
    model_streams: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(STREAMS))
    for item in docs:
        origin, doc = item if isinstance(item, tuple) else (None, item)
        if origin is not None and _excluded(origin, without):
            if origin != "user" and prior_data.run_id_of(doc) in stored:
                overlap[origin] += 1
            dropped[f"without {origin}"] += 1  # counted, not parsed; `--without user` frees the shipped copies
            continue
        if origin == "user":
            stored.add(prior_data.run_id_of(doc))
        elif origin is not None:
            if prior_data.run_id_of(doc) in stored:
                overlap[origin] += 1
                dropped[SHIPPED_COPY] += 1
                continue
        try:
            pr = parse_run(doc, now=now, origin=origin)
        except Unusable as exc:
            dropped[str(exc)] += 1
            continue
        except (KeyError, TypeError, ValueError) as exc:
            dropped[f"unreadable: {type(exc).__name__}"] += 1
            continue
        if pr.run_id in seen:
            dropped["duplicate run id"] += 1
            continue
        seen.add(pr.run_id)
        if _excluded(pr.source, without):
            dropped[f"without {pr.source}"] += 1
            continue
        if pr.source == "user" and pr.task.extra.get("source_label"):
            labels[pr.task.extra["source_label"]] += 1
        for reason, n in pr.dropped.items():
            dropped[reason] += n
        checks.update(pr.checks)
        runs_by_source[pr.source] += 1
        tally.add(pr.task, task_features(pr.task, features), pr.run_id, pr.source)
        chain = [n for n, _, _ in task_terms(pr.task, pr.source) if n.split(":", 1)[0] in TASK_LEVELS]
        config_support[pr.config.id].update(chain + ["all"])
        st = structure(pr.config)
        ttype = pr.task.type
        for a in pr.attempts:
            terms = cost_row(pr.task, pr.source, st, a.piece, a.round, a.setting, features)
            if a.usd:
                heads["cost"].add(terms, math.log(a.usd), a.weight, pr.run_id, pr.source, ttype)
                heads["cost"].not_repriced += not a.repriced
                model_runs[a.setting.model].add(pr.run_id)
                acc = model_streams[a.setting.model]
                for i, v in enumerate(a.streams):
                    acc[i] += v
            if a.tokens:
                heads["tokens"].add(terms, math.log(a.tokens), a.weight, pr.run_id, pr.source, ttype)
        for gi, k, passed in pr.gates:
            heads["gate"].add(gate_row(pr.task, pr.source, st, st.gates[gi], k, features), 1.0 if passed else 0.0,
                              1.0, pr.run_id, pr.source, ttype)
        row = run_row(pr.task, pr.source, st, features)
        ev = pr.evidence
        if ev is not None and ev.z is not None:
            heads["success"].add(row, float(ev.z), float(ev.q), pr.run_id, pr.source, ttype)
        for name, value in (ev.scores if ev else {}).items():
            meta = pr.score_meta.get(name) or {}
            score_meta[name][(meta.get("scale") or "linear", meta.get("better") or "higher", meta.get("unit"))] += 1
            score_raw[name].append((row, float(value), pr.run_id, pr.source, ttype))
    for name, items in score_raw.items():
        scale, better, unit = score_meta[name].most_common(1)[0][0]
        hr = HeadRows(f"score:{name}", "gaussian", "score")
        for row, value, run, source, ttype in items:
            t = transform(value, scale)
            if t is None or not math.isfinite(t):
                dropped[f"score {name} outside its scale"] += 1
                continue
            hr.add(row, t, 1.0, run, source, ttype)
        if len(set(hr.runs)) >= MIN_SCORE_RUNS:
            heads[hr.name] = hr
            score_info[name] = {"scale": scale, "better": better, "unit": unit}
        else:
            dropped[f"score {name} below {MIN_SCORE_RUNS} runs"] += len(hr.runs)
    tally.nodes, tally.report = tally.admit()
    coding = tally.horizon_coding()  # the horizon terms need every run's horizon first (spec 04 section 1)
    for hr in heads.values():
        hr.rows = add_horizon(drop_features(hr.rows, tally.nodes), hr.runs, tally.run_horizon, coding)
    add_prices(heads["cost"], {m: len(r) for m, r in model_runs.items()}, model_streams)
    return (heads, score_info, dict(dropped), dict(runs_by_source), {c: dict(n) for c, n in config_support.items()},
            dict(checks), dict(overlap), dict(labels), tally)


def add_prices(hr: HeadRows, runs: dict[str, int], streams: dict[str, list[float]]) -> None:
    """The price offset (spec 04 section 2): the fit's offsets from its runs and token mix under the current price
    table, and each cost row with its model's `price:offset` term (`pricing.price_term`, as at prediction)."""
    hr.prices = fit_offsets(runs, streams, design_rows._price_table())
    offsets = hr.prices["models"]
    if not offsets:
        return
    rows = []
    for row in hr.rows:
        model = next((n[len("model:"):] for n, _, _ in row if n.startswith("model:")), None)
        term = price_term(offsets, model) if model else ()
        rows.append(row + list(term) if term else row)
    hr.rows = rows


def price_spec(hr: HeadRows) -> list:
    """The prior that holds `price:offset` at 1, on the cost head of a fit with offsets, --no-prior included."""
    if not hr.prices.get("models"):
        return []
    return [prior_data.FactorSpec("cost", [(PRICE_NODE, None, 1.0)], 1.0, PRICE_PRIOR_SD ** 2,
                                  f"{PRICE_NODE} held at 1, prior N(1, {PRICE_PRIOR_SD}^2)")]


# ---------------------------------------------------------------- matrices and priors

def build_matrix(rows: list[list[Term]], forest: Forest, extra_nodes: Iterable[tuple[str, str | None]] = ()):
    local: dict[str, int] = {}
    ri, ci, vals = [], [], []
    for r, terms in enumerate(rows):
        for node, parent, value in terms:
            forest.add(node, parent)
            j = local.get(node)
            if j is None:
                j = local[node] = len(local)
            ri.append(r)
            ci.append(j)
            vals.append(value)
    for node, parent in extra_nodes:
        forest.add(node, parent)
        if node not in local:
            local[node] = len(local)
    X = sparse.csr_matrix((vals, (ri, ci)), shape=(len(rows), len(local)))
    X.sum_duplicates()
    return X, list(local)


def make_prior(node_ids: list[str], kind: str) -> Prior:
    groups: list[str] = []
    gidx = np.full(len(node_ids), -1, dtype=np.int64)
    fvar = np.ones(len(node_ids))
    for j, node in enumerate(node_ids):
        g = scale_group(node)
        if g is None:
            fvar[j] = fixed_sd(node, kind) ** 2
            continue
        if g not in groups:
            groups.append(g)
        gidx[j] = groups.index(g)
    default = np.array([default_scale(g, kind) for g in groups])
    return Prior(gidx, fvar, groups, default)


def make_factors(specs: list, node_ids: list[str]) -> Factors | None:
    if not specs:
        return None
    index = {n: j for j, n in enumerate(node_ids)}
    ri, ci, vals = [], [], []
    for r, spec in enumerate(specs):
        for node, _parent, value in spec.terms:
            ri.append(r)
            ci.append(index[node])
            vals.append(value)
    F = sparse.csr_matrix((vals, (ri, ci)), shape=(len(specs), len(node_ids)))
    return Factors(F, np.array([s.mean for s in specs]), np.array([s.var for s in specs]))


def _support(hr: HeadRows, X: sparse.csr_matrix, node_ids: list[str]) -> dict[str, list]:
    """Per node: [distinct runs, {source: runs}]."""
    runs = np.array(hr.runs, dtype=object)
    sources = np.array(hr.sources, dtype=object)
    Xc = X.tocsc()
    out = {}
    for j, node in enumerate(node_ids):
        rows = Xc.indices[Xc.indptr[j]:Xc.indptr[j + 1]]
        if len(rows) == 0:
            out[node] = [0, {}]
            continue
        pairs = set(zip(runs[rows], sources[rows]))
        out[node] = [len(pairs), dict(Counter(s for _, s in pairs))]
    return out


def seed_for(fit_id: str, head: str) -> int:
    """A head's seed from the fit's `seed_key` (its fit id before 0.2.1)."""
    return int.from_bytes(hashlib.sha256(f"{fit_id}/{head}".encode()).digest()[:4], "little")


def input_key(heads_rows: dict[str, HeadRows], specs: list, settings: dict) -> str:
    """The fit's `seed_key` (spec 04 section 3): the sha256 of the design version, the settings that change the
    posterior, the prior factors and every head's rows (terms, response, weight, run, source), each head's rows
    sorted, so the same data and settings seed the same draws whatever the store's reading order."""
    h = hashlib.sha256()
    h.update(repr((DESIGN_VERSION, design_rows.TIMEBOX_EFFORT_COST, design_rows.TIMEBOX_LEVELS,
                   sorted(settings.items()))).encode())
    for spec in specs:
        h.update(repr((spec.head, spec.terms, spec.mean, spec.var)).encode())
    for name in sorted(heads_rows):
        hr = heads_rows[name]
        h.update(f"\n{name}:{len(hr.rows)}".encode())
        for line in sorted(repr((t, y, w, r, src)) for t, y, w, r, src in zip(hr.rows, hr.y, hr.w, hr.runs, hr.sources)):
            h.update(line.encode())
    return h.hexdigest()


def fit_head(hr: HeadRows, forest: Forest, factor_specs: list, *, eb: bool = True) -> dict:
    """Fit one head; returns arrays and metadata for the state files."""
    extra = [(n, p) for spec in factor_specs for n, p, _ in spec.terms]
    X, node_ids = build_matrix(hr.rows, forest, extra)
    prior = make_prior(node_ids, hr.kind)
    factors = make_factors(factor_specs, node_ids)
    # leaves first (block.py): no two of the leading nodes share a row or a factor, so the stored
    # precision's Cholesky factor is the block factor, rebuilt in O(c^3) by the state
    aX = abs(X)
    pattern = aX.T @ aX + (abs(factors.F).T @ abs(factors.F) if factors is not None else 0)
    perm, _ = leaf_split(pattern)
    X = X[:, perm].tocsr()
    node_ids = [node_ids[i] for i in perm]
    prior = Prior(prior.group_idx[perm], prior.fixed_var[perm], prior.groups, prior.default_phi)
    if factors is not None:
        factors = Factors(factors.F[:, perm].tocsr(), factors.f, factors.v)
    y = np.array(hr.y, float)
    w = np.array(hr.w, float)
    center, scale = 0.0, 1.0
    if hr.kind == "score":
        center = float(np.average(y, weights=w))
        sd = float(np.sqrt(np.average((y - center) ** 2, weights=w)))
        scale = sd if sd > 1e-9 else 1.0
        y = (y - center) / scale
    if hr.engine == "gaussian":
        res: HeadFit = GaussianHead(X, y, w, prior, factors).fit(eb=eb)
    else:
        res = LogisticHead(X, y, w, prior, factors).fit(eb=eb)
    baseline = float(np.mean(X @ res.mean)) if X.shape[0] else 0.0
    return {"name": hr.name, "engine": hr.engine, "kind": hr.kind, "node_ids": node_ids, "fit": res,
            "phi": {g: float(v) for g, v in zip(prior.groups, res.phi)}, "center": center, "scale": scale,
            "baseline": baseline, "support": _support(hr, X, node_ids), "n_rows": X.shape[0],
            "n_runs": len(set(hr.runs)), "factors": [s.note for s in factor_specs],
            "design": (X, y, w, prior, factors)}  # for the PyMC check (fit --full)


# ---------------------------------------------------------------- the fit

@contextmanager
def fit_lock(home: Path, wait_s: float = 0.0):
    fits = Path(home) / "fits"
    fits.mkdir(parents=True, exist_ok=True)
    fd = os.open(fits / FIT_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise FitBusy("another fit is running") from None
                time.sleep(0.2)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield
    finally:
        os.close(fd)


def new_fit_id(fits: Path, now: datetime) -> str:
    t = now
    while True:
        fid = "fit_" + t.strftime("%Y%m%d%H%M%S")
        if not (fits / fid).exists() and not (fits / f"{fid}.partial").exists():
            return fid
        t += timedelta(seconds=1)


def _code_version() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except ImportError:
        return "unknown"


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def point_latest(fits: Path, fit_id: str) -> None:
    tmp = fits / ".latest.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(fit_id, tmp)
    os.replace(tmp, fits / "latest")


def fit(home: Path, *, no_prior: bool = False, without: tuple[str, ...] = (), full: bool = False,
        now: datetime | None = None, docs: Iterable[dict] | None = None, bundle_dir: Path | None = None,
        benchmarks: Path | None = None, eb: bool = True, wait_s: float = 0.0) -> Path:
    """Build the belief and write `fits/<fit>/`; returns its path.

    Raises FitBusy, or UnknownSource / NothingToFit before anything is written.
    """
    home = Path(home)
    fits = home / "fits"
    started = time.monotonic()
    now = now or datetime.now().astimezone()
    without = tuple(without)
    with fit_lock(home, wait_s):
        remove_partials(home)
        config = _read_config(home)
        weight = float(config.get("benchmark_prior_weight", prior_data.BENCHMARK_PRIOR_WEIGHT))
        try:
            features, features_error = FeatureSet.from_config(config.get("features")), None
        except FeatureConfigError as exc:  # a hand-edited config.toml: fit with the built-ins, say why
            features, features_error = FeatureSet(), str(exc)

        def all_docs() -> Iterator[dict | tuple[str | None, dict]]:
            if docs is not None:
                yield from docs
                return
            for doc in store_docs(home):  # the store first, so its copy of a shipped run wins
                yield "user", doc
            if not no_prior:
                yield from prior_data.bundle_entries(bundle_dir)
                yield from prior_data.shared_docs(home)

        (heads_rows, score_info, dropped, runs_by_source, config_support, checks, overlap,
         labels, tally) = collect_rows(all_docs(), without=without, now=now, features=features)
        removed = {k[len("without "):] for k in dropped if k.startswith("without ")}
        unknown, known = unknown_sources(without, set(runs_by_source) | removed, bundle_dir)
        if unknown:
            raise UnknownSource(unknown_source_message(unknown, known))
        if not any(hr.rows for hr in heads_rows.values()):
            raise NothingToFit(_nothing_to_fit(no_prior, without, dropped))
        specs = []
        if not no_prior and "benchmark" not in without:
            specs = prior_data.benchmark_factors(benchmarks, weight=weight)
        seed_key = input_key(heads_rows, specs, {"no_prior": no_prior, "without": sorted(without), "eb": eb,
                                                  "benchmark_prior_weight": weight})
        forest = Forest()
        fitted = {}
        for name, hr in heads_rows.items():
            head_specs = [s for s in specs if s.head == name]
            if not hr.rows:
                continue
            nodes = horizon_nodes(hr.rows)
            if nodes:  # the horizon priors, --no-prior included (spec 04 section 1)
                head_specs += horizon_specs(name, hr.kind, nodes)
            if name == "cost":  # the price offset (spec 04 section 2)
                head_specs += price_spec(hr)
            fitted[name] = fit_head(hr, forest, head_specs, eb=eb)
        fit_id = new_fit_id(fits, now)
        partial = fits / f"{fit_id}.partial"
        partial.mkdir(parents=True)
        arrays = {}
        for name, h in fitted.items():
            key = name.replace(":", "@")
            res: HeadFit = h["fit"]
            arrays[f"{key}__nodes"] = np.array([forest.index[n] for n in h["node_ids"]], dtype=np.int64)
            arrays[f"{key}__ids"] = np.array(h["node_ids"], dtype=str)
            arrays[f"{key}__mean"] = res.mean
            # The sparse posterior precision, not its dense factor or the draws; the state
            # rebuilds both on first use (a dense 4,808-node factor was 185 MB per head)
            arrays[f"{key}__P_data"] = res.precision.data
            arrays[f"{key}__P_indices"] = res.precision.indices.astype(np.int32)
            arrays[f"{key}__P_indptr"] = res.precision.indptr.astype(np.int64)
        np.savez(partial / "state.npz", **arrays)
        task_support: dict[str, int] = {}
        for h in fitted.values():
            for node, (n, _) in h["support"].items():
                if node.split(":", 1)[0] in TASK_LEVELS:
                    task_support[node] = max(task_support.get(node, 0), n)
        score_types = {}
        for name, hr in heads_rows.items():
            if name.startswith("score:") and name in fitted:
                pairs = set(zip(hr.runs, hr.types))
                score_types[name] = dict(Counter(t for _, t in pairs))
        design = {"nodes": forest.to_json(),
                  "support": {name: h["support"] for name, h in fitted.items()},
                  "task_support": task_support, "score_support_by_type": score_types,
                  "config_support": config_support,
                  "task_features": tally.task_values(tally.report["admitted"])}
        _atomic_json(partial / "design.json", design)
        meta = {
            "fit": fit_id, "created_at": now.isoformat(timespec="seconds"), "code_version": _code_version(),
            "seed_key": seed_key, "timebox_effort": design_rows.TIMEBOX_EFFORT_COST,
            "timebox_terms": list(design_rows.TIMEBOX_LEVELS),
            "options": {"no_prior": no_prior, "without": list(without), "full": full, "eb": eb,
                        "benchmark_prior_weight": weight},
            "runs_by_source": runs_by_source,
            "n_runs": {"prior": sum(v for k, v in runs_by_source.items() if k != "user"),
                       "user": runs_by_source.get("user", 0)},
            "shipped_overlap": overlap,
            "user_labels": labels,
            "dropped": dropped,
            "not_model_attempts": checks,
            "heads": {name: {"engine": h["engine"], "kind": h["kind"], "sigma": float(h["fit"].sigma),
                             "phi": h["phi"], "n_rows": h["n_rows"], "n_runs": h["n_runs"],
                             "rows_by_source": _rows_by_source(heads_rows[name]),
                             "rows_not_repriced": heads_rows[name].not_repriced,
                             "center": h["center"], "scale": h["scale"], "baseline": h["baseline"],
                             "iterations": h["fit"].iterations, "optimizer": h["fit"].info,
                             "log_evidence": _finite(h["fit"].log_evidence), "factors": h["factors"],
                             "seed": seed_for(seed_key, name)}
                      for name, h in fitted.items()},
            "scores": score_info,
            "draws": N_DRAWS,
            "design_version": DESIGN_VERSION,
            "features": {**tally.report, **({"error": features_error} if features_error else {})},
            "horizons": tally.horizon_report(),
            "price_offsets": heads_rows["cost"].prices,
        }
        if full:
            meta["full"] = _full_check(fitted)
            table = meta["full"].pop("table", None)
            if table is not None:
                _atomic_json(partial / "pymc_check.json", table)
        meta["seconds"] = round(time.monotonic() - started, 3)
        _atomic_json(partial / "meta.json", meta)
        final = fits / fit_id
        os.replace(partial, final)
        point_latest(fits, fit_id)
        prune_fits(home)
        return final


def _rows_by_source(hr: HeadRows) -> dict[str, int]:
    return dict(Counter(hr.sources))


def _finite(x: float):
    return float(x) if math.isfinite(x) else None


def _full_check(fitted: dict) -> dict:
    try:
        from .check_pymc import compare
    except ImportError as exc:
        return {"ran": False, "reason": f"the bayes extra is not installed ({exc.name})"}
    return compare(fitted)


# ---------------------------------------------------------------- reading a kept fit

def kept_fits(home: Path) -> list[str]:
    """Complete fits on disk, oldest first (D91 keeps the newest KEEP_FITS and the named ones)."""
    fits = Path(home) / "fits"
    if not fits.is_dir():
        return []
    return sorted(p.name for p in fits.glob("fit_*")
                  if p.is_dir() and not p.name.endswith(".partial") and (p / "meta.json").is_file())


def fit_folder(home: Path, fit_id: str) -> Path:
    """The folder of fit `fit_id` (`latest` is `fits/latest`); raises UnknownFit naming the kept fits."""
    fits = Path(home) / "fits"
    fid = str(fit_id or "").strip()
    if fid == "latest" and (fits / "latest" / "meta.json").is_file():
        return (fits / "latest").resolve()
    kept = kept_fits(home)
    if fid not in kept:
        listed = ", ".join(kept[-KEEP_FITS - 3:]) if kept else "none yet (run `loopmath fit`)"
        raise UnknownFit(f"no fit {fid} in {fits}; kept fits, newest last: {listed}")
    return fits / fid


def load_fit(home: Path, fit_id: str):
    """The belief state of a kept fit, for `recommend --fit ID` and `posterior --fit ID`; raises UnknownFit
    for a fit from another design version."""
    from .state import FitState, other_design

    folder = fit_folder(home, fit_id)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    why = other_design(meta)
    if why is not None:
        raise UnknownFit(why)
    return FitState(folder, meta=meta)


# ---------------------------------------------------------------- background

def remove_partials(home: Path) -> None:
    """Delete partial fits left by a crashed fit (called under the fit lock)."""
    for p in (Path(home) / "fits").glob("*.partial"):
        shutil.rmtree(p, ignore_errors=True)


def named_fits(home: Path) -> set[str] | None:
    """Fit ids a receipt (`fit`, `after.fit_after`) or a stored recommendation (`fit.id`) names,
    or None when a receipt or recommendation file could not be read (the store skips those)."""
    from ..store.home import Store

    store = Store(home)
    receipts = list(store.receipts())
    recs = [rec for _, rec in store.recs()]
    on_disk = [len(list((Path(home) / sub).glob(pattern)))
               for sub, pattern in (("receipts", "*.json"), ("recs", "rec_*.json"))]
    if [len(receipts), len(recs)] != on_disk:
        return None
    names: set[str] = set()
    for r in receipts:
        after = r.get("after") if isinstance(r.get("after"), dict) else {}
        names.update((str(r.get("fit") or ""), str(after.get("fit_after") or "")))
    for rec in recs:
        names.add(str((rec.get("fit") if isinstance(rec.get("fit"), dict) else {}).get("id") or ""))
    names.discard("")
    return names


def prune_fits(home: Path, keep: int = KEEP_FITS) -> list[str]:
    """Delete the complete fits other than the newest `keep`, the one `fits/latest` points at
    and any a receipt or stored recommendation names (called under the fit lock). Nothing is
    deleted when a receipt or recommendation cannot be read. Returns the deleted ids."""
    fits = Path(home) / "fits"
    done = sorted(p.name for p in fits.glob("fit_*") if p.is_dir() and not p.name.endswith(".partial"))
    if len(done) <= keep:
        return []
    try:
        named = named_fits(home)
    except (OSError, ValueError):
        return []
    if named is None:
        return []
    kept = set(done[-keep:]) | named
    latest = fits / "latest"
    if latest.is_symlink():
        kept.add(Path(os.readlink(latest)).name)
    removed = [fid for fid in done if fid not in kept]
    for fid in removed:
        shutil.rmtree(fits / fid, ignore_errors=True)
    return removed
