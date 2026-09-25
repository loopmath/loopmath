"""Build the bundle from the sweep, E0, RQ1 phase 1 and repo-history runs (lane 11).

`build_bundle` converts each source to OCP v0.3, applies the share reduction,
validates every document, and writes one gzipped JSON Lines file per source plus
`manifest.json` with provenance. Output bytes are deterministic for the same
inputs and salt (runs sorted by id, gzip mtime 0), so a rebuild diffs cleanly;
only the manifest's `built_at` changes. The total must stay under 5 MB.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import io
import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .. import __version__
from . import ocpdoc
from .reduce import reduce_for_bundle
from .registry import BUNDLE_SOURCES
from .validate import validate_bundle_doc

SIZE_CAP_BYTES = 5_000_000
MANIFEST_SCHEMA = "loopmath.prior.manifest/1"
BUNDLE_DIR = Path(__file__).resolve().parent / "bundle"


class BundleError(Exception):
    pass


@dataclass
class SourceResult:
    """What one source converter produced."""

    name: str
    docs: list[dict]
    inputs: dict                       # {description, files, sha256}
    converter: str
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=dict)


def now_local() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def digest_files(paths: Iterable[Path]) -> dict:
    """File count and a SHA-256 over the sorted per-file digests (file names only, no folders)."""
    items = []
    for p in sorted(Path(x) for x in paths):
        items.append(f"{p.name}\t{hashlib.sha256(p.read_bytes()).hexdigest()}")
    return {"files": len(items), "sha256": hashlib.sha256("\n".join(items).encode()).hexdigest()}


def gz_lines(docs: list[dict]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        for doc in docs:
            gz.write(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            gz.write(b"\n")
    return buf.getvalue()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def build_bundle(out_dir: Path, results: list[SourceResult], *, salt: str | None = None,
                 size_cap: int = SIZE_CAP_BYTES, strict: bool = True) -> dict:
    """Reduce, validate and write every source. Returns the manifest.

    `salt` hashes non-public repo names; without one a random salt is drawn and
    not stored, so the hashes cannot be reversed from the package. With
    `strict`, any validation error, wrong source kind, duplicate id or a bundle
    at or over `size_cap` raises `BundleError` and nothing is written.
    """
    from .show import run_summary

    salt_kind = "given" if salt else "random per build, not stored"
    salt = salt or secrets.token_hex(32)
    manifest: dict = {
        "schema": MANIFEST_SCHEMA,
        "built_at": now_local(),
        "loopmath_version": __version__,
        "ocp": ocpdoc.OCP_VERSION,
        "config_id_impl": ocpdoc.CONFIG_ID_IMPL,
        "tariff": ocpdoc.tariff(),
        "reduction": {"profile": "share reduction, spec 03 section 7 (Analyst D19)", "repo_salt": salt_kind},
        "size_cap_bytes": size_cap,
        "sources": {},
    }
    blobs: dict[str, bytes] = {}
    problems: list[str] = []
    for res in results:
        if res.name not in BUNDLE_SOURCES:
            raise BundleError(f"unknown source {res.name!r}; known: {', '.join(BUNDLE_SOURCES)}")
        reduced = [reduce_for_bundle(d, salt=salt) for d in res.docs]
        reduced.sort(key=lambda d: str(d["run"]["id"]))
        ids = [d["run"]["id"] for d in reduced]
        if len(set(ids)) != len(ids):
            problems.append(f"{res.name}: duplicate run ids")
        errors = warnings = 0
        checker = ""
        for d in reduced:
            kind = ((d["run"].get("task") or {}).get("source") or {}).get("kind")
            if kind != res.name:
                problems.append(f"{res.name}: run {d['run']['id']} has task.source.kind {kind!r}")
            found, checker = validate_bundle_doc(d)
            errs = [f for f in found if f["level"] == "error"]
            errors += len(errs)
            warnings += len(found) - len(errs)
            if errs and len(problems) < 20:
                problems.append(f"{res.name}: {d['run']['id']}: {errs[0]['code']} {errs[0]['message']}")
        blob = gz_lines(reduced)
        fname = f"{res.name}.jsonl.gz"
        blobs[fname] = blob
        manifest["sources"][res.name] = {
            "file": fname,
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "runs": len(reduced),
            "converter": res.converter,
            "inputs": res.inputs,
            "validation": {"checker": checker, "errors": errors, "warnings": warnings},
            "summary": run_summary(reduced),
            "counts": res.counts,
            "notes": res.notes,
            "warnings": res.warnings[:50],
        }
    total = sum(len(b) for b in blobs.values())
    manifest["size_bytes"] = total
    if total >= size_cap:
        problems.append(f"bundle is {total} bytes, cap {size_cap}")
    manifest["problems"] = problems
    if problems and strict:
        raise BundleError("; ".join(problems[:10]))
    out_dir = Path(out_dir)
    for fname, blob in blobs.items():
        atomic_write(out_dir / fname, blob)
    for stale in out_dir.glob("*.jsonl.gz"):
        if stale.name not in blobs:
            stale.unlink()
    atomic_write(out_dir / "manifest.json",
                 (json.dumps(manifest, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    return manifest


# ---------------------------------------------------------------- source runners
def run_sweep(results_dir: Path) -> SourceResult:
    from .sweep import CONVERTER_VERSION, iter_sweep, sweep_layout

    runs_dir, attempts = sweep_layout(results_dir)
    if not any(runs_dir.glob("*.run.json")):
        raise BundleError(f"sweep input {results_dir} has no run files (*.run.json), directly or in dagr/")
    docs, warnings = [], []
    counts = {"infra_error_runs": 0, "runs_with_infra_rounds": 0, "cost_incomplete_runs": 0,
              "superseded_rows": 0, "later_rows": 0}
    for _, doc, warn in iter_sweep(results_dir, producer_version=__version__):
        docs.append(doc)
        warnings += warn
        info = doc["run"]["ext"]["dev.loopmath.prior"]
        counts["infra_error_runs"] += bool(info["infra_error"])
        counts["runs_with_infra_rounds"] += info["infra_rounds"] > 0
        counts["cost_incomplete_runs"] += not info["cost_complete"]
        counts["superseded_rows"] += info["superseded_rows"]
        counts["later_rows"] += info["later_rows"]
    files = sorted(runs_dir.glob("*.run.json")) + [attempts]
    inputs = {"description": "loopmath internal sweep (sweep0830): the experiment repository's results, "
                             "contract v3 run files and attempts.jsonl",
              **digest_files([f for f in files if f.exists()])}
    notes = [
        "run files are authoritative; attempts.jsonl rows are matched by developer start time",
        "superseded_rows: earlier executions the harness re-ran after a crash; later_rows: executions after the "
        "run file was written; neither is in the bundle",
        "a developer or reviewer round whose CLI crashed without a timeout is flagged dev.loopmath.infra_error; "
        "when every developer round crashed the tests verdict is error",
        "timed-out rounds that reported no usage carry dev.loopmath.tokens_unknown and no cost",
        "planner tokens are the one shared plan per task and planner (dev.loopmath.shared_across_runs)",
        "dollars priced with the packaged prices.toml (tariff above), not the recorded cost_usd",
        "em dashes in source receipts are replaced by commas; receipts are then dropped by the reduction",
    ]
    return SourceResult("sweep", docs, inputs, CONVERTER_VERSION, notes, warnings, counts)


def run_e0(corpus_dir: Path) -> SourceResult:
    from .e0 import CONVERTER_VERSION, iter_e0

    files = [corpus_dir / "sessions.jsonl", corpus_dir / "session-dag-join.jsonl"]
    if not files[0].is_file():
        raise BundleError(f"E0 input {corpus_dir} has no sessions.jsonl")
    counts: dict = {}
    docs = list(iter_e0(corpus_dir, producer_version=__version__, counts=counts))
    inputs = {"description": "E0 corpus copy (parser-spec v1, built 2026-08-28): sessions.jsonl and "
                             "session-dag-join.jsonl", **digest_files([f for f in files if f.is_file()])}
    notes = [
        "one logged habit run per real main session (catalog solo shape: harness, primary model, dominant effort)",
        "no verdicts (D7): no signals and no acceptance rule; these runs feed the cost and tokens heads only",
        "no task type: the corpus has no text to label from",
        "subagent transcripts are extra attempts on their parent's run (dev.loopmath.subagent)",
        "left out: fleet stubs (no model, an API error), sessions with no model or tokens, temporary folders",
        "Codex input tokens exclude the cached tokens (input minus cached, as ingest.codex does)",
        "dollars priced with the packaged prices.toml",
    ]
    return SourceResult("e0", docs, inputs, CONVERTER_VERSION, notes, [], counts)


def run_rq1(ocp_dir: Path) -> SourceResult:
    from .rq1 import CONVERTER_VERSION, iter_rq1

    files = sorted(ocp_dir.glob("*.ocp.json"))
    if not files:
        raise BundleError(f"RQ1 input {ocp_dir} has no *.ocp.json run documents")
    docs, changed = [], 0
    for _, doc in iter_rq1(ocp_dir, producer_version=__version__):
        docs.append(doc)
        info = doc["run"]["ext"]["dev.loopmath.prior"]
        changed += info["lane10_config_id"] != doc["run"]["configuration"]["id"]
    manifest = ocp_dir / "MANIFEST.json"
    inputs = {"description": "loopmath internal RQ1 phase 1: lane 10's OCP v0.3 conversion (D15)",
              **digest_files(files + ([manifest] if manifest.is_file() else []))}
    notes = [
        "D15 labels checked on every run: type feature, repo ale-bench, subtype ahc/<problem>, source rq1, "
        "rule heldout_perf>=2400",
        "configuration ids recomputed with the bundle's canonical form; lane 10's id kept as lane10_config_id",
        "attempt roles written as their piece's role (D32)",
        "dollars repriced from tokens with the packaged prices.toml; lane 10's list-price total is recorded_usd",
        "lane 10's submission curve (dev.loopmath.rq1) is dropped by the share reduction",
    ]
    return SourceResult("rq1", docs, inputs, CONVERTER_VERSION, notes, [],
                        {"runs": len(docs), "config_ids_changed": changed})


def run_lanes(input_dir: Path) -> SourceResult:
    from .lanes import CONVERTER_VERSION, INPUT_FILE, iter_lanes

    path = input_dir / INPUT_FILE
    if not path.is_file():
        raise BundleError(f"lanes input {input_dir} has no {INPUT_FILE}")
    counts: dict = {}
    docs = list(iter_lanes(input_dir, producer_version=__version__, counts=counts))
    inputs = {"description": "our own agent build lanes (lane 22L): metadata-only rows from the lane records, "
                             "reviews and session logs", **digest_files([path])}
    notes = [
        "implementer lanes with reviews: catalog implement_review, one round per review up to the first merge, "
        "tokens measured per round from the implementer's log; follow-up work after the first merge not bundled",
        "review attempts carry the verdict and the reviewer lane's session tokens divided by the reviews it gave "
        "(cost basis allocated); reviewer lanes are not runs of their own",
        "implementer lanes with no review record: catalog solo, settled_unverified, cost and tokens heads only",
        "integration lanes, lanes still running and lanes with no matching log are left out (counted by the "
        "extractor, not here)",
        "tokens from loopmath's own Claude Code and Codex parsers; dollars priced with the packaged prices.toml",
    ]
    return SourceResult("lanes", docs, inputs, CONVERTER_VERSION, notes, [], counts)


RUNNERS: dict[str, Callable[[Path], SourceResult]] = {"sweep": run_sweep, "e0": run_e0, "rq1": run_rq1,
                                                      "lanes": run_lanes}
