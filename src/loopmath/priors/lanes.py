"""Our own build lanes as OCP v0.3 runs (lane 22L).

Input, read only: `lanes.jsonl`, one metadata-only row per agent lane that built
loopmath (and the other lanes of the same agent swarm on the same models). A
private extractor on the build machine writes it from the lane records, the
reviews and the lanes' Claude Code and Codex session logs, parsed with
loopmath's own parsers (`ingest.claude_code`, `ingest.codex`). A row holds only
numbers and categories: role, task type, repo (`loopmath` or `other`), harness,
model, effort, times, four-stream tokens, and the review verdicts in order. No
lane names, titles, objectives, paths or review text reach this module.

- An implementer lane with reviews is the catalog `implement_review` shape: the
  implementer on the lane's model and a separate reviewer lane (Codex) judging
  each round. Round `i` is the implementer's log up to review `i`, so its tokens
  are measured per round; the rounds stop at the first `merge` verdict, and the
  follow-up work after it is counted, not bundled. Each review is an attempt of
  the review piece with its verdict (`fix blocking` rejected, `merge` done) and
  the reviewer lane's session tokens divided by the reviews it gave (cost basis
  `allocated`, which the fit weights as heuristic); without them the review has
  no cost (`dev.loopmath.tokens_unknown`). The run's `referee` verdict is the
  last review's, and the acceptance rule is `referee`.
- An implementer lane with no review record (research, prototypes, side tasks)
  is the catalog `solo` shape with its session tokens and no verdict
  (`settled_unverified`), like E0: it feeds the cost and tokens heads only.
- A reviewer lane is not a run of its own: its verdicts are the gate outcomes
  on the implementer runs it judged and its tokens are allocated to those
  review attempts. (One session judging 20 to 116 diffs as a single run of a
  review-only shape inflated the topology scale and every cost prediction.)
  Its rows are counted in `reviewer_rows`.
- Integration lanes are left out by the extractor: they merge other lanes'
  work, which is not a task a user brings to plan.

No real date or clock time ships: every run is on a synthetic clock that starts
at the UTC epoch (`1970-01-01T00:00:00Z`), and each run, attempt, signal and
producer time is that epoch plus the real elapsed seconds since the lane
started. Durations and round order survive; work hours and dates do not. The
fit reads absolute times only to order a run's signals and to date late events
against the run's end (`belief.outcome`), both relative within one run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from . import ocpdoc
from .registry import LANES

CONVERTER_VERSION = "lanes/1"
SOURCE_REF = "lane-rows"
INPUT_FILE = "lanes.jsonl"
_ROLES = ("implementer",)
_TYPES = ("bug_fix", "feature", "refactor", "tests", "docs", "research", "infra", "data")
_VERDICTS = {"merge": ("done", "accept"), "fix_blocking": ("rejected", "reject")}
ACCEPTANCE = {"name": "referee", "definition": "the reviewer lane's last verdict is merge",
              "requires": ["referee"], "score": None, "excludes_events": ["revert", "incident"],
              "window_days": 14}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)  # every run's synthetic start
EPOCH_TEXT = "1970-01-01T00:00:00Z"


def load_rows(input_dir: Path) -> list[dict]:
    path = Path(input_dir) / INPUT_FILE
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _parse(value) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _clock(row: dict):
    """A row's real times to its synthetic clock: the lane's start is the epoch, elapsed seconds kept."""
    times = [row.get("started_at"), row.get("ended_at"), *[r.get("at") for r in row.get("rounds") or []]]
    real = [t for t in map(_parse, times) if t is not None]
    base = _parse(row.get("started_at")) or (min(real) if real else None)

    def at(value) -> str | None:
        t = _parse(value)
        if t is None or base is None:
            return None
        return (EPOCH + timedelta(seconds=round((t - base).total_seconds()))).strftime("%Y-%m-%dT%H:%M:%SZ")

    return at


def _catalog_workflow(shape: str) -> dict:
    from ..workflows.format import catalog
    from ..workflows.ocp import workflow_to_ocp

    return workflow_to_ocp(catalog()[shape])


def _run_key(row: dict) -> str:
    return hashlib.sha256(str(row["key"]).encode("utf-8")).hexdigest()[:16]


def _attempt(node: str, n: int, *, harness: str, model: str | None, eff: str | None, role: str,
             result: str, tokens: dict | None, prices, started: str | None = None,
             ended: str | None = None, ext: dict | None = None) -> dict:
    rec: dict = {"id": f"{node}.a{n}", "node": node, "n": n, "vertex": node, "round": n,
                 "actor": node, "harness": harness, "model": ocpdoc.model_ref(model),
                 "effort": ocpdoc.effort(eff), "role": {"value": role, "tier": "verified", "evidence": None},
                 "status": result}
    if started:
        rec["started_at"] = started
    if ended:
        rec["ended_at"] = ended
    rec["outcome"] = {"result": result, "evidence": "reported"}
    if tokens is not None:
        rec["cost"] = ocpdoc.cost_block(model, tokens, prices=prices)
    if ext:
        rec["ext"] = ext
    return rec


def _task(row: dict, key: str) -> dict:
    task: dict = {"id": f"lanes/{key}", "repo": row.get("repo") or "other", "org": ocpdoc.ORG,
                  "source": {"kind": LANES, "ref": SOURCE_REF}}
    if row.get("task_type") in _TYPES:
        task["type"] = row["task_type"]
        task["labeled_by"] = {"how": "user", "tier": "reported"}
    return task


def _release(value) -> str:
    """The loopmath version the lane built (`0.1` .. `0.2.2`), else `other`."""
    text = str(value or "")
    return text if text[:1].isdigit() else "other"


def _info(row: dict, **extra) -> dict:
    tokens = row.get("tokens") or {}
    info = {"source": LANES, "converter": CONVERTER_VERSION, "role": row.get("role"),
            "release": _release(row.get("release")), "minutes": round(float(row.get("wall_s") or 0) / 60, 1),
            "session_tokens": sum(int(v or 0) for v in tokens.values()),
            "subagents": int(row.get("subagents") or 0), "clock": "synthetic, starts at the epoch"}
    info.update(extra)
    return info


def convert_lane(row: dict, *, prices=None, producer_version: str = "") -> dict:
    """One OCP v0.3 run for one lane row."""
    role = row.get("role")
    if role not in _ROLES:
        raise ValueError(f"lane row role {role!r} is not one of {', '.join(_ROLES)}")
    key = _run_key(row)
    harness, model, eff = row.get("harness") or "claude-code", row.get("model"), row.get("effort")
    rounds = row.get("rounds") or []
    at = _clock(row)
    run: dict = {"id": f"lanes/{key}", "started_at": at(row.get("started_at")),
                 "ended_at": at(row.get("ended_at")), "task": _task(row, key),
                 "provenance": {"kind": "designed", "chooser": "planner"}}
    signals: list[dict] = []
    edges = None
    if role == "implementer" and rounds:
        rv = row.get("reviewer") or {}
        rv_harness = "codex" if rv.get("harness") == "codex" else "claude-code"
        workflow = _catalog_workflow("implement_review")
        settings = {"implement": ocpdoc.setting(harness, model, eff),
                    "review": ocpdoc.setting(rv_harness, rv.get("model"), rv.get("effort"))}
        attempts, prev = [], at(row.get("started_at"))
        for i, rnd in enumerate(rounds, start=1):
            result, verdict = _VERDICTS.get(rnd.get("verdict"), ("settled_unverified", "error"))
            toks = rnd.get("tokens") or {}
            attempts.append(_attempt("implement", i, harness=harness, model=model, eff=eff, role="implementer",
                                     result="done", tokens=toks if any(toks.values()) else None, prices=prices,
                                     started=prev, ended=at(rnd.get("at")),
                                     ext=None if any(toks.values()) else
                                     {"dev.loopmath.tokens_unknown": "no log lines between two reviews"}))
            review = _attempt("review", i, harness=rv_harness, model=rv.get("model"), eff=rv.get("effort"),
                              role="reviewer", result=result, tokens=None, prices=prices, ended=at(rnd.get("at")))
            if any((rv.get("tokens_per_review") or {}).values()):
                review["cost"] = ocpdoc.cost_block(rv.get("model"), rv["tokens_per_review"], prices=prices,
                                                   basis="allocated")
                # one reviewer session split evenly over the reviews it gave (OCP E171's shared-session case)
                review["cost"]["ext"] = {"dev.loopmath.logmatch": {"shared_session": True}}
            else:
                review["ext"] = {"dev.loopmath.tokens_unknown": "no reviewer session to allocate from"}
            attempts.append(review)
            prev = at(rnd.get("at"))
        _, last_verdict = _VERDICTS.get(rounds[-1].get("verdict"), ("settled_unverified", "error"))
        signals.append({"id": "sig_referee", "kind": "verdict", "name": "referee", "value": last_verdict,
                        "at_attempt": f"review.a{len(rounds)}", "observed_at": at(rounds[-1].get("at")) or EPOCH_TEXT,
                        "source": {"kind": "orchestrator", "ref": "reviewer lane"}, "tier": "reported"})
        nodes = [{"id": "implement", "kind": "impl", "vertex": "implement", "state": "done"},
                 {"id": "review", "kind": "review", "vertex": "review",
                  "state": "done" if last_verdict == "accept" else "rejected", "gate": {"rule": "referee"}}]
        edges = [{"from": "implement", "to": "review", "kind": "dep", "tier": "verified"}]
        after = row.get("after_accept") or {}
        info = _info(row, rounds=len(rounds), merged=row.get("merged"),
                     after_accept_tokens=sum(int(v or 0) for v in (after.get("tokens") or {}).values()),
                     after_accept_reviews=int(after.get("reviews") or 0))
        run["acceptance_rule"] = dict(ACCEPTANCE)
        run["signals"] = signals
    else:
        workflow = ocpdoc.workflow_solo()
        settings = {"implement": ocpdoc.setting(harness, model, eff)}
        attempts = [_attempt("implement", 1, harness=harness, model=model, eff=eff, role="implementer",
                             result="settled_unverified", tokens=row.get("tokens") or {}, prices=prices,
                             started=at(row.get("started_at")), ended=at(row.get("ended_at")))]
        nodes = [{"id": "implement", "kind": "impl", "vertex": "implement", "state": "settled_unverified"}]
        info = _info(row, rounds=1)
    run["configuration"] = ocpdoc.configuration(workflow, settings, source="designed")
    run["ext"] = {"dev.loopmath.prior": info}
    run = {k: v for k, v in run.items() if v is not None}
    return ocpdoc.run_doc(run=run, nodes=nodes, attempts=attempts, edges=edges, producer_version=producer_version,
                          emitted_at=at(row.get("ended_at")) or EPOCH_TEXT, source_contract="lane-rows/1")


def iter_lanes(input_dir: Path, *, prices=None, producer_version: str = "",
               counts: dict | None = None) -> Iterator[dict]:
    """Bundled runs from `lanes.jsonl`, in row order; `counts` gets runs by model and role."""
    tally = counts if counts is not None else {}
    by: dict[str, int] = {}
    reviewers = 0
    for row in load_rows(input_dir):
        if row.get("role") == "reviewer":
            reviewers += 1
            continue
        doc = convert_lane(row, prices=prices, producer_version=producer_version)
        k = f"{row.get('model')}:{row.get('role')}"
        by[k] = by.get(k, 0) + 1
        yield doc
    tally["runs_by_model_role"] = dict(sorted(by.items()))
    tally["runs"] = sum(by.values())
    tally["reviewer_rows"] = reviewers


def merge_into_bundle(bundle_dir: Path, built_dir: Path, name: str = LANES) -> dict:
    """Copy one source's file and manifest entry from a bundle built with only that source.

    `prior build` writes every selected source and removes the others' files, so adding
    one source to the shipped bundle without rebuilding the rest goes through a separate
    build (`LOOPMATH_PRIOR_SOURCES=lanes loopmath prior build --out TMP`) and this merge.
    The other sources' files are not touched; the manifest's size is recomputed.
    """
    from .build import SIZE_CAP_BYTES, BundleError, atomic_write

    bundle_dir, built_dir = Path(bundle_dir), Path(built_dir)
    built = json.loads((built_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = built["sources"].get(name)
    if entry is None:
        raise BundleError(f"{built_dir} has no source {name!r}")
    if built.get("problems"):
        raise BundleError("; ".join(built["problems"][:5]))
    target = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    blob = (built_dir / entry["file"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise BundleError(f"{entry['file']} does not match its manifest digest")
    sources = dict(target.get("sources") or {})
    sources[name] = entry
    target["sources"] = sources
    target["size_bytes"] = sum(int(e["bytes"]) for e in sources.values())
    if target["size_bytes"] >= int(target.get("size_cap_bytes") or SIZE_CAP_BYTES):
        raise BundleError(f"bundle would be {target['size_bytes']} bytes")
    atomic_write(bundle_dir / entry["file"], blob)
    atomic_write(bundle_dir / "manifest.json",
                 (json.dumps(target, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    return target
