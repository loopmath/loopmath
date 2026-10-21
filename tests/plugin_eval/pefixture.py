"""Synthetic `claude plugin eval` result documents, written from Anthropic's docs (README.md here cites a URL
for every field). Nothing here runs Claude Code or reads a real result; the text inside is loopmath's own.

The docs give no JSON example of the result (https://code.claude.com/docs/en/plugin-evals#json-result lists
"the fields a gating script usually reads"). Two names below are assumed, not documented: the run's own
`score` and the key of its grader list (`graders`). The reader must not depend on the list's key, and it checks
its run grades against the documented `cases[].aggregates.score` and `delta`; `build(..., run_score=False)`
writes runs with no `score`, so the grade comes from the grader list alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CLAUDE_VERSION = "2.1.301"


def grader(name: str, passed: bool, *, weight: float = 1.0, scored: bool = True, kind: str = "regex",
           why: str = "the reply names the fix") -> dict[str, Any]:
    """One grader's result on one run: `passed` and `scored` are documented (PE#troubleshooting,
    PE#compare-against-a-no-plugin-baseline); `weight` is the grader's documented frontmatter (PE#grader-frontmatter);
    `name`, `type` and the explanation's key are assumed names for documented content (PE#json-result).
    A valid fixture holds JSON booleans and numbers, as the reader requires: a wrong type here is the fixture's
    bug, so it fails at once (malformed inputs are made by changing a built document)."""
    if not isinstance(passed, bool) or not isinstance(scored, bool):
        raise TypeError(f"grader {name}: passed and scored must be booleans, not {passed!r} and {scored!r}")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        raise TypeError(f"grader {name}: weight must be a number at or above 0, not {weight!r}")
    return {"name": name, "type": kind, "weight": float(weight), "passed": passed, "scored": scored,
            "explanation": f"loopmath fixture: {why}"}


def grade_of(graders: list[dict[str, Any]]) -> float:
    """The documented run score: the fraction of graders passed, weighted by weight (PE#how-a-case-is-scored),
    over the graders that count (`scored`)."""
    counted = [g for g in graders if g["scored"]]
    total = sum(g["weight"] for g in counted)
    return sum(g["weight"] for g in counted if g["passed"]) / total if total else 0.0


def run(graders: list[dict[str, Any]], *, error: str | None = None, aborted: dict | None = None,
        skipped: bool = False, run_score: bool = True, extra: dict | None = None) -> dict[str, Any]:
    """One run of one arm. `error`, `aborted` and `skippedPaidGraders` are documented (PE#json-result); an aborted
    run scores 0 with `error` null (PE#json-result)."""
    out: dict[str, Any] = {"error": error, "aborted": aborted, "skippedPaidGraders": skipped, "graders": graders}
    if run_score:
        out["score"] = 0.0 if aborted else grade_of(graders)
    if extra:
        out.update(extra)
    return out


def case(name: str, with_runs: list[dict], without_runs: list[dict] | None, *, extra: dict | None = None,
         skipped_in_mean: bool = True) -> dict[str, Any]:
    """One case: `name`, `aggregates.score` (mean with-arm run score), `aggregates.delta` (with minus without),
    `arms.with[]` and `arms.without[]` (PE#json-result). `skipped_in_mean=False` leaves skippedPaidGraders runs
    out of the means, the other reading the reader must accept."""
    def grade(r: dict) -> float:
        return r["score"] if "score" in r else (0.0 if r["aborted"] else grade_of(r["graders"]))

    def mean(rs: list[dict]) -> float | None:
        use = [grade(r) for r in rs if skipped_in_mean or not r["skippedPaidGraders"]]
        return sum(use) / len(use) if use else None

    agg: dict[str, Any] = {"score": mean(with_runs)}
    arms: dict[str, Any] = {"with": with_runs}
    if without_runs is not None:
        arms["without"] = without_runs
        if mean(without_runs) is not None and agg["score"] is not None:
            agg["delta"] = agg["score"] - mean(without_runs)
    out = {"name": name, "aggregates": agg, "arms": arms}
    if extra:
        out.update(extra)
    return out


def document(cases: list[dict], *, cost_usd: float = 1.84, threshold: float = 1.0, partial: str | None = None,
             extra: dict | None = None, schema_version: Any = SCHEMA_VERSION,
             claude_version: str = CLAUDE_VERSION) -> dict[str, Any]:
    """The result document's documented top level (PE#json-result). `threshold` only sets `casesPassed`: the
    threshold itself is not a documented JSON field."""
    scores = [c["aggregates"]["score"] for c in cases if c["aggregates"]["score"] is not None]
    deltas = [c["aggregates"]["delta"] for c in cases if "delta" in c["aggregates"]]
    doc: dict[str, Any] = {
        "schemaVersion": schema_version, "claudeVersion": claude_version,
        "partial": partial is not None, "partialReason": partial,
        "costUsd": cost_usd, "durationSeconds": 412.5,
        "aggregates": {"overallScore": sum(scores) / len(scores) if scores else None,
                       "casesPassed": sum(1 for s in scores if s >= threshold - 1e-12), "casesTotal": len(cases),
                       **({"meanDelta": sum(deltas) / len(deltas)} if deltas else {})},
        "cases": cases}
    if extra:
        doc.update(extra)
    return doc


@dataclass
class Built:
    path: Path                 # aggregate-result.json
    root: Path                 # the plugin root (holds .claude-plugin/plugin.json when `manifest`)
    doc: dict[str, Any]
    runs: dict[str, dict[str, int]] = field(default_factory=dict)   # case -> arm -> run count


def write(root: Path, doc: dict[str, Any], *, manifest: dict | None = None, stamp: str = "2026-10-20T17-02-11-482Z",
          raw: str | None = None) -> Built:
    """Lay the document out where plugin eval puts it: `<plugin>/evals/results/<timestamp>/aggregate-result.json`
    (PE#read-the-results; the timestamp form is the docs' own example), with the manifest at
    `.claude-plugin/plugin.json` (PE#requirements)."""
    folder = root / "evals" / "results" / stamp
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "aggregate-result.json"
    path.write_text(raw if raw is not None else json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if manifest is not None:
        (root / ".claude-plugin").mkdir(parents=True, exist_ok=True)
        (root / ".claude-plugin" / "plugin.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    runs = {c["name"]: {arm: len(rs) for arm, rs in c["arms"].items()} for c in doc.get("cases") or []}
    return Built(path=path, root=root, doc=doc, runs=runs)


def three_graders(a: int, b: int, c: int) -> list[dict[str, Any]]:
    """Weights 2, 1 and 1, so grades fall on 0, 0.25, 0.5, 0.75 and 1. `a`, `b` and `c` are 1 (passed) or 0,
    shorthand only: the graders hold true or false."""
    for v in (a, b, c):
        if v not in (0, 1) or isinstance(v, float):
            raise TypeError(f"three_graders takes 1 or 0, not {v!r}")
    return [grader("names-the-fix", a == 1, weight=2.0), grader("runs-the-tests", b == 1, kind="tool_used"),
            grader("no-new-files", c == 1, kind="file_exists")]


def docs_suite(*, run_score: bool = True, skipped_in_mean: bool = True) -> dict[str, Any]:
    """The docs-built suite: 4 cases at 3 runs per arm (the docs' default, PE#command-options), mixed weights, one
    errored run, one aborted run, one skippedPaidGraders run, a grader excluded in two-arm mode, unknown fields
    and free text."""
    G = three_graders
    rs = run_score
    cases = [
        case("fix-null-check",
             [run(G(1, 1, 1), run_score=rs), run(G(1, 1, 0), run_score=rs), run(G(1, 1, 1), run_score=rs)],
             [run(G(1, 0, 1), run_score=rs), run(G(0, 1, 1), run_score=rs), run(G(1, 1, 1), run_score=rs)],
             extra={"tags": ["bugfix"], "prompt": "loopmath fixture prompt: fix the null check"}),
        case("rename-config-key",
             [run(G(1, 1, 1), run_score=rs), run(G(1, 1, 1), run_score=rs),
              run(G(0, 0, 0), error="timed out after 300s", run_score=rs)],
             [run(G(0, 1, 1), run_score=rs), run(G(1, 0, 0), run_score=rs), run(G(0, 0, 1), run_score=rs)]),
        case("add-skill-call",
             [run(G(1, 1, 1) + [grader("used-skill", True, scored=False, kind="tool_used")], run_score=rs),
              run(G(1, 0, 1), run_score=rs),
              run(G(1, 1, 1), skipped=True, run_score=rs)],
             [run(G(1, 0, 1), run_score=rs),
              run(G(0, 0, 0), aborted={"server": "github", "tool": "create_issue",
                                       "reason": "loopmath fixture: unexpected call"}, run_score=rs),
              run(G(1, 1, 0), run_score=rs)],
             skipped_in_mean=skipped_in_mean),
        case("write-docs-page",
             [run(G(1, 1, 1), run_score=rs, extra={"turns": 7}), run(G(1, 1, 1), run_score=rs),
              run(G(1, 0, 1), run_score=rs)],
             [run(G(1, 1, 1), run_score=rs), run(G(1, 1, 1), run_score=rs), run(G(0, 1, 1), run_score=rs)]),
    ]
    return document(cases, extra={"suite": {"name": "lint-helper", "runs": 3}})


def twin_suite() -> dict[str, Any]:
    """spec/examples/harness-runs/plugin-evals.jsonl as a plugin eval result: the same 3 tasks, 1 run per arm,
    the same accept and reject pattern (one grader standing in for the tests, grade 1 or 0)."""
    passed = {("lint-fix-12", "without"): True, ("lint-fix-12", "with"): True,
              ("lint-fix-31", "without"): False, ("lint-fix-31", "with"): True,
              ("rename-api-4", "without"): True, ("rename-api-4", "with"): True}
    cases = [case(name, [run([grader("tests-pass", passed[(name, "with")], kind="file_exists")])],
                  [run([grader("tests-pass", passed[(name, "without")], kind="file_exists")])])
             for name in ("lint-fix-12", "lint-fix-31", "rename-api-4")]
    return document(cases, cost_usd=1.80)


# The twin's per-run dollars and tokens, as the shipped rows report them (one line per run, `--cost-from`).
TWIN_COSTS = [
    {"case": "lint-fix-12", "arm": "without", "run": 1, "usd": 0.31,
     "tokens": {"input": 42000, "cache_read": 180000, "cache_write": 6000, "output": 3100}},
    {"case": "lint-fix-12", "arm": "with", "run": 1, "usd": 0.27,
     "tokens": {"input": 42000, "cache_read": 180000, "cache_write": 6000, "output": 3100}},
    {"case": "lint-fix-31", "arm": "without", "run": 1, "usd": 0.32,
     "tokens": {"input": 43000, "cache_read": 180000, "cache_write": 6000, "output": 3150}},
    {"case": "lint-fix-31", "arm": "with", "run": 1, "usd": 0.28,
     "tokens": {"input": 43000, "cache_read": 180000, "cache_write": 6000, "output": 3150}},
    {"case": "rename-api-4", "arm": "without", "run": 1, "usd": 0.33,
     "tokens": {"input": 44000, "cache_read": 180000, "cache_write": 6000, "output": 3200}},
    {"case": "rename-api-4", "arm": "with", "run": 1, "usd": 0.29,
     "tokens": {"input": 44000, "cache_read": 180000, "cache_write": 6000, "output": 3200}},
]


def write_costs(path: Path, lines: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path
