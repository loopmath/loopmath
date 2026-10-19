"""Synthetic harness2 runs folders, written from reading harness2's code (never from running it).

harness2 is read only: no test imports it, runs it or reads its clone. Every folder and field below
names the harness2 line it follows, relative to `src/harness2/` of the clone at 1cfe254 (Apache 2.0).
README.md in this folder gathers the citations. The text inside the trees and outputs is loopmath's own.

`build(root, ...)` writes one bench's runs root under `root` and returns a `Built` that lists every
rollout it wrote, with what the importer should make of it. Variants switch on the edge cases.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# config.py:26 METHOD_VERSION; runmark.py:33 VERSION
METHOD_VERSION = 2
MARK_VERSION = 1
H2_VERSION = "0.1.0"  # pyproject.toml:7

VARIANTS = frozenset({
    "judge-failed",      # one adopted rollout's judge row has no score (core/judge.py:80-85, cli/judge.py:190)
    "base-unscored",     # one base_judged row has score null (cli/judge.py:121-130)
    "fallback",          # the seq selection fell back: ok false, fallback true (core/selection.py:237-252)
    "dead-rollout",      # one stage rollout has no output or trajectory folder
    "legacy-arms",       # summary keyed `arms` and judged rows keyed `arm` (core/schedule.py:74-81, cli/judge.py:57-59)
    "bare-base",         # summary base entries are bare labels (cli/judge.py:115-118)
    "base-failed",       # status base-failed: bases only, no modes (core/schedule.py:630)
    "mode-crashed",      # the seq mode record is {ok false, error, traceback} (core/schedule.py:655-659)
    "no-config",         # no config.json (the demo writes none: cli/demo.py:35-52)
    "no-judge",          # neither judged.json nor base_judged.json
    "retry",             # JB: two attempt files for one rollout (benchmarks/jobbench.py:801, 872-875)
    "zero-cost",         # OpenCode step costs all 0 with tokens (priced by loopmath)
    "garbage-lines",     # stream lines that are not JSON, as oc-output.txt mixes stderr in (oc_agent.py:224)
    "stale-judge",       # the agg selection now adopts agg2, judged.json still scores agg1 (cli/judge.py:65-80)
    "failed-rollout",    # dom1 started, stream cut mid-step, no .adapt_complete (core/runmark.py:64-67)
    "mixed-cost",        # OpenCode reports cost on one billable step and none on the other
    "moved",             # WB: recorded run dirs name the machine that ran it, not this folder (config.py:91)
    "rerun-trial",       # WB: an earlier trial of inv1 on another tree, marked with the same run id (workbuddy.py:805-824)
})
WB_SUBSETS = ("code", "web", "office")   # benchmarks/workbuddy.py:84-110
RAN_AT = "/srv/harness2/runs"            # where the "moved" variant says harness2 ran


@dataclass
class Rollout:
    run_id: str
    task_id: str
    mode: str            # "agg" or "seq"; bases are listed under "agg", where they ran (schedule.py:606, 619-620)
    label: str           # base<i>, inv<i>, dom<i>, agg<i>, c<t>
    tree: Path | None    # the tree the rollout ran on
    parent: Path | None  # the tree that stage edited, or None for a base
    arm: str             # base | candidate | adopted
    score: float | None  # what the importer should record; None is unscored
    judged: bool         # a judge row exists for it (scored or not)
    usd: float | None
    tokens: dict | None
    dead: bool = False
    failed: bool = False         # started, no marker: a failed attempt, cost incomplete
    partial_usd: float | None = None
    run_dir: Path | None = None  # the folder harness2 returned for it (a WB trial folder)


@dataclass
class Built:
    root: Path               # the runs root (HARNESS2_RUNS_DIR, config.py:91)
    adapt: Path              # <root>/<bench>/adapt (cli/_common.py:70-72)
    bench: str
    substrate: str
    k: int
    executor: str
    judge: str
    rollouts: list[Rollout] = field(default_factory=list)

    @property
    def wb(self) -> bool:
        return self.bench.startswith("wb-")

    @property
    def home(self) -> Path:
        """The bench's runs root: `<root>/<bench>`, or `<root>/wb/<subset>` (benchmarks/workbuddy.py:653-658)."""
        return self.adapt.parent

    def by_label(self, task_id: str, mode: str, label: str) -> Rollout:
        return next(r for r in self.rollouts if r.task_id == task_id and r.mode == mode and r.label == label)


def ns(bench: str, sub: str, mode: str, k: int, seed: int | None = None) -> str:
    """core/schedule.py:111-127: `<bench>-<sub>-<mode><k>[-s<seed>][-<tag>]`."""
    out = f"{bench}-{sub}-{mode}{k}"
    return out + (f"-s{seed}" if seed is not None else "")


def fingerprint(tree: Path) -> str:
    """core/runmark.py:41-61, re-written: SHA-1 over the sorted files, first 16 hex."""
    h = hashlib.sha1()
    for p in sorted(tree.rglob("*")):
        if not p.is_file() or p.name == ".adapt_complete" or "__pycache__" in p.parts \
                or p.suffix == ".pyc" or p.name == ".DS_Store":
            continue
        h.update(str(p.relative_to(tree)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path, value, indent: int = 2) -> Path:
    return _write(path, json.dumps(value, indent=indent) + "\n")


def _tree(path: Path, note: str, *, codex: bool) -> Path:
    """A harness tree as core/tree.py writes one. OpenCode: root systemprompt.md, guardrails.md,
    LongTermMEMORY.md, tool_descriptions/, skills/, scripts/ (core/prompts.py:424-431). Codex: AGENTS.md and
    .agents/skills/ (substrate/codex.py:41, 47); no root markers, hence layout="harness2"."""
    if codex:
        _write(path / "AGENTS.md", f"Work the ledger task carefully.\n{note}\n")
        _write(path / ".agents/skills/tally/SKILL.md", "---\nname: tally\n---\nAdd signed amounts.\n")
    else:
        _write(path / "systemprompt.md", f"Work the ledger task carefully.\n{note}\n")
        _write(path / "guardrails.md", "Never invent numbers.\n")
        _write(path / "LongTermMEMORY.md", "")
        _write(path / "tool_descriptions/tool_guidance.md", "Read files before editing them.\n")
        _write(path / "skills/tally/SKILL.md", "---\nname: tally\n---\nAdd signed amounts.\n")
    _write(path / "scripts/tally.py", "import sys\nprint(sum(float(x) for x in sys.argv[1:]))\n")
    _write(path / "scripts/README.md", "stub\n")  # core/tree.py:141, ignored by the digest
    return path


def _oc_stream(steps: list[tuple[dict, float]], *, garbage: bool) -> str:
    """`opencode run --format json` lines (benchmarks/jb_opencode_runner.py:32, 203-206): step_finish
    parts carry tokens {input, output, reasoning, cache {read, write}} and cost (lab_runner/opencode/
    session.py:38-49; wb_runner/agents/oc_agent.py:353-370; core/proposer.py:486-504)."""
    lines = []
    if garbage:
        lines.append("INFO starting session")  # stderr mixed in (oc_agent.py:224)
    for i, (tok, cost) in enumerate(steps):
        ts = 1760830000000 + 1000 * i
        lines.append(json.dumps({"type": "step_start", "timestamp": ts, "sessionID": "ses_fixture",
                                 "part": {"type": "step-start"}}))
        lines.append(json.dumps({"type": "text", "timestamp": ts + 400, "sessionID": "ses_fixture",
                                 "part": {"type": "text", "text": "Tallied."}}))
        part = {"type": "step-finish", "tokens": tok}
        if cost is not None:   # mixed-cost: a step that reports no cost has no `cost` key
            part["cost"] = cost
        lines.append(json.dumps({"type": "step_finish", "timestamp": ts + 900, "sessionID": "ses_fixture",
                                 "part": part}))
    if garbage:
        lines.append("{not json")
    return "\n".join(lines) + "\n"


def _cx_stream(turns: list[dict], *, garbage: bool) -> str:
    """`codex exec --json` lines: turn.completed carries usage {input_tokens (cached included),
    cached_input_tokens, output_tokens, reasoning_output_tokens} (substrate/codex.py:432-444)."""
    lines = [json.dumps({"type": "thread.started", "thread_id": "th_fixture"})]
    if garbage:
        lines.append("warning: not json")
    for usage in turns:
        lines.append(json.dumps({"type": "turn.started"}))
        lines.append(json.dumps({"type": "turn.completed", "usage": usage}))
    return "\n".join(lines) + "\n"


def _usage(seed: int, *, codex: bool) -> tuple[list, dict]:
    """Per-step usage and the four-stream total loopmath should read from it."""
    if codex:
        turns = [{"input_tokens": 3000 + seed, "cached_input_tokens": 1000, "output_tokens": 200 + seed,
                  "reasoning_output_tokens": 50},
                 {"input_tokens": 2000, "cached_input_tokens": 500, "output_tokens": 100,
                  "reasoning_output_tokens": 0}]
        total = {"input": (3000 + seed - 1000) + (2000 - 500), "cache_read": 1500, "cache_write": 0,
                 "output": 300 + seed}
        return turns, total
    steps = [{"input": 1200 + seed, "output": 80, "reasoning": 20, "cache": {"read": 400, "write": 10}},
             {"input": 600, "output": 40 + seed, "reasoning": 0, "cache": {"read": 200, "write": 0}}]
    total = {"input": 1800 + seed, "cache_read": 600, "cache_write": 10, "output": 140 + seed}
    return steps, total


def build(root: Path, *, bench: str = "jb", substrate: str = "oc", k: int = 2,
          tasks: tuple[str, ...] = ("finance/task1", "finance/task2"), modes: tuple[str, ...] = ("agg", "seq"),
          seed: int | None = None, executor: str | None = None, judge: str = "judge-model-x",
          method_version: int = METHOD_VERSION, variants: frozenset[str] | set[str] = frozenset()) -> Built:
    unknown = set(variants) - VARIANTS
    assert not unknown, unknown
    wb = bench.startswith("wb-")
    assert bench in ("jb", "lab") or (wb and bench[3:] in WB_SUBSETS), bench
    assert wb or not {"moved", "rerun-trial"} & set(variants), "WB variants"
    codex = substrate == "cx"
    executor = executor or ("gpt-5.4" if codex else "zai/glm-5" if wb else "gemini-3.5-flash")
    root = Path(root)
    adapt = (root / "wb" / bench[3:] if wb else root / bench) / "adapt"   # cli/_common.py:70-72; workbuddy.py:653-658
    built = Built(root=root, adapt=adapt, bench=bench, substrate=substrate, k=k, executor=executor, judge=judge)
    agg_ns = ns(bench, substrate, "agg", k, seed)                    # cli/_common.py:91-94
    agg_dir = adapt / agg_ns
    if "no-config" not in variants:                                  # cli/run_sweep.py:209-235
        _json(agg_dir / "config.json", {
            "method_version": method_version, "executor": executor, "improver": "improver-model-z",
            "judge": judge, "project": "", "location": "", "judge_endpoint": "", "endpoint": "",
            "split": "test", "tasks": [[t.split("/")[0], t] for t in tasks], "max_tasks": None,
            "domains": [], "seed": seed})
    judged_rows, base_rows = [], []
    for n, task in enumerate(tasks):
        task_dir = task.replace("/", "__")                           # core/schedule.py:101-102
        _one_task(built, agg_dir, adapt, task, task_dir, n, modes, seed, variants, judged_rows, base_rows)
    if "no-judge" in variants:
        for r in built.rollouts:
            r.score, r.judged = None, False
    else:
        if judged_rows:
            _json(agg_dir / "judged.json", judged_rows)              # cli/judge.py:190; core/judge.py:146-155
        _json(agg_dir / "base_judged.json", base_rows, indent=1)     # cli/judge.py:143-144
    return built


def _rollout_dirs(built: Built, task: str, run_id: str) -> tuple[Path, Path]:
    """(output folder, trajectory folder) for one rollout.
    JB: `<RUNS>/jb/dataset/<task>/model_output/<run_id with / as __>` and `.../model_traj/<same>`
    (benchmarks/jobbench.py:642, 676-681, 721-723). LAB: `<RUNS>/lab/results/<oc|cx>/<run_id>` for both
    (benchmarks/lab.py:524-529, 585-586)."""
    if built.bench == "jb":
        label = run_id.replace("/", "__")
        base = built.root / "jb" / "dataset" / task
        return base / "model_output" / label, base / "model_traj" / label
    d = built.root / "lab" / "results" / built.substrate / run_id
    return d, d


def _write_rollout(built: Built, r: Rollout, n: int, variants, *, cut: bool = False) -> None:
    """`cut`: the run died mid-step: the stream ends on a step_start or turn.started with no usage after it
    (lab_runner/opencode/session.py:42 counts step-start; the turn.started line is codex exec's own event, as
    loopmath's tests/fixtures/graph/heldout/annotation-audit.json:26 records it), and no marker is written,
    since mark_complete runs only after the runner returned normally (core/runmark.py:64-67)."""
    out, traj = _rollout_dirs(built, r.task_id, r.run_id)
    codex = built.substrate == "cx"
    _write(out / "answer.md", f"Total for {r.task_id}: {100 + n}\n")
    per_step, total = _usage(n, codex=codex)
    garbage = "garbage-lines" in variants
    attempts = 2 if ("retry" in variants and r.label.startswith("base") and n == 0) else 1
    usd_total = 0.0
    tokens = {key: 0 for key in total}
    for a in range(attempts):
        if codex:
            text = _cx_stream(per_step, garbage=garbage)
        else:
            costs = [0.0, 0.0] if "zero-cost" in variants else [0.0125, None] if "mixed-cost" in variants \
                else [0.0125, 0.0075]
            usd_total += sum(c or 0.0 for c in costs)
            text = _oc_stream(list(zip(per_step, costs)), garbage=garbage)
        if cut:
            text += json.dumps({"type": "turn.started"} if codex else
                               {"type": "step_start", "timestamp": 1760830009000, "sessionID": "ses_fixture",
                                "part": {"type": "step-start"}}) + "\n"
        if built.bench == "jb":
            _write(traj / f"attempt-{a + 1}.jsonl", text)            # jobbench.py:869-880
        else:
            name = "codex_session.jsonl" if codex else "opencode_session.jsonl"   # lab.py:762
            _write(traj / name, text)
            _json(out / "config.json", {"model": built.executor,  # lab_runner/opencode/run_opencode.py:230-251
                                        "reasoning_effort": "medium" if codex else None,
                                        "started_at": "2026-10-18T09:00:00-07:00"})
        for key in tokens:
            tokens[key] += total[key]
    if not cut:
        _json(traj / ".adapt_complete", {"version": MARK_VERSION, "run_id": r.run_id, "at": 1760803200.0 + n,
                                         "harness": fingerprint(r.tree) if r.parent is not None else "base"},
              )                                                      # core/runmark.py:64-81 (None digests to "base")
    r.tokens = tokens
    if cut:
        r.failed, r.usd = True, None
        r.partial_usd = None if codex or "zero-cost" in variants else round(usd_total, 6)
    elif "mixed-cost" in variants and not codex:
        r.usd, r.partial_usd = None, round(usd_total, 6)   # priced from tokens, or unpriced
    elif codex:
        r.usd = None  # priced by loopmath from tokens, or unpriced
    elif "zero-cost" in variants:
        r.usd = None  # priced by loopmath from tokens
    else:
        r.usd = round(usd_total, 6)


def _label_sanitize(text: str) -> str:
    """benchmarks/workbuddy.py:113-114."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", text)


def _write_wb(built: Built, r: Rollout, n: int, *, label: str, marker_id: str, suffix: str, cut: bool = False,
              run: str = "2026-10-18__09-00-00", harness: str | None = None, at: float = 0.0) -> Path:
    """One WorkBuddy trial, `<home>/rollouts/<label>/<run>/<task, first 32>__<suffix>/` (workbuddy.py:451,
    565-579, 716-728; Harbor names a trial after at most 32 characters of the task, 121-129). It holds the
    verifier's score.json {reward} (408-420, 986-1004), the agent stream agent/oc-output.txt or
    agent/codex-output.jsonl (863-864) and, unless the trial was cut, the marker with the cell identity
    {solver, oc_ver, connection} (runmark.py:64-81; workbuddy.py:738-746, 821-823, 853-854; config.py:168,
    172). The importer reads scores from the judge files and no cost from the stream."""
    trial = built.home / "rollouts" / _label_sanitize(label) / run / f"{r.task_id[:32]}__{suffix}"
    _json(trial / "verifier" / "score.json", {"reward": 0.5})
    stream = "agent/codex-output.jsonl" if built.substrate == "cx" else "agent/oc-output.txt"
    _write(trial / stream, "The agent's capture, as Harbor writes it.\n")
    _write(trial / "outputs" / "memo.md", f"Memo for {r.task_id}.\n")
    if not cut:
        _json(trial / ".adapt_complete", {
            "version": MARK_VERSION, "run_id": marker_id, "at": 1760803200.0 + n + at,
            "harness": harness or (fingerprint(r.tree) if r.parent is not None else "base"),
            "solver": built.executor, "oc_ver": "1.14.18", "connection": "local_proxy"})
    r.run_dir, r.failed = trial, cut
    r.usd, r.tokens, r.partial_usd = None, None, None
    return trial


def _recorded(built: Built, path: Path, variants) -> str:
    """A run dir as harness2 records it: absolute (config.py:91; core/schedule.py:540-541, 628). `moved`: on the
    machine that ran it, not where the folder now is."""
    if "moved" in variants:
        return RAN_AT + "/" + path.relative_to(built.root).as_posix()
    return str(path)


def _one_task(built, agg_dir, adapt, task, task_dir, n, modes, seed, variants, judged_rows, base_rows):
    k, codex = built.k, built.substrate == "cx"
    agg_ns = agg_dir.name
    base_tree = _tree(agg_dir / task_dir / "harnesses" / "base", "Base.", codex=codex)   # recursion.py:366
    bases = []
    for i in range(1, k + 1):                                        # core/schedule.py:606, 619-628
        label = f"base{i}"
        r = Rollout(run_id=f"{agg_ns}/{task}/{label}", task_id=task, mode="agg", label=label, tree=base_tree,
                    parent=None, arm="base", score=None, judged=False, usd=None, tokens=None)
        if built.wb:   # one batch of k, every trial marked with the first base's id (workbuddy.py:826-858)
            first = f"{agg_ns}/{task}/base1"
            _write_wb(built, r, n, label=first.replace("/", "__") + (f"__x{k}" if k > 1 else ""), marker_id=first,
                      suffix=f"B{k - i}x7", at=60.0 * i)   # base1 sorts last: only run_dir maps a base to its trial
        else:
            _write_rollout(built, r, n, variants)
        bases.append(r)
        built.rollouts.append(r)
    out_dir, _ = _rollout_dirs(built, task, bases[0].run_id)
    summary = {"task_id": task, "domain": task.split("/")[0], "k": k, "seed": seed, "status": "ok",
               "base": [b.label if "bare-base" in variants else
                        {"label": b.label, "run_id": b.run_id, "run_dir": _recorded(
                            built, b.run_dir if built.wb else _rollout_dirs(built, task, b.run_id)[0], variants)}
                        for b in bases],
               "modes": {}, "selections": {}, "errors": []}           # core/schedule.py:251-262
    for i, b in enumerate(bases):                                    # cli/judge.py:121-130
        unscored = "base-unscored" in variants and n == 0 and i == 1
        got = None if unscored else {"n_passed": 3 + (i + n) % 2, "n_criteria": 4,
                                     "all_pass": False, "metric": "weighted", "judge_model": built.judge}
        if built.wb and got is not None:      # WorkBuddy's score() (benchmarks/workbuddy.py:986-1004)
            sc = (3 + (i + n) % 2) / 4 - 0.0071
            got = {"n_passed": round(sc * 100), "n_criteria": 100, "all_pass": sc >= 1.0, "raw_score": sc}
        if "bare-base" in variants:
            continue  # cli/judge.py:115-118: no run_dir, never scored
        base_rows.append({"task": task, "label": b.label, "run_id": b.run_id, "score": got})
        b.judged = True
        b.score = None if got is None else got["raw_score"] if built.wb else got["n_passed"] / got["n_criteria"]
    if "base-failed" in variants:
        summary["status"] = "base-failed"
        _json(agg_dir / task_dir / "summary.json", summary)
        return
    for mode in modes:
        mode_ns = ns(built.bench, built.substrate, mode, k, seed)
        work = adapt / mode_ns / task_dir
        if mode == "seq" and "mode-crashed" in variants:
            summary["modes"]["seq"] = {"ok": False, "error": "RuntimeError: editor timed out",
                                       "traceback": "Traceback (most recent call last): ..."}
            summary["status"] = "partial"
            continue
        labels, parents = _stage_plan(mode, k)
        trees = {"base": base_tree if mode == "agg" else _tree(work / "harnesses" / "base", "Base.", codex=codex)}
        stage_rollouts = []
        for label in labels:
            parent = parents[label]
            trees[label] = _tree(work / "harnesses" / label, f"Edit {label}.", codex=codex)  # recursion.py:274-276
            r = Rollout(run_id=f"{mode_ns}/{task}/{label}", task_id=task, mode=mode, label=label,
                        tree=trees[label], parent=trees[parent], arm="candidate", score=None, judged=False,
                        usd=None, tokens=None)
            if "dead-rollout" in variants and n == 0 and label == labels[0]:
                r.dead = True
                r.tree, r.usd, r.tokens = trees[label], None, None
            else:
                cut = "failed-rollout" in variants and n == 0 and mode == "agg" and label == "dom1"
                if built.wb:
                    if "rerun-trial" in variants and n == 0 and mode == "agg" and label == "inv1":
                        _write_wb(built, r, n, label=r.run_id.replace("/", "__"), marker_id=r.run_id, suffix="R0q1",
                                  run="2026-10-17__08-00-00", harness="0123456789abcdef")
                    _write_wb(built, r, n, label=r.run_id.replace("/", "__"), marker_id=r.run_id, suffix="S1q9", cut=cut)
                else:
                    _write_rollout(built, r, n, variants, cut=cut)
            stage_rollouts.append(r)
            built.rollouts.append(r)
        finals = [lb for lb in labels if lb.startswith("agg" if mode == "agg" else "c")]
        ledger = {"mode": mode, "task_id": task, "k": k, "R": len(labels) + k, "expected_R": len(labels) + k,
                  "P": len(labels), "session_attempts": len(labels), "expected_P": len(labels),
                  "usd": 0.21, "rollout_usd": 0.0, "session_usd": 0.21,     # core/ledger.py:115-132
                  "rollouts": [b.label for b in bases] + labels, "sessions": list(labels)}
        summary["modes"][mode] = {
            "mode": mode, "task_id": task, "k": k, "finals": finals, "ok": True, "fell_back": [],
            "kept_base": [], "similarity": None, "ledger": ledger, "errors": [],     # recursion.py:79-105
            "stages": [{"label": lb, "dim": _dim(lb), "fell_back": False, "kept_base": False, "attempts": 1,
                        "usd": 0.035, "cached": False, "notes": "Edited the tally script.", "error": ""}
                       for lb in labels]}
        stale = "stale-judge" in variants and n == 0 and mode == "agg"
        chosen = finals[1] if stale else finals[0]
        fallback = mode == "seq" and "fallback" in variants
        sel = {"task_id": task, "mode": mode, "k": k, "chosen_id": "A", "chosen_label": "" if fallback else chosen,
               "chosen_run_id": "" if fallback else f"{mode_ns}/{task}/{chosen}", "chosen_is_base": False,
               "why": "Matches the source.", "id_map": {"A": finals[0]}, "dropped": [], "n_candidates": len(finals),
               "ok": not fallback, "fallback": fallback, "error": "selector failed" if fallback else "",
               "usd": 0.012, "experience": {}, "trivial": False, "evidence": "deliverable",
               "chosen_run_dir": "" if fallback else _recorded(built, next(
                   r.run_dir for r in stage_rollouts if r.label == chosen) if built.wb else _rollout_dirs(
                   built, task, f"{mode_ns}/{task}/{chosen}")[0], variants),
               "chosen_tree_is_base": False,
               "harness_fingerprint": "" if fallback else fingerprint(trees[chosen])}   # core/selection.py:46-68
        summary["selections"][mode] = sel
        if fallback:
            summary["status"] = "partial"
            judged_rows.append(_judged(task, mode, k, "", "", None, built.judge, "selection did not complete successfully"))
            continue
        adopted = next(r for r in stage_rollouts if r.label == chosen)
        adopted.arm = "adopted"
        if stale:   # judged.json is from the earlier selection of finals[0]; the summary was redone since
            old = next(r for r in stage_rollouts if r.label == finals[0])
            judged_rows.append(_judged(task, mode, k, finals[0], old.run_id, (4, 4), built.judge, "", wb=built.wb))
            continue
        failed = "judge-failed" in variants and n == 0 and mode == "agg"
        row = _judged(task, mode, k, chosen, adopted.run_id, None if failed else (2 + n % 3, 4), built.judge,
                      "TimeoutError: judge" if failed else "", wb=built.wb)
        judged_rows.append(row)
        adopted.judged = True
        adopted.score = row["raw"]["raw_score"] if built.wb and row["graded"] else row["frac"]
    if "legacy-arms" in variants:
        summary["arms"] = summary.pop("modes")
        for sel in summary["selections"].values():
            sel["arm"] = sel.pop("mode")
    _json(agg_dir / task_dir / "summary.json", summary)              # core/schedule.py:289-328, 609


def _judged(task, mode, k, label, run_id, passed, judge, error, *, wb: bool = False) -> dict:
    """One judged.json row: Scored plus frac and graded (core/judge.py:32-53, 146-155). `wb`: raw is
    WorkBuddy's score() {n_passed of 100, n_criteria 100, all_pass, raw_score} (benchmarks/workbuddy.py:986-1004)."""
    n_passed, n_criteria = (passed if passed else (None, None))
    graded = n_criteria is not None and n_criteria > 0
    raw = {} if not graded else {"n_passed": n_passed, "n_criteria": n_criteria, "all_pass": n_passed == n_criteria,
                                 "metric": "weighted", "judge_model": judge}
    if wb and graded:
        sc = n_passed / n_criteria - 0.0033
        n_passed, n_criteria = round(sc * 100), 100
        raw = {"n_passed": n_passed, "n_criteria": n_criteria, "all_pass": sc >= 1.0, "raw_score": sc}
    return {"task_id": task, "mode": mode, "k": k, "label": label, "run_id": run_id, "chosen_is_base": False,
            "n_passed": n_passed, "n_criteria": n_criteria, "all_pass": bool(graded and n_passed == n_criteria),
            "raw": raw, "error": error, "frac": (n_passed / n_criteria) if graded else None, "graded": graded}


def _dim(label: str) -> str:
    return {"inv": "INV", "dom": "DOM", "agg": "AGG", "c": "COMPOSE"}[label.rstrip("0123456789")]


def _stage_plan(mode: str, k: int) -> tuple[list[str], dict[str, str]]:
    """Labels in ledger order and each one's parent tree label.
    Parallel: inv<i>, dom<i> for i in 1..k, then agg<i>; all edit base (recursion.py:366-425).
    Sequential: per round t, inv<t>, dom<t>, c<t>, all editing base (t=1) or c<t-1> (recursion.py:476-562)."""
    if mode == "agg":
        labels = [f"{d}{i}" for i in range(1, k + 1) for d in ("inv", "dom")] + [f"agg{i}" for i in range(1, k + 1)]
        return labels, {lb: "base" for lb in labels}
    labels, parents = [], {}
    for t in range(1, k + 1):
        cur = "base" if t == 1 else f"c{t - 1}"
        for lb in (f"inv{t}", f"dom{t}", f"c{t}"):
            labels.append(lb)
            parents[lb] = cur
    return labels, parents
