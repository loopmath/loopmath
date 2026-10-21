"""A realistic harness2 runs folder: every file harness2 writes for one sweep, written from reading its code.

harness2 is read only (clone at 1cfe254, never run or imported). `build(root, bench=, sub=)` writes what
`harness2-run-sweep --k 2` (both modes, the library on) and then `harness2-judge --modes parallel,sequential`
and `harness2-judge --bases` leave under HARNESS2_RUNS_DIR (h2:harness2/config.py:91) for JobBench with
OpenCode (the CLI's default substrate) or LAB with Codex (the r2 pilot cell): two tasks of one domain, so the
library threads forward; one stage whose first proposal was rejected (two editor attempts); one stage rollout
and one base that died before their runner made a folder; one stage rollout that ran and left no completion
marker; and the sequential mode of task 2 crashed in round 2. `H2REAL.md` lists every file and key written
here with the harness2 line that writes it; `test_h2real.py` reads the files back and holds the two together.
Text inside files (prompts, notes, deliverables) is loopmath's own.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from h2fixture import _dim, _stage_plan, _tree, _write, fingerprint  # noqa: E402
finally:
    sys.path.remove(str(HERE))

K = 2
SEED = None                       # the CLI's default (cli/run_sweep.py:224)
PID = 4242                        # the adapter's process id in per-process names (benchmarks/lab.py:822-826)
STAMP = "20261018T160000Z"        # cli/run_sweep.py:175-177
TASKS = {"jb": ("finance/task1", "finance/task2"),            # <profession>/task<N> (benchmarks/jobbench.py:671-675)
         "lab": ("corporate/nda-review", "corporate/lease-summary")}   # <practice-area>/<slug> (benchmarks/lab.py:540)
EXECUTOR = {"oc": "gemini-3.5-flash", "cx": "gemini-3.5-flash"}
JUDGE = "vertex/claude-judge-model"
JB_JUDGE_SAFE = "claude-judge-model"   # config.JB_JUDGE_MODEL's last part, '.' as '-' (benchmarks/jobbench.py:1011)
CRASH = "OSError: [Errno 28] No space left on device"


@dataclass
class Built:
    root: Path
    bench: str
    sub: str
    files: dict[str, str] = field(default_factory=dict)        # rel path -> H2REAL.md file pattern
    rollout_ids: list[str] = field(default_factory=list)       # every rollout harness2 ran
    dead_id: str = ""
    dead_base_id: str = ""
    unmarked_id: str = ""
    stages: dict[str, dict] = field(default_factory=dict)      # stage rollout run id -> its stage record
    modes_ok: list[tuple[str, str]] = field(default_factory=list)
    selections: dict[str, dict] = field(default_factory=dict)  # "<task>|<mode>" -> summary selection record
    judged: dict[str, dict] = field(default_factory=dict)      # run id -> the judge's raw score dict
    crashed_ids: list[str] = field(default_factory=list)
    crash: dict = field(default_factory=dict)
    namespaces: list[str] = field(default_factory=list)        # agg namespaces, one experiment each
    config: dict = field(default_factory=dict)
    sweep: dict = field(default_factory=dict)

    def applies(self, pattern: str) -> bool:
        """An H2REAL.md file pattern this cell writes: `<B>/...` for both benches, `jb/...` or `lab/...` for one,
        and a trailing `(oc)` or `(cx)` or a `results/<sub>/` part for one substrate."""
        if pattern.split("/", 1)[0] not in ("<B>", self.bench):
            return False
        for sub in ("oc", "cx"):
            if (pattern.endswith(f"({sub})") or f"/results/{sub}/" in pattern) and sub != self.sub:
                return False
        return True

    def put(self, path: Path, pattern: str) -> Path:
        self.files[path.relative_to(self.root).as_posix()] = pattern
        return path


def _j(b: Built, path: Path, value, pattern: str, *, indent: int | None = 2, newline: bool = True,
       sort_keys: bool = False) -> Path:
    _write(path, json.dumps(value, indent=indent, sort_keys=sort_keys) + ("\n" if newline else ""))
    return b.put(path, pattern)


def _t(b: Built, path: Path, text: str, pattern: str) -> Path:
    _write(path, text)
    return b.put(path, pattern)


def _ns(bench: str, sub: str, mode: str) -> str:
    return f"{bench}-{sub}-{mode}{K}"                          # core/schedule.py:111-127, 122-126


def _key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]   # core/proposer.py:507-534


def build(root: Path, *, bench: str = "jb", sub: str = "oc") -> Built:
    assert (bench, sub) in (("jb", "oc"), ("lab", "cx")), (bench, sub)
    root = Path(root)
    b = Built(root=root, bench=bench, sub=sub)
    adapt = root / bench / "adapt"                                # cli/_common.py:70-72
    agg_ns = _ns(bench, sub, "agg")
    b.namespaces.append(agg_ns)
    tasks = TASKS[bench]
    b.config = {"method_version": 2, "executor": EXECUTOR[sub], "improver": "claude-improver-model",
                "judge": JUDGE, "project": "example-project", "location": "us-east5",
                "judge_endpoint": "", "endpoint": "https://vertex.example/v1" if sub == "cx" else "",
                "split": "test", "tasks": [[t.split("/")[0], t] for t in tasks], "max_tasks": None,
                "domains": [], "seed": SEED}                     # cli/run_sweep.py:209-224
    _j(b, adapt / agg_ns / "config.json", b.config, "<B>/adapt/<agg ns>/config.json")   # :227-235
    if bench == "lab":                                           # the base harness, once per process (lab.py:822-842)
        _tree(root / "lab" / f"_base_harness_cx_{PID}", "Base.", codex=True)
        b.put(root / "lab" / f"_base_harness_cx_{PID}", "lab/_base_harness_cx_<pid>/")
    sweep_tasks = []
    judged_rows, base_rows = [], []
    for n, task in enumerate(tasks):
        sweep_tasks.append(_one_task(b, adapt, task, n, judged_rows, base_rows))
    # harness2-judge --modes parallel,sequential: judged.json, merged and sorted by (task, mode, k)
    judged_rows.sort(key=lambda r: (r["task_id"], r["mode"], r["k"]))            # cli/judge.py:65-81
    # a judge row carries the bench's own score dict, so each bench's judge files have their own pattern
    _j(b, adapt / agg_ns / "judged.json", judged_rows, f"{bench}/adapt/<agg ns>/judged.json")  # core/judge.py:146-155
    _j(b, adapt / agg_ns / "base_judged.json", base_rows, f"{bench}/adapt/<agg ns>/base_judged.json", indent=1,
       newline=False)                                                             # cli/judge.py:143-144
    b.sweep = {"benchmark": bench, "substrate": sub, "k": K, "modes": ["agg", "seq"], "seed": SEED, "tag": "",
               "domain_order": [tasks[0].split("/")[0]], "max_tasks": None, "only_tasks": [],
               "started_at": "2026-10-18T15:00:00+00:00", "finished_at": "2026-10-18T16:00:00+00:00",
               "n_domains": 1,
               "domains": [{"domain": tasks[0].split("/")[0], "n_tasks": len(tasks), "seed": SEED,
                            "done": sum(t["status"] == "ok" for t in sweep_tasks),
                            "failed": sum(t["status"] != "ok" for t in sweep_tasks), "tasks": sweep_tasks}]}
    _j(b, adapt / "_sweeps" / f"{agg_ns}-{STAMP}.json", b.sweep, "<B>/adapt/_sweeps/<agg ns>-<stamp>.json",
       newline=False)                                                             # core/schedule.py:837-849
    return b


def _one_task(b: Built, adapt: Path, task: str, n: int, judged_rows: list, base_rows: list) -> dict:
    bench, sub = b.bench, b.sub
    codex = sub == "cx"
    agg_ns = _ns(bench, sub, "agg")
    tdir = task.replace("/", "__")                               # core/schedule.py:101-102
    work = adapt / agg_ns / tdir
    domain = task.split("/")[0]
    if bench == "jb":                                            # the shared task inputs (jobbench.py:692-715)
        _write(b.root / "jb" / "dataset" / task / "task_folder" / "TASK_INSTRUCTIONS.txt", "Reconcile the ledger.\n")
        b.put(b.root / "jb" / "dataset" / task / "task_folder", "jb/dataset/<task>/task_folder/")
    bases = []
    for i in range(1, K + 1):                                    # core/schedule.py:619-628
        label, rid = f"base{i}", f"{agg_ns}/{task}/base{i}"
        b.rollout_ids.append(rid)
        dead = n == 1 and i == 2
        if dead:                                                 # died before its runner made a folder
            b.dead_base_id = rid
            run_dir = str(work / "missing")                      # core/schedule.py:622, 424-427
        else:
            run_dir = str(_rollout(b, task, rid, None, n * 10 + i))
        bases.append({"label": label, "run_id": rid, "run_dir": run_dir})
    summary = {"task_id": task, "domain": domain, "k": K, "seed": SEED, "status": "partial", "base": bases,
               "modes": {}, "selections": {}, "errors": []}     # core/schedule.py:251-262
    lib_before = {}
    for mode in ("agg", "seq"):
        lib_before[mode] = n if mode == "agg" else min(n, 1)
        crashed = mode == "seq" and n == 1
        rec, sel = _mode(b, adapt, task, n, mode, crashed)
        summary["modes"][mode] = rec
        if crashed:
            summary["errors"].append(f"seq: {CRASH}")            # core/schedule.py:660
            continue
        summary["selections"][mode] = sel
        b.selections[f"{task}|{mode}"] = sel
        b.modes_ok.append((task, mode))
        raw = _judge(b, task, sel["chosen_run_id"], n * 3 + (mode == "seq"))
        judged_rows.append({"task_id": task, "mode": mode, "k": K, "label": sel["chosen_label"],
                            "run_id": sel["chosen_run_id"], "chosen_is_base": False, "n_passed": raw["n_passed"],
                            "n_criteria": raw["n_criteria"], "all_pass": raw["all_pass"], "raw": raw, "error": "",
                            "frac": raw["n_passed"] / raw["n_criteria"], "graded": True})   # core/judge.py:32-53, 151
        b.judged[sel["chosen_run_id"]] = raw
    summary["status"] = "partial" if summary["errors"] else "ok"   # core/schedule.py:265-279, 664
    for i, base in enumerate(bases):                             # cli/judge.py:108-147
        alive = base["run_id"] != b.dead_base_id
        raw = _judge(b, task, base["run_id"], 20 + n * 2 + i) if alive else None
        base_rows.append({"task": task, "label": base["label"], "run_id": base["run_id"], "score": raw})
        if raw is not None:
            b.judged[base["run_id"]] = raw
    _j(b, work / "summary.json", summary, "<B>/adapt/<agg ns>/<task dir>/summary.json", newline=False)   # :324-326
    _t(b, work / "summary.lock", "", "<B>/adapt/<agg ns>/<task dir>/summary.lock")                   # :296-298
    row = {"task": task, "status": summary["status"], "library_entries_before": lib_before}
    if summary["errors"]:
        row["error"] = "; ".join(summary["errors"])              # core/schedule.py:722-725
    return row


def _mode(b: Built, adapt: Path, task: str, n: int, mode: str, crashed: bool) -> tuple[dict, dict | None]:
    bench, sub, codex = b.bench, b.sub, b.sub == "cx"
    mode_ns = _ns(bench, sub, mode)
    work = adapt / mode_ns / task.replace("/", "__")
    trees = {"base": _tree(work / "harnesses" / "base", "Base.", codex=codex)}   # recursion.py:366, 480-482
    b.put(work / "harnesses" / "base", "<B>/adapt/<mode ns>/<task dir>/harnesses/<label>/")
    labels, parents = _stage_plan(mode, K)
    if crashed:                                                  # round 2 failed writing its evidence
        labels = [lb for lb in labels if lb.endswith("1")]
    stages, sessions, rolled = [], [], []
    for j, label in enumerate(labels):
        rid = f"{mode_ns}/{task}/{label}"
        b.rollout_ids.append(rid)
        retry = n == 0 and mode == "agg" and label == "dom1"
        ctx = work / "context" / label                           # recursion.py:156; context.py:126-136
        _write(ctx / "task.md", f"# {task}\n")
        _tree(ctx / "harness", "Base.", codex=codex)
        _write(ctx / "rollouts" / "base1.md", "The base rollout's trajectory.\n")
        if n == 1:
            _write(ctx / "library" / "INDEX.md", "# Library\n")  # context.py:160-165
        b.put(ctx, "<B>/adapt/<mode ns>/<task dir>/context/<stage label>/")
        for a in range(1, 3 if retry else 2):                    # recursion.py:191-200: <label>.a<attempt>
            _audit(b, adapt, f"{label}.a{a}", str(ctx), harvest=True, ok=True)
        trees[label] = _tree(work / "harnesses" / label, f"Edit {label} of {task}.", codex=codex)   # :274-276
        b.put(work / "harnesses" / label, "<B>/adapt/<mode ns>/<task dir>/harnesses/<label>/")
        stage = {"label": label, "dim": _dim(label), "fell_back": False, "kept_base": False,
                 "attempts": 2 if retry else 1, "usd": round(0.031 + 0.002 * j, 6), "cached": False,
                 "notes": "Tightened the tally script's rounding." if not retry else
                 "Second proposal: kept the guardrails and changed the script only.",
                 "error": ""}                                    # recursion.py:88-99
        stages.append(stage)
        sessions.append(label)
        rolled.append(label)
        if crashed:
            b.crashed_ids.append(rid)
        else:
            b.stages[rid] = stage
        if n == 0 and mode == "agg" and label == "inv1":
            b.dead_id = rid                                      # died before its runner made a folder
            b.stages[rid] = stage
            continue
        cut = n == 0 and mode == "agg" and label == "dom2"
        if cut:
            b.unmarked_id = rid
        _rollout(b, task, rid, trees[label], 100 + n * 20 + j, cut=cut)
    if crashed:
        trees["c1"] = _tree(work / "harnesses" / "c1", f"Edit c1 of {task}.", codex=codex)   # rewritten, :480-482
        b.crash = {"ok": False, "error": CRASH, "traceback": "Traceback (most recent call last):\n  ...\n" + CRASH}
        return b.crash, None                                     # core/schedule.py:655-659
    finals = [lb for lb in labels if lb.startswith("agg" if mode == "agg" else "c")]
    chosen = finals[1] if n == 0 else finals[0]
    id_map = {"A": chosen, "B": next(f for f in finals if f != chosen)}            # selection.py:71-76
    sel_dir = work / "selection" / f"{mode}{K}"                  # selection.py:296-305
    _t(b, sel_dir / "task.md", f"# {task}\n", "<B>/adapt/<mode ns>/<task dir>/selection/<mode><k>/task.md")
    _write(sel_dir / "source" / "brief.txt", "The task's files.\n")
    b.put(sel_dir / "source", "<B>/adapt/<mode ns>/<task dir>/selection/<mode><k>/source/")
    for anon, lb in id_map.items():
        _t(b, sel_dir / "candidates" / f"{anon}.md", f"<!-- candidate {anon} -->\nThe deliverable of one final.\n",
           "<B>/adapt/<mode ns>/<task dir>/selection/<mode><k>/candidates/<anon>.md")
    _audit(b, adapt, f"select.{mode}{K}", str(sel_dir), harvest=False, ok=True)
    if bench == "jb":
        chosen_dir = b.root / "jb" / "dataset" / task / "model_output" / f"{mode_ns}/{task}/{chosen}".replace("/", "__")
    else:
        chosen_dir = b.root / "lab" / "results" / sub / mode_ns / task / chosen
    experience = {"cue": "A ledger task with signed amounts", "lesson": "Round once, at the end.",
                  "evidence": "The chosen final's totals matched the source.", "confidence": "medium",
                  "edits": "scripts/tally.py: round at the end."}                  # selection.py:494-503
    sel = {"task_id": task, "mode": mode, "k": K, "chosen_id": "A", "chosen_label": chosen,
           "chosen_run_id": f"{mode_ns}/{task}/{chosen}", "chosen_is_base": False,
           "why": "A's totals match the source documents.", "id_map": id_map, "dropped": [], "n_candidates": 2,
           "ok": True, "fallback": False, "error": "", "usd": 0.024, "experience": experience, "trivial": False,
           "evidence": "deliverable", "chosen_run_dir": str(chosen_dir)}           # selection.py:46-68
    sel_file = dict(sel)                                         # selection.py:255-262, rewritten at 450
    _j(b, sel_dir / "selection.json", sel_file, "<B>/adapt/<mode ns>/<task dir>/selection/<mode><k>/selection.json",
       newline=False)
    sel["chosen_tree_is_base"] = False                           # core/schedule.py:546
    sel["harness_fingerprint"] = fingerprint(trees[chosen])      # :547-548
    ref = work / "reflection" / f"{mode}{K}"                     # selection.py:416-443
    _write(ref / "task.md", f"# {task}\n")
    _tree(ref / "harness", f"Edit {chosen} of {task}.", codex=codex)
    _tree(ref / "base_harness", "Base.", codex=codex)
    b.put(ref, "<B>/adapt/<mode ns>/<task dir>/reflection/<mode><k>/")
    _j(b, ref / "adoption.json", {"chosen_label": chosen, "candidate_ids": id_map, "reason": sel["why"]},
       "<B>/adapt/<mode ns>/<task dir>/reflection/<mode><k>/adoption.json")
    exp_dir = work / "reflection" / f"{mode}{K}.experience"      # selection.py:477-479
    _write(exp_dir / "task.md", f"# {task}\n")
    _write(exp_dir / "adoption.json", json.dumps({"chosen_label": chosen}) + "\n")
    b.put(exp_dir, "<B>/adapt/<mode ns>/<task dir>/reflection/<mode><k>.experience/")
    _audit(b, adapt, f"experience.{mode}{K}", str(exp_dir), harvest=False, ok=True)
    _t(b, ref / "edits.md", experience["edits"] + "\n", "<B>/adapt/<mode ns>/<task dir>/reflection/<mode><k>/edits.md")
    _library(b, task, mode, chosen, trees[chosen], sel)
    sim = {"n": 2, "n_distinct": 2, "mean_line_jaccard": 0.8125} if mode == "agg" else None   # tree.py:411, 424-426
    R = K + len(labels)
    ledger = {"mode": mode, "task_id": task, "k": K, "R": R, "expected_R": 4 * K, "P": len(sessions),
              "session_attempts": sum(s["attempts"] for s in stages), "expected_P": 3 * K,
              "usd": round(sum(s["usd"] for s in stages) + sel["usd"], 6), "rollout_usd": 0.0,
              "session_usd": round(sum(s["usd"] for s in stages) + sel["usd"], 6),
              "rollouts": [f"base{i}" for i in range(1, K + 1)] + rolled, "sessions": sessions}   # ledger.py:115-132
    rec = {"mode": mode, "task_id": task, "k": K, "finals": finals, "ok": True, "fell_back": [], "kept_base": [],
           "stages": stages, "similarity": sim, "ledger": ledger, "errors": []}    # recursion.py:79-105
    return rec, sel


def _library(b: Built, task: str, mode: str, chosen: str, tree: Path, sel: dict) -> None:
    """library.py:115-181, under the library root cli/_common.py:75-88."""
    lib = b.root / b.bench / "library" / b.bench / f"k{K}-{b.sub}" / mode / task.split("/")[0]
    slug = task.replace("/", "__")                               # library.py:39-40
    entry = lib / "harnesses" / slug
    _tree(entry / "harness", f"Edit {chosen} of {task}.", codex=b.sub == "cx")
    b.put(entry / "harness", "<B>/library/<name>/k<k>-<sub>/<mode>/<domain dir>/harnesses/<slug>/harness/")
    _t(b, entry / "EXPERIENCE.md", f"# {task}\n\n- adopted: `{chosen}` (run `{sel['chosen_run_id']}`)\n",
       "<B>/library/<name>/k<k>-<sub>/<mode>/<domain dir>/harnesses/<slug>/EXPERIENCE.md")
    _t(b, lib / "INDEX.md", f"# Library: {task.split('/')[0]}\n",
       "<B>/library/<name>/k<k>-<sub>/<mode>/<domain dir>/INDEX.md")


def _audit(b: Built, adapt: Path, label: str, context: str, *, harvest: bool, ok: bool) -> None:
    """core/proposer.py:313-386: one flat folder per sweep, `<safe label>.<cache key>.<kind>`."""
    d = adapt / "_audit" / _ns(b.bench, b.sub, "agg")
    stem = f"{label}.{_key(label, context)}"
    pat = "<B>/adapt/_audit/<agg ns>/<safe label>.<key>"
    _t(b, d / f"{stem}.mission.md", f"Edit the harness for {label}.\n", pat + ".mission.md")
    _t(b, d / f"{stem}.response.md", "<notes>Edited.</notes>\n", pat + ".response.md")
    _j(b, d / f"{stem}.meta.json", {"label": label, "ok": ok, "channel": "claude_cli", "model": "claude-improver-model",
                                    "effort": "high", "attempts": 1, "wall_s": 41.2, "cost_usd": 0.031, "error": ""},
       pat + ".meta.json")
    if harvest:
        _tree(d / f"{stem}.workspace", f"Edit {label}.", codex=b.sub == "cx")
        b.put(d / f"{stem}.workspace", pat + ".workspace/")


def _judge(b: Built, task: str, rid: str, n: int) -> dict:
    """The judge's files for one judged rollout, and the raw dict harness2's score() returns for it."""
    if b.bench == "jb":                                          # benchmarks/jobbench.py:590-619, 1008-1014
        label = rid.replace("/", "__")
        passed, count = 3 + n % 3, 6
        details = {"rubrics": [{"result": {"criteria_count": 3, "criteria_passed": min(3, passed),
                                           "overall_reasoning": "Totals reconcile."}},
                               {"result": {"criteria_count": 3, "criteria_passed": max(0, passed - 3),
                                           "overall_reasoning": "One rounding slip."}}],
                   "total_score": float(passed * 2), "max_score": 12.0, "judge_model": JUDGE, "total_count": 2}
        _j(b, b.root / "jb" / "dataset" / task / "eval_result" / f"eval_{label}" / f"{JB_JUDGE_SAFE}_judge.json",
           details, "jb/dataset/<task>/eval_result/eval_<label>/<safe judge>_judge.json")
        return {"n_passed": float(passed * 2), "n_criteria": 12.0, "all_pass": passed == count, "metric": "weighted",
                "criteria_passed": passed, "criteria_count": count, "judge_model": JUDGE}
    run = b.root / "lab" / "results" / b.sub / rid             # benchmarks/lab.py:961-1002
    got = {"n_passed": 4 + n % 4, "n_criteria": 8, "all_pass": False}
    _j(b, run / "scores.json", got, "lab/results/cx/<run_id>/scores.json")
    _t(b, run / "judge.log", "judging 8 criteria\ndone\n", "lab/results/cx/<run_id>/judge.log")
    return got


def _rollout(b: Built, task: str, rid: str, tree: Path | None, n: int, *, cut: bool = False) -> Path:
    """One rollout's folders, as the bench's runner and execute leave them. `cut`: the agent delivered
    nothing and the runner exited non-zero, so execute raised before mark_complete (JB:
    jb_opencode_runner.py:260-279, jobbench.py:747-756; LAB: run_codex_lab.py:323-363, lab.py:419-420, 618)."""
    if b.bench == "jb":
        return _jb_oc(b, task, rid, tree, n, cut)
    return _lab_cx(b, task, rid, tree, n, cut)


def _mark(b: Built, d: Path, rid: str, tree: Path | None, out: Path | None, n: int, pattern: str) -> None:
    """core/runmark.py:64-81: indent 2, sorted keys, written only after a normal return."""
    _j(b, d / ".adapt_complete", {"version": 1, "run_id": rid, "at": 1760803200.0 + n,
                                  "harness": "base" if tree is None else fingerprint(tree),
                                  "output_fingerprint": None if out is None else fingerprint(out)},
       pattern, sort_keys=True)


def _jb_oc(b: Built, task: str, rid: str, tree: Path | None, n: int, cut: bool) -> Path:
    label = rid.replace("/", "__")                                # jobbench.py:721
    store = b.root / "jb" / "dataset" / task
    out, traj = store / "model_output" / label, store / "model_traj" / label   # jobbench.py:676-680
    task_name = task.replace("/", "_")                          # jb_opencode_runner.py:222
    stamp = f"20261018_09{n % 60:02d}00"
    log = b.root / "jb" / "dataset" / ".harness2-logs" / f"hevo_{task_name}_{label}_{stamp}.log"   # :95, 225
    _t(b, log, f"[09:00:00] run {task} ({label}) attempt 1\n", "jb/dataset/.harness2-logs/hevo_<task_name>_<label>_<stamp>.log (oc)")
    pat = "jb/dataset/<task>/model_traj/<label>/<task_name>_<label>_attempt<N>_<stamp>.jsonl (oc)"
    for attempt in (1, 2) if cut else (1,):                      # MAX_RETRIES 2 (jobbench.py:801); :239-244
        _t(b, traj / f"{task_name}_{label}_attempt{attempt}_{stamp}.jsonl", _oc_lines(n, attempt, cut), pat)
    if cut:
        return traj
    _write(out / "answer.md", f"Total for {task}: {100 + n}\n")  # jb_opencode_runner.py:250-256
    b.put(out, "jb/dataset/<task>/model_output/<label>/")
    _mark(b, traj, rid, tree, out, n, "jb/dataset/<task>/model_traj/<label>/.adapt_complete")
    return out


def _oc_lines(n: int, attempt: int, cut: bool) -> str:
    """`opencode run --format json` lines, filtered to `{"type":` (jb_opencode_runner.py:32, 203-206). Fields as
    harness2's readers take them: lab_runner/opencode/session.py:38-59, 81; substrate/opencode.py:339-386."""
    ts = 1760803200000 + 60000 * n + 30000 * (attempt - 1)
    ses = f"ses_{n:03d}{attempt}"
    lines = [{"type": "step_start", "timestamp": ts, "sessionID": ses, "part": {"type": "step-start"}},
             {"type": "tool_use", "timestamp": ts + 300, "sessionID": ses,
              "part": {"type": "tool", "tool": "read", "state": {"status": "completed",
                                                                 "input": {"filePath": "task_folder/ledger.csv"},
                                                                 "output": "date,amount\n"}}},
             {"type": "text", "timestamp": ts + 700, "sessionID": ses,
              "part": {"type": "text", "text": "I will tally the ledger."}},
             {"type": "step_finish", "timestamp": ts + 900, "sessionID": ses,
              "part": {"type": "step-finish", "cost": 0.0125,
                       "tokens": {"input": 1200 + n, "output": 80, "reasoning": 20, "cache": {"read": 400, "write": 10}}}}]
    if cut:
        lines.append({"type": "error", "timestamp": ts + 1200, "sessionID": ses,
                      "error": {"data": {"message": "quota exceeded"}}})   # substrate/opencode.py:356-363
    return "".join(json.dumps(x) + "\n" for x in lines)


def _lab_cx(b: Built, task: str, rid: str, tree: Path | None, n: int, cut: bool) -> Path:
    run = b.root / "lab" / "results" / "cx" / rid               # lab.py:529, 585-586
    pat = "lab/results/cx/<run_id>/"
    hdir = str(b.root / "lab" / f"_base_harness_cx_{PID}") if tree is None else f"/tmp/adapt-cx-root-{n:04d}"
    scratch = f"{b.root}/lab/scratch/lab/{rid.replace('/', '__')}__{PID}/ws"
    _j(b, run / "config.json", {"runner": "codex", "codex_bin": "/usr/local/bin/codex", "codex_version": "codex-cli 0.144.6",
                                "model": EXECUTOR["cx"], "model_chain": "codex -> https://vertex.example/v1 (Responses API)",
                                "reasoning_effort": "high", "task": task, "run_id": rid, "timeout": 3600,
                                "harness_dir": hdir, "harness_files": ["AGENTS.md", ".agents/skills/tally/SKILL.md",
                                                                       "scripts/tally.py"],
                                "doc_parser": "sandbox", "workspace": scratch,
                                "codex_home": f"{b.root}/lab/scratch/codex_home/{rid.replace('/', '__')}__{PID}",
                                "started_at": f"2026-10-18T09:{n % 60:02d}:00+00:00"},
       pat + "config.json")                                      # run_codex_lab.py:264-281
    _t(b, run / "codex_session.jsonl", _cx_lines(n, cut), pat + "codex_session.jsonl")   # :306-310
    _t(b, run / "codex_stderr.log", "", pat + "codex_stderr.log")                      # :308
    out = run / "output"                                         # :313-319
    out.mkdir(parents=True, exist_ok=True)
    if not cut:
        _write(out / "memo.md", f"Memo for {task}: {100 + n}\n")
    b.put(out, pat + "output/")
    usage = _cx_usage(n)
    _j(b, run / "metrics.json", {"runner": "codex", "model": EXECUTOR["cx"], "task": task, "run_id": rid,
                                 "turn_count": 1, "tool_calls": 1, "stream_errors": 1 if cut else 0,
                                 "input_tokens": usage["input_tokens"],
                                 "input_tokens_note": "As reported by the configured Codex endpoint.",
                                 "cached_input_tokens": usage["cached_input_tokens"],
                                 "output_tokens": usage["output_tokens"],
                                 "reasoning_tokens": usage["reasoning_output_tokens"],
                                 "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                                 "wall_clock_seconds": 312.5, "finished_cleanly": not cut, "exit_code": 1 if cut else 0,
                                 "deliverables": [] if cut else ["memo.md"], "documents_read": 2, "total_documents": 3,
                                 "harness_dir": hdir, "harness_files": ["AGENTS.md", ".agents/skills/tally/SKILL.md",
                                                                        "scripts/tally.py"],
                                 "workspace_scratch_files": ["notes.txt"],
                                 "completed_at": f"2026-10-18T09:{n % 60:02d}:30+00:00"},
       pat + "metrics.json")                                     # run_codex_lab.py:335-359
    if not cut:
        _mark(b, run, rid, tree, out, n, pat + ".adapt_complete")    # lab.py:618-623
    return run


def _cx_usage(n: int) -> dict:
    return {"input_tokens": 3000 + n, "cached_input_tokens": 1000, "output_tokens": 200 + n,
            "reasoning_output_tokens": 50}


def _cx_lines(n: int, cut: bool) -> str:
    """`codex exec --json` stdout (run_codex_lab.py:150-187, 306-310; substrate/codex.py:300-346, 432-443)."""
    lines = [{"type": "thread.started", "thread_id": f"th_{n:04d}"}, {"type": "turn.started"},
             {"type": "item.completed", "item": {"id": "item_0", "type": "command_execution",
                                                 "command": "python scripts/tally.py 1 2", "aggregated_output": "3\n",
                                                 "exit_code": 0, "status": "completed"}},
             {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "Memo written."}}]
    if cut:
        lines.append({"type": "item.completed", "item": {"id": "item_2", "type": "error",
                                                         "message": "stream disconnected"}})
    lines.append({"type": "turn.completed", "usage": _cx_usage(n)})
    return "".join(json.dumps(x) + "\n" for x in lines)
