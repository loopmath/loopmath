"""A user brings designed runs, fits, asks `recommend` and reads `status`: the benchmark journey, synthetic.

A benchmark study hands the user designed runs: gpt-5.6-sol and gpt-5.6-luna through codex at efforts low to
max, solo runs, a planner with 3 workers and 3 workers with no selector, each document's task.source naming a
source the package also ships (rq1). This module builds that shape from made-up numbers, 8 configurations on
two tasks of one (type, repo), and checks what 0.1.1 promises:

- `recommend --models` naming the runs' models lists every recorded configuration as a candidate, `max` effort
  included (0.2.2: without it those models are retired, so their workflows are at most the reference);
- with no habit, the reference is the best recorded workflow, and nothing is called "your usual";
- `fit` counts store runs as the user's, whatever their task.source says, and `--without rq1` keeps them;
- `status` names the current fit's options and counts;
- `plan` is gone from `--help` but still answers, with a note on stderr.

Nothing here comes from a real store: every document is built below.
"""

from __future__ import annotations

import contextlib
import io
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.workflows.ids import config_id_ocp

REPO = "example/bench"
TARGET = "heldout_perf>=2400"
SHIPPED_SOURCE = "rq1"
SOL, LUNA = "gpt-5.6-sol", "gpt-5.6-luna"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
TASKS = (("bench/t1", "t1"), ("bench/t2", "t2"))
# (arm, shape, {piece: (model, effort)}, heldout_perf on each task, dollars per attempt)
ARMS = [(f"sol-{e}", "solo", {"implement": (SOL, e)}, (2150 + 90 * i, 2050 + 100 * i), 2.0 + 2.5 * i)
        for i, e in enumerate(EFFORTS)]
ARMS += [("luna-high", "solo", {"implement": (LUNA, "high")}, (2000, 1900), 0.8),
         ("plan-3", "plan_3", {"plan": (SOL, "max"), "work": (LUNA, "high")}, (2600, 2450), 1.2),
         ("best-of-3", "best_of_3", {"implement": (SOL, "xhigh")}, (2550, 2500), 6.0)]


def _piece(pid: str, role: str, width: int = 1) -> dict:
    return {"id": pid, "role": role, "width": width}


def _workflow(shape: str) -> dict:
    control = {"gates": [], "repair": {}, "budget": 1, "rescue": {"kind": "none"}}
    issue_diff = [{"id": "issue", "kind": "issue"}, {"id": "diff", "kind": "diff"}]
    if shape == "solo":
        return {"id": "solo", "version": 1, "title": "Solo", "pieces": [_piece("implement", "implementer")],
                "artifacts": issue_diff, "edges": [["issue", "implement"], ["implement", "diff"]], "control": control}
    if shape == "best_of_3":  # three workers, no selector piece
        return {"id": "best_of_n", "version": 1, "title": "Best of 3, no selector",
                "pieces": [_piece("implement", "implementer", 3)], "artifacts": issue_diff,
                "edges": [["issue", "implement"], ["implement", "diff"]], "control": control}
    return {"id": "plan_implement", "version": 1, "title": "Planner and 3 workers",
            "pieces": [_piece("plan", "planner"), _piece("work", "worker", 3)],
            "artifacts": [{"id": "issue", "kind": "issue"}, {"id": "plan_doc", "kind": "plan"},
                          {"id": "diff", "kind": "diff"}],
            "edges": [["issue", "plan"], ["plan", "plan_doc"], ["plan_doc", "work"], ["work", "diff"]],
            "control": control}


def configuration(shape: str, settings: dict) -> dict:
    workflow = _workflow(shape)
    ocp_settings = {piece: {"harness": "codex", "model": {"raw": model, "id": model}, "effort": effort,
                            "context_policy": "fresh", "options": {}} for piece, (model, effort) in settings.items()}
    return {"id": config_id_ocp(workflow, ocp_settings), "workflow": workflow, "settings": ocp_settings,
            "source": "designed"}


def designed_doc(arm: str, shape: str, settings: dict, perf: int, usd: float, task: tuple[str, str],
                 started: datetime) -> dict:
    subtype, name = task
    config = configuration(shape, settings)
    run_id = f"run_scenario-{name}-{arm}"
    start, end = started.isoformat(timespec="seconds"), (started + timedelta(minutes=50)).isoformat(timespec="seconds")
    attempts = [{"id": f"{p['id']}-{k + 1}.a1", "node": p["id"], "n": k + 1, "harness": "codex",
                 "model": {"raw": settings[p["id"]][0], "id": settings[p["id"]][0], "provider": "openai"},
                 "effort": settings[p["id"]][1], "status": "done", "started_at": start, "ended_at": end,
                 "vertex": p["id"], "round": 1, "outcome": {"result": "done", "evidence": "verified"},
                 "cost": {"input_tokens": 100000, "cached_input_tokens": 900000, "output_tokens": 40000,
                          "reasoning_tokens": 15000, "usd": usd, "basis": "measured"}}
                for p in config["workflow"]["pieces"] for k in range(p["width"])]
    judge = {"kind": "judge", "ref": "bench"}
    signals = [{"id": f"sig_{run_id}_perf", "kind": "score", "name": "heldout_perf", "value": perf, "observed_at": end,
                "source": judge, "tier": "verified", "unit": "perf", "better": "higher", "target": 2400.0,
                "scale": "linear"},
               {"id": f"sig_{run_id}_accepted", "kind": "verdict", "name": "accepted",
                "value": "pass" if perf >= 2400 else "fail", "observed_at": end, "source": judge, "tier": "verified"}]
    return {
        "ocp": "0.3",
        "producer": {"name": "scenario-bench", "version": "1", "emitted_at": end,
                     "capabilities": {"task": True, "configuration": True, "signals": True, "slate": False,
                                      "receipt": False}},
        "privacy": {"profile": "metadata_only"},
        "run": {"id": run_id, "started_at": start, "ended_at": end,
                "task": {"id": f"{REPO}/{name}", "type": "feature", "subtype": subtype, "repo": REPO,
                         "features": {"lang": "cpp"}, "source": {"kind": SHIPPED_SOURCE, "ref": "synthetic scenario"},
                         "labeled_by": {"how": "orchestrator", "tier": "verified"}},
                "configuration": config,
                "provenance": {"kind": "designed", "chooser": "planner"},
                "acceptance_rule": {"name": TARGET, "definition": "held-out performance is at least 2400",
                                    "requires": [], "score": {"name": "heldout_perf", "target": 2400.0,
                                                              "better": "higher", "scale": "linear"}},
                "signals": signals},
        "nodes": [{"id": p["id"], "kind": "plan" if p["role"] == "planner" else "impl", "state": "done",
                   "vertex": p["id"]} for p in config["workflow"]["pieces"]],
        "edges": [], "attempts": attempts, "artifacts": [], "events": [],
    }


def write_docs(folder: Path) -> list[Path]:
    now = datetime.now().astimezone()
    paths = []
    for t, task in enumerate(TASKS):
        for i, (arm, shape, settings, perfs, usd) in enumerate(ARMS):
            doc = designed_doc(arm, shape, settings, perfs[t], usd, task, now - timedelta(hours=40 - 2 * i - 20 * t))
            path = folder / f"{doc['run']['id']}.ocp.json"
            path.write_text(json.dumps(doc, indent=1))
            paths.append(path)
    return paths


def call(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(list(argv))
        except SystemExit as exc:  # --help
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def call_json(*argv: str) -> dict:
    code, out, err = call(*argv, "--json")
    assert code == 0, err
    return json.loads(out)


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    """The whole journey once, in order; each test reads its step."""
    root = tmp_path_factory.mktemp("designed")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOOPMATH_HOME", str(root / "home"))
        mp.setenv("LOOPMATH_CACHE_DIR", str(root / "cache"))
        mp.setenv("CLAUDE_CONFIG_DIR", str(root / "logs" / "claude"))  # no session logs to match
        mp.setenv("CODEX_HOME", str(root / "logs" / "codex"))
        import loopmath.store.fitjob as fitjob

        mp.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": False, "reason": "scenario"})
        (root / "docs").mkdir()
        docs = write_docs(root / "docs")
        code, out, err = call("ocp", "validate", *map(str, docs))
        assert code == 0, out + err
        for path in docs:
            code, out, err = call("run", "import", str(path), "--finish")
            assert code == 0, err
        steps = {"docs": [json.loads(p.read_text()) for p in docs], "home": root / "home"}
        steps["fit"] = call_json("fit")
        steps["status"] = call("status")[1]
        # 0.2.2: gpt-5.6 models are retired unless named; this benchmark user names the models they ran
        rec = ("recommend", "--type", "feature", "--repo", REPO, "--target", TARGET, "--models", f"{SOL},{LUNA}")
        steps["rec"] = call_json(*rec)
        steps["rec_text"] = call(*rec)[1]
        steps["stored"] = json.loads((root / "home" / "recs" / f"{steps['rec']['rec']}.json").read_text())
        steps["help"] = call("--help")[1]
        (root / "backlog.jsonl").write_text(json.dumps({"type": "feature", "repo": REPO, "title": "next task"}) + "\n")
        # the flags an experiment runner uses; stdout stays one JSON object
        steps["plan"] = call("plan", "--backlog", str(root / "backlog.jsonl"), "--budget-usd", "40", "--slate-size", "2",
                             "--members", "grid", "--max-slates", "1", "--target", TARGET, "--json")
        steps["fit_without"] = call_json("fit", "--without", SHIPPED_SOURCE)
        steps["status_without"] = call("status")[1]
        yield steps


def recorded_ids(journey) -> set[str]:
    return {d["run"]["configuration"]["id"] for d in journey["docs"]}


def test_the_store_holds_eight_recorded_configurations(journey):
    ids = recorded_ids(journey)
    assert len(journey["docs"]) == 16 and len(ids) == 8


def test_recommend_lists_every_recorded_configuration_including_max(journey):
    origins = {c["config"]["id"]: c["origin"] for c in journey["stored"]["candidates"]}
    missing = recorded_ids(journey) - set(origins)
    assert not missing, f"recorded configurations that are not candidates: {sorted(missing)}"
    assert {origins[i] for i in recorded_ids(journey)} == {"recorded"}
    efforts = {c["config"]["settings"]["implement"]["effort"] for c in journey["stored"]["candidates"]
               if c["origin"] == "recorded" and c["config"]["workflow"]["id"] == "solo"}
    assert "max" in efforts


def test_the_reference_is_the_best_recorded_workflow_and_nothing_is_your_usual(journey):
    rec = journey["rec"]
    assert rec["schema"] == "loopmath.recommend/2" and rec["usual"] is None  # designed runs are not a habit
    ref = rec["reference"]
    recorded = [c for c in journey["stored"]["candidates"] if c["origin"] == "recorded"]
    if rec["rule"].get("score"):  # 0.2.1 (F7): with a score target, the most likely one, ties by run cost
        best = min(recorded, key=lambda c: (-c["prediction"]["p_success"]["mean"],
                                            c["prediction"]["cost"]["usd"]["mean"]))
    else:
        best = min(recorded,
                   key=lambda c: c["prediction"]["cost"]["usd"]["mean"] / c["prediction"]["p_success"]["mean"])
    assert ref["kind"] == "best_recorded" and ref["from"] == "recorded"
    assert ref["config"]["id"] == best["config"]["id"]
    assert ref["text"] == f"no usual workflow; reference: your best recorded workflow ({ref['label']})"
    assert "your usual" not in json.dumps(rec).lower()
    assert "your usual" not in journey["rec_text"].lower()
    assert "No usual workflow; reference: your best recorded workflow (" in journey["rec_text"]


def test_fit_counts_the_runs_as_the_users_and_without_keeps_them(journey):
    assert journey["fit"]["fit"]["n_runs"]["user"] == 16
    assert journey["fit_without"]["fit"]["n_runs"]["user"] == 16
    assert journey["fit_without"]["fit"]["n_runs"]["prior"] < journey["fit"]["fit"]["n_runs"]["prior"]
    assert journey["fit_without"]["options"]["without"] == [SHIPPED_SOURCE]


def test_status_names_the_fit_options(journey):
    line = next(x for x in journey["status_without"].splitlines() if x.startswith("fit: "))
    assert journey["fit_without"]["fit"]["id"] in line
    assert f"--without {SHIPPED_SOURCE}" in line and "+ 16 yours" in line
    before = next(x for x in journey["status"].splitlines() if x.startswith("fit: "))
    assert "--without" not in before and "+ 16 yours" in before


def test_plan_is_hidden_and_still_works(journey):
    helptext = journey["help"]
    assert "recommend" in helptext and ",plan," not in helptext
    assert not any(line.split()[:1] == ["plan"] for line in helptext.splitlines())
    code, out, err = journey["plan"]
    assert code == 0, err
    assert isinstance(json.loads(out), dict)  # stdout stays one JSON object
    assert "loopmath plan is experimental and may change" in err
