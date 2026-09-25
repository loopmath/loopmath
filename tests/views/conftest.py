"""Shared pieces for the view tests (lane 12): fixtures, a v0.3 run document builder, the page probe."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path
from signal import SIGKILL  # not `import signal`: signal() below builds signal documents

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v0_1"
PROBE = Path(__file__).resolve().parent / "view_probe.mjs"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


@pytest.fixture
def fixture_json():
    def load(name: str) -> dict:
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return load


WORKFLOW = {
    "id": "implement_review", "version": 1, "title": "Implement, review",
    "pieces": [{"id": "implement", "role": "implement"}, {"id": "review", "role": "review"}],
    "artifacts": [{"id": "issue", "kind": "issue"}, {"id": "diff", "kind": "diff"}, {"id": "verdict", "kind": "verdict"}],
    "edges": [["issue", "implement"], ["implement", "diff"], ["diff", "review"], ["review", "verdict"]],
    "control": {"gates": ["review"], "repair": {"review": "implement"}, "budget": 3},
}
SOLO = {"id": "solo", "version": 1, "pieces": [{"id": "implement", "role": "implement"}],
        "artifacts": [{"id": "issue", "kind": "issue"}, {"id": "diff", "kind": "diff"}],
        "edges": [["issue", "implement"], ["implement", "diff"]], "control": {"gates": [], "repair": {}, "budget": 0}}


def setting(model: str, effort: str, harness: str = "codex") -> dict:
    return {"harness": harness, "model": {"raw": model, "id": model}, "effort": effort, "context_policy": "fresh", "options": {}}


def default_settings(workflow: dict) -> dict:
    return {p["id"]: setting("gpt-6-astra", "xhigh") for p in workflow["pieces"]}


def config_id(workflow: dict, settings: dict) -> str:
    """The canonical configuration id (lane 1) of an OCP v0.3 workflow and settings."""
    from loopmath.ocp.canonical import config_id as canonical_config_id

    return canonical_config_id(workflow, settings)


def attempt(aid: str, vertex: str, rnd: int, start: str, end: str, usd: float, *, result: str = "done",
            model: str = "gpt-6-astra", effort: str = "xhigh") -> dict:
    return {"id": aid, "node": vertex, "n": rnd, "harness": "codex", "model": {"raw": model, "id": model},
            "effort": effort, "status": "done", "started_at": start, "ended_at": end, "vertex": vertex, "round": rnd,
            "cost": {"input_tokens": 1000, "cached_input_tokens": 20000, "cache_creation_tokens": 0,
                     "output_tokens": 500, "usd": usd}, "outcome": {"result": result}}


def run_doc(run_id: str, *, started: str = "2026-09-20T10:00:00-07:00", ended: str | None = "2026-09-20T11:00:00-07:00",
            task_type: str = "feature", repo: str = "acme/api", title: str = "Add a thing", workflow: dict | None = None,
            source: str | None = "usual", provenance: dict | None = None, attempts: list | None = None,
            signals: list | None = None, rule: dict | None = None, slate: dict | None = None,
            preferences: list | None = None, artifacts: list | None = None, receipt: dict | None = None) -> dict:
    wf = copy.deepcopy(workflow or WORKFLOW)
    settings = default_settings(wf)
    run = {"id": run_id, "title": title, "started_at": started,
           "task": {"id": f"task-{run_id}", "title": title, "type": task_type, "repo": repo},
           "configuration": {"id": config_id(wf, settings), "workflow": wf, "settings": settings},
           "signals": signals or []}
    if ended:
        run["ended_at"] = ended
    if source:
        run["configuration"]["source"] = source
    if provenance:
        run["provenance"] = provenance
    if rule:
        run["acceptance_rule"] = rule
    if slate:
        run["slate"] = slate
    if preferences:
        run["preferences"] = preferences
    if receipt:
        run["receipt"] = receipt
    nodes = [{"id": p["id"], "kind": "impl", "title": p["id"], "state": "done", "vertex": p["id"]} for p in wf["pieces"]]
    doc = {"ocp": "0.3", "producer": {"name": "lane-12-test", "version": "0"}, "privacy": {"profile": "metadata_only"},
           "run": run, "nodes": nodes, "edges": [],
           "attempts": attempts if attempts is not None else [
               attempt(f"{run_id}.a1", wf["pieces"][0]["id"], 1, started, ended or started, 1.25)]}
    if artifacts is not None:
        doc["artifacts"] = artifacts
    return doc


def signal(sid: str, kind: str, name: str, value, at: str, tier: str = "verified", **extra) -> dict:
    return dict({"id": sid, "kind": kind, "name": name, "value": value, "observed_at": at, "tier": tier}, **extra)


def write_store(root: Path, docs: list[dict], *, receipts: list[dict] = (), appended: dict | None = None) -> Path:
    (root / "runs").mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (root / "runs" / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    if receipts:
        (root / "receipts").mkdir(exist_ok=True)
        for rct in receipts:
            (root / "receipts" / f"{rct['id']}.json").write_text(json.dumps(rct), encoding="utf-8")
    for run_id, sigs in (appended or {}).items():
        (root / "signals").mkdir(exist_ok=True)
        (root / "signals" / f"{run_id}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sigs), encoding="utf-8")
    return root


@pytest.fixture
def v03():
    """The builders above, for tests that write their own store."""
    class Builders:
        run_doc = staticmethod(run_doc)
        attempt = staticmethod(attempt)
        signal = staticmethod(signal)
        write_store = staticmethod(write_store)
        config_id = staticmethod(config_id)
        default_settings = staticmethod(default_settings)
        WORKFLOW = WORKFLOW
        SOLO = SOLO
    return Builders


@pytest.fixture
def probe(tmp_path):
    """Load a page in headless Chrome (no network) and run a script in it; skips without node or Chrome."""
    node = shutil.which("node")
    if not node or not CHROME.exists():
        pytest.skip("node or Google Chrome not found; the in-page check did not run")

    def run(page: Path, script: str | None = None, timeout: float = 60) -> dict:
        args = [node, str(PROBE), str(page)]
        if script is not None:
            script_path = tmp_path / f"probe-{page.stem}.js"
            script_path.write_text(script, encoding="utf-8")
            args.append(str(script_path))
        # Own process group, so a timeout kills Chrome with node rather than orphaning it.
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              start_new_session=True) as proc:
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, SIGKILL)
                proc.communicate()
                raise
        assert proc.returncode == 0, err
        return json.loads(out.strip().splitlines()[-1])
    return run
