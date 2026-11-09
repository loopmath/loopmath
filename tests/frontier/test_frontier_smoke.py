"""Ask/tell, store import, restart, budget and strategy extension behavior."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from loopmath.frontier import commands as C, record as R
from loopmath.frontier.state import check_resume
from loopmath.frontier.strategy import STRATEGIES, Pick, RandomTasks
from loopmath.harness_runs.rows import row
from loopmath.store.commands import UserError
from loopmath.store.home import Store


def setup_frontier(tmp_path, **kw):
    store = Store(tmp_path / "store")
    tasks = [{"id": f"task-{i}", "weight": i + 1} for i in range(5)]
    settings = [{"name": f"setting-{i}", "model": f"fake-{i}", "effort": "high"} for i in range(3)]
    return store, C.initialize(store, tasks, settings, kw.pop("budget", 1000), **kw)


def cli(store, *args):
    env = dict(os.environ, LOOPMATH_NO_UPDATE_CHECK="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", "loopmath", "frontier", *args, "--home", str(store.home), "--json"],
                            env=env, text=True, capture_output=True)
    assert result.stdout, result.stderr
    return result, json.loads(result.stdout)


def test_budget_restart_and_duplicate(tmp_path):
    store, rec = setup_frontier(tmp_path)
    frontier = rec["frontier"]
    # Leave a child alive after it writes next, then kill that exact child.
    # The next command must recover the trial entirely from the store.
    worker = """
import time
import json
from io import StringIO
from contextlib import redirect_stdout
from types import SimpleNamespace
from loopmath.frontier.commands import frontier_next
args = SimpleNamespace(home=HOME, frontier=FRONTIER, k=1, strategy=None, json=True)
text = StringIO()
with redirect_stdout(text):
    frontier_next(args)
print(json.dumps(json.loads(text.getvalue())), flush=True)
time.sleep(60)
""".replace("HOME", repr(str(store.home))).replace("FRONTIER", repr(frontier))
    process = subprocess.Popen([sys.executable, "-u", "-c", worker], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, env=dict(os.environ))
    try:
        first = json.loads(process.stdout.readline())
        assert first["trials"]
    finally:
        if process.poll() is None:
            print(f"restart test target pid {process.pid}, parent {os.getpid()}", flush=True)
            process.kill()
        process.wait(timeout=5)
    assert process.returncode < 0
    resumed, second = cli(store, "next", frontier)
    assert resumed.returncode == 0, resumed.stderr
    assert first == second
    trial = first["trials"][0]
    recorded, payload = cli(store, "record", trial["trial"], "--frontier", frontier, "--passed", "1", "--usd", "75")
    assert recorded.returncode == 0, recorded.stderr
    duplicate, again = cli(store, "record", trial["trial"], "--frontier", frontier, "--passed", "1", "--usd", "75")
    assert duplicate.returncode == 0, duplicate.stderr
    assert payload == again
    conflict, error = cli(store, "record", trial["trial"], "--frontier", frontier, "--passed", "0", "--usd", "75")
    assert conflict.returncode == 1
    assert "different content" in error["error"]
    assert len(list((store.home / "runs").glob("*.json"))) == 1
    for _ in range(20):
        try:
            rec, trials = C.ask(store, frontier)
        except UserError as exc:
            assert "report" in str(exc)
            break
        predicted = trials[0]["predicted_usd"]
        spend = sum(t.get("usd", 0) for t in rec["trials"])
        assert spend + predicted <= rec["budget_usd"]
        C.tell(store, frontier, trials[0]["trial"], passed=0, usd=75)
    else:
        pytest.fail("budget never stopped next")
    final = R.load(store, frontier)
    assert sum(t["state"] == "recorded" for t in final["trials"]) == len(list((store.home / "runs").glob("*.json")))
    view = C.report(store, frontier, draws=100)
    assert view["probability_label"] == C.PROBABILITY_LABEL
    assert len(view["settings"]) == 3


def test_strategy_registered_alone_drives_next(tmp_path, monkeypatch):
    class Fixed:
        name = "fixed-test"

        def pick(self, post, cost, open_, k, rng):
            return [Pick(2, 4, 1.0, {"test": True})]

    monkeypatch.setitem(STRATEGIES, "fixed-test", lambda cfg: Fixed())
    store, rec = setup_frontier(tmp_path, strategy="fixed-test")
    _, trials = C.ask(store, rec["frontier"])
    assert (trials[0]["setting"], trials[0]["task"], trials[0]["p_task"]) == (2, 4, 1.0)
    with pytest.raises(UserError, match="changed strategy"):
        C.ask(store, rec["frontier"], strategy="kg+random")


def test_snapshot_disposable_and_rebuild_identical(tmp_path):
    store, rec = setup_frontier(tmp_path)
    frontier = rec["frontier"]
    _, trials = C.ask(store, frontier)
    C.tell(store, frontier, trials[0]["trial"], passed=1, usd=2)
    rec = R.load(store, frontier)
    cached = C.rebuild(store, rec)
    rebuilt = C.rebuild(store, rec, cache=False)
    for field in ("m", "Sq", "mc", "Sc", "sig2", "nst"):
        np.testing.assert_array_equal(getattr(cached, field), getattr(rebuilt, field))
    # Issue once from a snapshot, then restore only the durable record and
    # corrupt the cache. The next setting and task are identical.
    _, first = C.ask(store, frontier)
    R.save(store, rec)
    C._snapshot_path(store, rec).write_bytes(b"invalid snapshot")
    _, second = C.ask(store, frontier)
    for key in ("setting", "task", "p_task", "predicted_usd"):
        assert first[0][key] == second[0][key]


def test_rows_import_and_conflict(tmp_path):
    store, rec = setup_frontier(tmp_path)
    _, trials = C.ask(store, rec["frontier"])
    trial = trials[0]
    result = row(rec["tasks"][trial["task"]]["id"], outcome={"accepted": True}, cost={"usd": 2.5})
    _, saved = C.tell(store, rec["frontier"], trial["trial"], input_row=result)
    assert saved["passed"] == 1 and saved["usd"] == 2.5
    C.tell(store, rec["frontier"], trial["trial"], input_row=result)
    with pytest.raises(UserError, match="different content"):
        C.tell(store, rec["frontier"], trial["trial"], input_row=dict(result, cost={"usd": 3}))


def test_declared_weights_and_resume_validation(tmp_path):
    store, rec = setup_frontier(tmp_path)
    assert np.isclose(sum(t["weight"] for t in rec["tasks"]), 1)
    chooser = RandomTasks(np.array([0, 1, 0]))
    rng = np.random.default_rng(0)
    assert chooser.choose(0, None, rng) == (1, 1.0)
    rec["settings"][0]["model"] = "changed"
    with pytest.raises(UserError, match="declarations changed"):
        check_resume(rec)


def test_parallel_picks_have_distinct_settings_and_reserve_budget(tmp_path):
    store, rec = setup_frontier(tmp_path)
    _, trials = C.ask(store, rec["frontier"], k=3)
    assert len(trials) == 3
    assert len({t["setting"] for t in trials}) == 3
    assert sum(t["predicted_usd"] for t in trials) <= rec["budget_usd"]


def test_cli_init_and_report(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    settings = tmp_path / "settings.toml"
    tasks.write_text('{"id":"task-one"}\n{"id":"task-two"}\n')
    settings.write_text('[[settings]]\nname="candidate"\nmodel="fake-model"\neffort="high"\n')
    store = Store(tmp_path / "cli-store")
    initialized, result = cli(store, "init", "--tasks", str(tasks), "--settings", str(settings), "--budget", "1000")
    assert initialized.returncode == 0, initialized.stderr
    viewed, report = cli(store, "report", result["frontier"])
    assert viewed.returncode == 0, viewed.stderr
    assert report["probability_label"] == C.PROBABILITY_LABEL
    assert report["settings"][0]["name"] == "candidate"
    result = subprocess.run([sys.executable, "-m", "loopmath", "frontier", "report", report["frontier"],
                             "--home", str(store.home)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert C.PROBABILITY_LABEL in result.stdout


def test_import_then_crash_retries_without_a_second_store_entry(tmp_path, monkeypatch):
    store, rec = setup_frontier(tmp_path)
    frontier = rec["frontier"]
    _, trials = C.ask(store, frontier)
    trial = trials[0]["trial"]
    save = R.save

    def interrupted(store, value):
        raise RuntimeError("interrupted after import")

    monkeypatch.setattr(R, "save", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        C.tell(store, frontier, trial, passed=1, usd=2)
    monkeypatch.setattr(R, "save", save)
    before = store.index_path.read_bytes()
    C.tell(store, frontier, trial, passed=1, usd=2)
    assert store.index_path.read_bytes() == before
    assert R.load(store, frontier)["trials"][0]["state"] == "recorded"


def test_concurrent_conflicting_records_preserve_winning_store_result(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from loopmath.frontier import FRONTIER_EXT
    store, rec = setup_frontier(tmp_path)
    frontier = rec["frontier"]
    _, trials = C.ask(store, frontier)
    trial = trials[0]["trial"]

    def record(passed):
        try:
            C.tell(store, frontier, trial, passed=passed, usd=2)
            return passed
        except UserError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(record, [0, 1]))
    assert sum(value is not None for value in results) == 1
    saved = R.load(store, frontier)["trials"][0]
    assert saved["passed"] in results
    ext = store.run_doc(saved["run"])["run"]["ext"][FRONTIER_EXT]
    assert ext["digest"] == saved["digest"]


def test_live_posterior_accepts_a_cost_with_only_protocol_methods(tmp_path):
    from loopmath.frontier.model import TaskPosterior

    class KnownCost:
        __slots__ = ("S",)

        def __init__(self, S):
            self.S = S

        def add(self, s, usd):
            pass

        def fit(self):
            pass

        def refit(self):
            return None

        def per_run(self):
            return np.full(self.S, 2.0)

        def sample_mean(self, K, rng):
            return np.full((K, self.S), 2.0)

    store, rec = setup_frontier(tmp_path)
    weights = np.tile([t["weight"] for t in rec["tasks"]], (3, 1))
    post = TaskPosterior(C._prior(rec), weights, cost=KnownCost(3))
    for _ in range(10):
        post.add(0, 1, 2.0, 0)
        post.fit()
    E, M = post.sample_latent(100, np.random.default_rng(0))
    q, c = post.to_qc(E, M)
    np.testing.assert_allclose(c, 2)
    assert post.pool_N is None and post.cell_N is None
    assert q[:, 0].std() > 0.01
    np.testing.assert_array_equal(post.W, weights)
