"""Strategy adaptive-blocks (the default for new frontiers) and the resume of kg+random frontiers."""

import copy
import json
import pathlib
import shutil

import numpy as np
import pytest
from scipy.stats import norm

from loopmath.frontier import blocks, commands as C, record as R, value
from loopmath.frontier.belief import AdaptiveBelief
from loopmath.frontier.misfit import Monitor
from loopmath.store.commands import UserError
from loopmath.store.home import Store

DATA = pathlib.Path(__file__).parent / "data" / "kg-random-0-4-5"
SETTINGS = [dict(name="opus-high", model="claude-opus-4-7", effort="high"),
            dict(name="haiku-low", model="claude-haiku-5-5", effort="low"),
            dict(name="fake-high", model="fake-1", effort="high")]


def make(tmp_path, n_tasks=12, budget=300.0, **kw):
    store = Store(tmp_path / "store")
    tasks = [{"id": f"task-{i}"} for i in range(n_tasks)]
    kw.setdefault("seed", 5)
    return store, C.initialize(store, tasks, [dict(s) for s in SETTINGS], budget, **kw)


def outcome(rec, t):
    """A fixed outcome per (setting, task, earlier runs of the pair), so that the order of recording is free."""
    s, u = t["setting"], t["task"]
    k = sum(x["state"] == "recorded" and x["setting"] == s and x["task"] == u for x in rec["trials"])
    return int((3 * s + u + k) % 4 != 0), 0.5 + s + 0.05 * u


def tell(store, frontier, trial, rec=None):
    rec = rec or R.load(store, frontier)
    passed, usd = outcome(rec, trial)
    C.tell(store, frontier, trial["trial"], passed=passed, usd=usd)


def run_blocks(store, frontier, n):
    out = []
    for _ in range(n):
        _, trials = C.ask(store, frontier)
        out.append(trials)
        for t in trials:
            tell(store, frontier, t)
    return out


def decision(trials):
    return [(t["setting"], t["task"], t["p_task"], t["explore"], t["support"], t["pos"], t["why"]) for t in trials]


# ---- the default and its settings

def test_new_frontiers_default_to_adaptive_blocks(tmp_path):
    store, rec = make(tmp_path)
    assert rec["strategy"] == C.DEFAULT_STRATEGY == "adaptive-blocks"
    assert rec["cost_model"] == "frontier-separate"
    assert rec["config"]["policy"] == dict(tasks="chosen", pair_cap=4, q0=0.2, explore=0.1, refit_every=20)
    assert rec["config"]["run_budgets"] == np.round(np.geomspace(0.0145, 26.5, 36), 6).tolist()


@pytest.mark.parametrize("kw, message", [
    (dict(policy={"pair_cap": 0}), "pair_cap"),
    (dict(policy={"tasks": "random"}), "tasks"),
    (dict(policy={"kappa": 1.0}), "unknown"),
    (dict(policy={"q0": 0.6}), "q0"),
    (dict(run_budgets=[1.0, 0.5]), "increasing"),
    (dict(cost_model="frontier"), "takes cost model"),
    (dict(strategy="kg+random", policy={"pair_cap": 2}), "apply to"),
    (dict(strategy="kg+random", cost_model="frontier-separate"), "takes cost model"),
])
def test_init_refuses_bad_settings(tmp_path, kw, message):
    with pytest.raises(UserError, match=message):
        make(tmp_path, **kw)


# ---- blocks of 1, 8 and 64 runs

def test_candidates_are_single_runs_and_blocks_of_8_and_64(tmp_path):
    store, rec = make(tmp_path, n_tasks=70, budget=5000.0)
    belief = C.rebuild(store, rec)
    policy = C._block_policy(rec)
    remaining = policy.open_runs(belief, np.full((3, 70), blocks.UNLIMITED))
    c_elig, cost = policy.prices(belief)
    cand = [np.flatnonzero(remaining[s] > 0) for s in range(3)]
    gains, own, mu, var = policy._pair_table(belief, cand, c_elig)
    assert np.isfinite(gains).all()                      # every open pair is a single-run candidate
    table = policy._block_table(belief, remaining, np.zeros(3, bool), own, cost, c_elig, mu, var)
    assert sorted((s, len(tasks)) for s, tasks, _, _ in table) == [(s, L) for s in range(3) for L in (8, 64)]
    lengths = {len(trials) for trials in run_blocks(store, rec["frontier"], 14)}
    assert 1 in lengths and lengths & {8, 64}            # both kinds are bought on this frontier


# ---- no penalty in the value that chooses runs

def test_the_choosing_value_has_no_penalty(tmp_path):
    store, rec = make(tmp_path)
    assert C._block_policy(rec).kappa == 0.0
    g = np.random.default_rng(3)
    S, N = 5, 7
    mu, var = g.uniform(0.2, 0.8, S), g.uniform(0.001, 0.02, S)
    shifts, var_after = g.normal(0, 0.05, (N, S)), var * g.uniform(0.5, 1.0, (N, S))
    c_hat, B = g.uniform(0.1, 3.0, S), np.geomspace(0.05, 5.0, 12)
    w = np.ones(12) / 12
    fast = value.gain_pairs(mu, var, shifts, var_after, c_hat, B, w)
    slow = [value.gain_exact_slow(mu, var, shifts[n], var_after[n], c_hat, B, w) for n in range(N)]
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-15)
    penalized = value.gain_pairs(mu, var, shifts, var_after, c_hat, B, w, kappa=1.0)
    assert not np.allclose(fast, penalized)


# ---- an optimistic price

def test_eligibility_uses_the_20th_percentile_and_cost_the_median(tmp_path):
    store, rec = make(tmp_path)
    run_blocks(store, rec["frontier"], 3)
    rec = R.load(store, rec["frontier"])
    belief = C.rebuild(store, rec)
    c_elig, cost = C._block_policy(rec).prices(belief)
    fits = [belief.p_fit([c])[0, s] for s, c in enumerate(c_elig)]
    np.testing.assert_allclose(fits, 0.2, rtol=1e-9)     # P(mean cost per run <= c_elig) = q0
    h = belief.cost
    median = np.exp(h.mc + h.sig2 / 2)                   # the median of the belief about the mean cost per run
    np.testing.assert_allclose(cost[:, 0], median, rtol=1e-12)
    assert (c_elig < median).all() and (median < belief.c()).all()


# ---- the pair cap

def test_pair_cap_holds_and_ends_the_frontier(tmp_path):
    store, rec = make(tmp_path, n_tasks=3, budget=1000.0, policy={"pair_cap": 2})
    frontier = rec["frontier"]
    with pytest.raises(UserError, match="no next trial"):
        for _ in range(40):
            _, trials = C.ask(store, frontier)
            for t in trials:
                tell(store, frontier, t)
    rec = R.load(store, frontier)
    counts = np.zeros((3, 3), int)
    for t in rec["trials"]:
        counts[t["setting"], t["task"]] += 1
    assert (counts == 2).all()                           # every pair to the cap and no further


def test_study_mode_respects_the_pool_and_the_cell_cap(tmp_path):
    cells = [[1, 0, 5], [2, 2, 2], [0, 0, 1]]
    store, rec = make(tmp_path, n_tasks=3, budget=1000.0, mode="study", config=dict(cell_n=cells, cell_cap=4))
    frontier = rec["frontier"]
    with pytest.raises(UserError, match="no next trial"):
        for _ in range(40):
            _, trials = C.ask(store, frontier)
            for t in trials:
                tell(store, frontier, t)
    counts = np.zeros((3, 3), int)
    for t in R.load(store, frontier)["trials"]:
        counts[t["setting"], t["task"]] += 1
    np.testing.assert_array_equal(counts, np.minimum(cells, 4))


# ---- the task rule as a parameter

def test_balanced_task_rule_keeps_tasks_level_and_probes_every_run(tmp_path):
    store, rec = make(tmp_path, n_tasks=10, budget=200.0, policy={"tasks": "balanced"})
    frontier = rec["frontier"]
    for trials in run_blocks(store, frontier, 8):
        assert all(t["explore"] and t["why"]["rule"] == "balanced" for t in trials)
        assert all(t["p_task"] == pytest.approx(1.0 / t["support"]) for t in trials)
        counts = np.zeros((3, 10), int)
        for t in R.load(store, frontier)["trials"]:
            counts[t["setting"], t["task"]] += 1
        assert (counts.max(1) - counts.min(1) <= 1).all()
    rec = R.load(store, frontier)
    belief = C.rebuild(store, rec)
    assert belief.misfit()["n"] == len(rec["trials"])    # the monitor scored every run


# ---- next, record, report

def test_next_hands_out_one_block_and_record_takes_part_of_it(tmp_path):
    store, rec = make(tmp_path, n_tasks=40, budget=2000.0)
    frontier = rec["frontier"]
    with pytest.raises(UserError, match="one block"):
        C.ask(store, frontier, k=2)
    blocks_ = run_blocks(store, frontier, 12)
    _, trials = C.ask(store, frontier)
    while len(trials) < 3:
        tell(store, frontier, trials[0])
        _, trials = C.ask(store, frontier)
    first = [t["trial"] for t in trials]
    tell(store, frontier, trials[1])
    view = C.report(store, frontier)                     # a report in the middle of a block
    assert view["pending"] == len(first) - 1
    _, again = C.ask(store, frontier)
    assert [t["trial"] for t in again] == first[:1] + first[2:]
    assert len({t["block"] for t in again}) == 1 and blocks_


def test_order_of_recording_and_reports_does_not_change_the_next_block(tmp_path):
    a_store, a = make(tmp_path / "a", n_tasks=30, budget=2000.0)
    b_store, b = make(tmp_path / "b", n_tasks=30, budget=2000.0)
    for _ in range(10):
        _, ta = C.ask(a_store, a["frontier"])
        _, tb = C.ask(b_store, b["frontier"])
        assert decision(ta) == decision(tb)
        rec = R.load(a_store, a["frontier"])
        for t in ta:
            tell(a_store, a["frontier"], t, rec)
            C.report(a_store, a["frontier"], draws=200)
        rec = R.load(b_store, b["frontier"])
        for t in reversed(tb):
            tell(b_store, b["frontier"], t, rec)


def test_report_keeps_every_key_and_adds_the_budget_view(tmp_path):
    store, rec = make(tmp_path)
    run_blocks(store, rec["frontier"], 4)
    view = C.report(store, rec["frontier"])
    old = {"frontier", "probability_label", "p_match", "settings", "budget_usd", "spend_usd", "pending", "refits"}
    assert old <= set(view) and {"strategy", "budgets", "misfit"} <= set(view)
    assert {"name", "on_frontier", "p_frontier", "pass_rate", "pass_interval", "cost_usd", "cost_interval",
            "runs"} <= set(view["settings"][0])
    assert len(view["budgets"]) == 36
    rows = [r for r in view["budgets"] if r["pick"] is not None]
    assert rows and all(0.0 <= r["p_fit"] <= 1.0 and r["cost_usd"] <= r["budget_usd"] for r in rows)
    lines = C.report_lines(view)
    assert lines[1].endswith(C.PROBABILITY_LABEL) and any(line.startswith("USD/run budget") for line in lines)
    assert lines[-1].startswith("misfit test:")
    json.dumps(view)
    belief = C.rebuild(store, R.load(store, rec["frontier"]))
    assert belief.last_full <= belief.n_runs             # the report refits a copy, never the saved state


# ---- resume

def _restore_fixture(tmp_path):
    expected = json.loads((DATA / "expected.json").read_text())
    store = Store(tmp_path / "store")
    store.ensure()
    path = R.frontier_path(store, expected["frontier"])
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "frontier.json", path)
    return store, expected


def test_kg_random_frontier_from_0_4_5_resumes_bit_for_bit(tmp_path):
    store, expected = _restore_fixture(tmp_path)
    frontier = expected["frontier"]
    assert expected["version"] == "0.4.5"
    i = 0
    for step in range(4):
        _, trials = C.ask(store, frontier, k=3 if step == 2 else 1)
        for t in trials:
            want = expected["after"][i]
            got = dict(setting=t["setting"], task=t["task"], p_task=t["p_task"], predicted_usd=t["predicted_usd"],
                       why=t["why"])
            assert got == {k: want[k] for k in got}
            C.tell(store, frontier, t["trial"], passed=want["passed"], usd=want["usd"])
            i += 1
    assert i == len(expected["after"])
    view = json.loads(json.dumps(C.report(store, frontier)))
    for key, want in expected["report"].items():
        assert view[key] == want, key
    assert view["strategy"] == "kg+random" and view["note"] == C.NEWER_STRATEGY_NOTE
    lines = C.report_lines(view)
    assert lines[-1] == f"note: {C.NEWER_STRATEGY_NOTE}"


def test_snapshots_from_other_package_versions_are_rebuilt(tmp_path, monkeypatch):
    store, expected = _restore_fixture(tmp_path)
    rec = R.load(store, expected["frontier"])
    cached = C.rebuild(store, rec)
    applied = []
    apply = C._apply
    monkeypatch.setattr(C, "_apply", lambda *a: applied.append(1) or apply(*a))
    C.rebuild(store, rec)
    assert not applied                                   # same version: the snapshot covers every trial
    monkeypatch.setattr(C, "__version__", "0.0.0")
    rebuilt = C.rebuild(store, rec)
    assert len(applied) == sum(t["state"] == "recorded" for t in rec["trials"])
    for field in ("m", "Sq", "mc", "Sc", "sig2", "nst"):
        np.testing.assert_array_equal(getattr(cached, field), getattr(rebuilt, field))


def _state(belief):
    return dict(z=belief.post.z, Cz=belief.post.Cz, a=belief.post.a, mc=belief.cost.mc, Sc=belief.cost.Sc,
                sig2=belief.cost.sig2, logm=belief.monitor.logm, mem=belief.monitor.mem, n=belief.n,
                q=belief.q(), c=belief.c())


def test_adaptive_blocks_resume_from_snapshot_scratch_or_corrupt_cache(tmp_path):
    store, rec = make(tmp_path, n_tasks=30, budget=2000.0)
    frontier = rec["frontier"]
    run_blocks(store, frontier, 9)
    rec = R.load(store, frontier)
    cached, scratch = C.rebuild(store, rec), C.rebuild(store, rec, cache=False)
    assert cached.refits == scratch.refits and cached.n_runs == scratch.n_runs
    for k, v in _state(cached).items():
        np.testing.assert_array_equal(v, _state(scratch)[k], err_msg=k)
    saved = copy.deepcopy(rec)
    _, first = C.ask(store, frontier)
    R.save(store, saved)
    C._snapshot_path(store, saved).write_bytes(b"invalid snapshot")
    _, second = C.ask(store, frontier)
    assert decision(first) == decision(second)
    R.save(store, saved)
    done = [t for g in C._blocks(saved)[0] for t in g]
    assert C._load_block_snapshot(store, saved, done)[0] == len(done)
    with pytest.MonkeyPatch.context() as mp:            # a snapshot from another model version is not read
        mp.setitem(C.MODEL_VERSIONS, "adaptive-blocks", "other/0")
        assert C._load_block_snapshot(store, saved, done) == (0, None)
        _, third = C.ask(store, frontier)
    assert decision(first) == decision(third)


def test_a_frontier_never_changes_strategy(tmp_path):
    store, rec = make(tmp_path)
    with pytest.raises(UserError, match="changed strategy"):
        C.ask(store, rec["frontier"], strategy="kg+random")
    old_store, old = make(tmp_path / "old", strategy="kg+random")
    with pytest.raises(UserError, match="changed strategy"):
        C.ask(old_store, old["frontier"], strategy="adaptive-blocks")


# ---- the misfit monitor

def test_misfit_monitor_flags_a_setting_the_model_misses_and_only_at_epoch_ends():
    mon = Monitor(["m-a", "m-b"], 6)
    g = np.random.default_rng(0)
    for _ in range(30):
        for _ in range(8):
            t = int(g.integers(6))
            mon.probe(0, t, int(g.random() < 0.9), 0.4)  # setting 0 passes far more often than predicted
            mon.probe(1, t, int(g.random() < 0.4), 0.4)
        before = mon.misfit()
        mon.tick()
    out = mon.misfit()
    assert out["flag"] and out["flag_s"] == [True, False] and out["s"] == 0 and out["p"] < 0.05
    assert before["n_open"] == 16 and out["n_open"] == 0 and out["epochs"] == 30
    with pytest.raises(ValueError):
        mon.probe(0, 0, 1, 1.5)


def test_adaptive_belief_refits_on_the_runner_schedule(tmp_path):
    store, rec = make(tmp_path)
    belief = C._new_belief(rec)
    assert isinstance(belief, AdaptiveBelief)
    for i in range(19):
        belief.add(i % 3, i % 12, i % 2, 1.0)
    belief.block_end()
    assert belief.last_full == 0 and not belief.refits   # fewer than 20 runs since the last refit
    belief.add(0, 0, 1, 1.0)
    belief.block_end()
    assert belief.last_full == 20 and len(belief.refits) == 1


def test_cli_init_options_and_text_report(tmp_path):
    from tests.frontier.test_frontier_smoke import cli

    tasks, settings = tmp_path / "tasks.jsonl", tmp_path / "settings.toml"
    tasks.write_text("".join(f'{{"id":"task-{i}"}}\n' for i in range(6)))
    settings.write_text('[[settings]]\nname="a"\nmodel="fake-model"\neffort="high"\n'
                        '[[settings]]\nname="b"\nmodel="claude-haiku-5-5"\neffort="low"\n')
    store = Store(tmp_path / "cli-store")
    done, result = cli(store, "init", "--tasks", str(tasks), "--settings", str(settings), "--budget", "100",
                       "--task-rule", "balanced", "--pair-cap", "2", "--run-budgets", "0.1:10:5")
    assert done.returncode == 0, done.stderr
    rec = R.load(store, result["frontier"])
    assert rec["config"]["policy"]["tasks"] == "balanced" and rec["config"]["policy"]["pair_cap"] == 2
    assert rec["config"]["run_budgets"] == blocks.budget_grid(0.1, 10, 5)
    bad, error = cli(store, "init", "--tasks", str(tasks), "--settings", str(settings), "--budget", "100",
                     "--strategy", "kg+random", "--pair-cap", "2")
    assert bad.returncode != 0 and "apply to" in json.dumps(error)
