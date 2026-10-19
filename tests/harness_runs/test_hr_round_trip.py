"""End to end: harness2 runs folder -> `adapt harness2` -> `run import --finish` -> `fit` and
`posterior --json` -> `compare.paired` on the imported base and adopted rollouts.

The fit runs only with LOOPMATH_SLOW=1. The compare steps skip by name until `loopmath.compare` is on the
branch.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2fixture as F  # noqa: E402

from loopmath import harness, harness_runs as HR  # noqa: E402
from loopmath.cli import main  # noqa: E402
from loopmath.store import Store  # noqa: E402

TASKS = ("finance/task1", "finance/task2", "legal/task3", "legal/task4")
SLOW = os.environ.get("LOOPMATH_SLOW") == "1"


@pytest.fixture
def imported(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    (tmp_path / "user-home").mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))
    built = F.build(tmp_path / "runs", tasks=TASKS, variants={"judge-failed"})
    out, home = tmp_path / "ocp", tmp_path / "store"
    capsys.readouterr()
    assert main(["adapt", "harness2", str(built.root), "--out", str(out), "--accept-at", "1.0"]) == 0
    capsys.readouterr()
    assert main(["run", "import", str(out), "--finish", "--no-fit", "--json", "--home", str(home)]) == 0
    got = json.loads(capsys.readouterr().out)
    store = Store(home)
    docs = [store.run_doc(rid) for rid in got["runs"]]
    return built, home, docs


def test_every_rollout_is_stored_with_cost_score_and_record(imported):
    built, _, docs = imported
    assert len(docs) == len(built.rollouts)
    by_run = {harness.run_record(d)["tool"]["run_id"]: d for d in docs}
    for r in built.rollouts:
        doc = by_run[r.run_id]
        [att] = doc["attempts"]
        assert att["cost"]["usd"] == pytest.approx(r.usd)
        sig = [s for s in doc["run"].get("signals") or [] if s["kind"] == "score"]
        if r.judged:
            assert sig and (sig[0]["value"] is None) == (r.score is None)
        else:
            assert not sig


def test_arms_pair_up_per_task(imported):
    _, _, docs = imported
    arms = HR.arms(docs)
    assert set(arms) >= {"base", "adopted", "candidate"}
    assert {d["run"]["task"]["id"] for d in arms["adopted"]} == set(TASKS)
    assert all(d["run"]["labels"]["arm"] == "base" for d in arms["base"])


@pytest.mark.skipif(not SLOW, reason="fits a model; set LOOPMATH_SLOW=1 (heavy.sh HEAVY_QUICK=1)")
def test_fit_and_posterior_read_the_imported_runs(imported, capsys):
    """56 rollouts: every one is a user run with a cost row; the 15 judged with a score (8 bases and 7
    adopted, one judge failed) are the only rows of the score and success heads."""
    built, home, docs = imported
    capsys.readouterr()
    assert main(["fit", "--home", str(home), "--json"]) == 0
    fit = json.loads(capsys.readouterr().out)
    assert len(built.rollouts) == 56
    assert fit["runs_by_source"]["user"] == 56
    assert fit["user_labels"] == {"benchmark": 56}
    heads = fit["heads"]
    assert heads["cost"]["rows_by_source"]["user"] == 56
    scored = [r for r in built.rollouts if r.score is not None]
    assert len(scored) == 15
    assert heads["score:judge"]["rows_by_source"] == {"user": 15} and heads["score:judge"]["runs"] == 15
    assert heads["success"]["rows_by_source"]["user"] == 15
    assert main(["posterior", "--type", "research", "--repo", "harness2-jb", "--home", str(home), "--json"]) == 0
    post = json.loads(capsys.readouterr().out)
    assert post["fit"]["n_runs"]["user"] == 56
    assert post["heads"]["score:judge"] == {"kind": "pp", "scale": "fraction", "better": "higher"}


def _interval(value: dict, name: str) -> None:
    assert set(value) >= {"mean", "median", "lo", "hi", "bands"}, name
    lo, hi = value["bands"]["80"]
    lo95, hi95 = value["bands"]["95"]
    assert (lo, hi) == (value["lo"], value["hi"]), name
    assert all(math.isfinite(x) for x in (lo, hi, lo95, hi95, value["median"])), name
    assert lo95 <= lo <= value["median"] <= hi <= hi95, name


def test_compare_paired_on_base_and_adopted(imported):
    """Four tasks, two bases and two adopted rollouts each (agg and seq), all in one experiment with no pair
    key: 8 pairs on the task, 7 with both runs scored (one adopted judge failed)."""
    compare = pytest.importorskip("loopmath.compare", reason="compare.paired is not on this branch yet")
    built, _, docs = imported
    arms = HR.arms(docs)
    assert len(arms["base"]) == len(arms["adopted"]) == 8
    result = compare.paired(arms["base"], arms["adopted"], rule="score:judge")
    got = result.to_dict()
    assert got["schema"] == "loopmath.compare/1"
    assert got["n_pairs"] == result.n_pairs == 7
    experiments = {d["run"]["ext"]["dev.loopmath.pair"]["experiment"] for d in docs}
    assert len(experiments) == 1   # one sweep, one experiment; keys are task:<experiment>/<task>
    exp = experiments.pop()
    assert len(got["pairs"]) == 8
    assert {p["key"] for p in got["pairs"]} == {f"task:{exp}/{t}" for t in TASKS}
    for name in ("gain", "cost_base", "cost_cand"):
        _interval(got[name], name)
    assert isinstance(got["p_better"], float) and 0.0 <= got["p_better"] <= 1.0
    usd = {r.usd for r in built.rollouts if r.arm in ("base", "adopted")}
    assert len(usd) == 1   # every base and adopted rollout cost the same here
    assert got["cost_base"]["lo"] <= usd.pop() * 1.5 and got["cost_base"]["hi"] > 0
    assert got["verdict"] in ("adopt", "reject", "more")


def test_the_pair_key_survives_the_round_trip(tmp_path, monkeypatch, capsys):
    """A row's `pair` reaches the stored run as dev.loopmath.pair, and compare pairs on it."""
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    (tmp_path / "user-home").mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))
    rows = tmp_path / "rows.jsonl"
    for j in range(3):
        for arm, score in (("base", 0.5 + 0.1 * j), ("candidate", 0.6 + 0.1 * j if j != 1 else None)):
            HR.append(rows, HR.row("ledger-1", arm=arm, pair=f"ledger-1#{j}", agent="codex", model="gpt-5.4",
                                   outcome={"score": score, "judge": "judge-x", "scale": "fraction"},
                                   cost={"usd": 0.1}, ended_at=f"2026-10-18T1{j}:00:00-07:00"))
    out, home = tmp_path / "ocp", tmp_path / "store"
    capsys.readouterr()
    assert main(["adapt", "harness-runs", str(rows), "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["run", "import", str(out), "--finish", "--no-fit", "--json", "--home", str(home)]) == 0
    got = json.loads(capsys.readouterr().out)
    docs = [Store(home).run_doc(rid) for rid in got["runs"]]
    pairs = sorted((d["run"]["ext"]["dev.loopmath.pair"]["pair"], d["run"]["ext"]["dev.loopmath.pair"]["arm"])
                   for d in docs)
    assert pairs == sorted((f"ledger-1#{j}", arm) for j in range(3) for arm in ("base", "candidate"))
    compare = pytest.importorskip("loopmath.compare", reason="compare.paired is not on this branch yet")
    arms = HR.arms(docs)
    from_docs = compare.paired(arms["base"], arms["candidate"], rule="score:judge")
    from_rows = compare.paired([r for r in HR.read(rows) if r["arm"] == "base"],
                               [r for r in HR.read(rows) if r["arm"] == "candidate"], rule="score:judge")
    assert from_docs.n_pairs == from_rows.n_pairs == 2
    keys = {p["key"] for p in from_docs.to_dict()["pairs"]}   # no experiment: the null namespace
    assert keys == {f"pair:ledger-1#{j}" for j in range(3)}
