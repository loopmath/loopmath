"""`loopmath check-change`: start a check, record its pairs with `run record`, read the verdict.

The check lives in the store (`checks/<chk>.json`). `start` prints the next pair to run: its task, the order
of the two arms, the goal and trial worktrees, the `run record` lines and the blinded referee step. `run
record --check CHK --arm base|cand` tags the run (`run.ext["dev.loopmath.check"]`), gives it the pair's task
and opens or joins the pair's slate. `status` prints the `Comparison` fields of `compare.paired`.
`from-runs` compares runs already in the store, chosen by label. Synthetic stores, faked log settling.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from loopmath import cli, compare
from loopmath.store import finish as finish_mod
from loopmath.store.home import Store

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "store"))
from store_helpers import fake_settle  # noqa: E402

SOLO = ["--workflow", "solo", "--set", "implement=claude-code:claude-opus-5:high", "--source", "designed"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("LOOPMATH_HOME", str(h))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "logs"))
    (tmp_path / "wt").mkdir()
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle(usd=0.5))
    h.mkdir()
    return h


def lm(capsys, *argv) -> tuple[int, dict | str, str]:
    capsys.readouterr()
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, (json.loads(out) if "--json" in argv and out.strip() else out), err


def tasks_file(tmp_path: Path, n: int) -> Path:
    p = tmp_path / "tasks.jsonl"
    p.write_text("".join(json.dumps({"type": "bug_fix", "repo": "acme/app", "title": f"fix {i}",
                                     "base_commit": f"{i:07d}"}) + "\n" for i in range(n)))
    return p


def record(capsys, home, chk, arm, verdict, *extra):
    code, out, err = lm(capsys, "run", "record", "--check", chk, "--arm", arm, *SOLO,
                        "--cwd", f"implement={home.parent / 'wt'}", "--since", "1h", "--verified", f"tests={verdict}",
                        "--no-commits", "--no-fit", *extra, "--json")
    assert code == 0, err
    return out


def test_the_command_is_registered_with_its_verbs(capsys):
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:  # argparse prints the help and exits
        cli.main(["check-change", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for verb in ("start", "status", "from-runs"):
        assert verb in out


def test_start_creates_a_check_and_prints_the_next_pair(home, tmp_path, capsys):
    code, out, err = lm(capsys, "check-change", "start", "--base", "main", "--cand", "short-claude-md",
                        "--tasks", str(tasks_file(tmp_path, 6)), "--json")
    assert code == 0, err
    assert out["schema"] == "loopmath.check-change/1"
    chk = out["check"]
    assert chk.startswith("chk_") and (home / "checks" / f"{chk}.json").is_file()
    assert out["base"]["label"] == "main" and out["cand"]["label"] == "short-claude-md"
    nxt = out["next"]
    assert nxt["pair"] == "p01" and sorted(nxt["order"]) == ["base", "cand"]
    assert nxt["task"]["title"] == "fix 0" and nxt["task"]["base_commit"] == "0000000"
    assert set(nxt["worktrees"]) == {"base", "cand"}
    joined = "\n".join(nxt["commands"])
    assert f"--check {chk} --arm base" in joined and f"--check {chk} --arm cand" in joined
    assert "--blinded" in joined  # the referee step


def test_run_record_tags_the_run_and_the_pair_shares_a_task_and_a_slate(home, tmp_path, capsys):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                   str(tasks_file(tmp_path, 3)), "--json")
    chk = out["check"]
    a = record(capsys, home, chk, "base", "pass")
    b = record(capsys, home, chk, "cand", "fail")
    st = Store(home)
    da, db = st.run_doc(a["run"]), st.run_doc(b["run"])
    assert da["run"]["ext"][compare.CHECK_EXT] == {"check": chk, "arm": "base", "pair": "p01"}
    assert db["run"]["ext"][compare.CHECK_EXT] == {"check": chk, "arm": "cand", "pair": "p01"}
    assert da["run"]["task"]["id"] == db["run"]["task"]["id"]
    assert da["run"]["slate"]["id"] == db["run"]["slate"]["id"]
    assert sorted(da["run"]["slate"]["members"]) == sorted([a["run"], b["run"]])


def test_status_gives_the_comparison_and_the_next_pair(home, tmp_path, capsys):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                   str(tasks_file(tmp_path, 4)), "--budget-usd", "20", "--json")
    chk = out["check"]
    for verdicts in (("pass", "pass"), ("fail", "pass")):
        record(capsys, home, chk, "base", verdicts[0])
        record(capsys, home, chk, "cand", verdicts[1])
    code, st, err = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 0, err
    assert st["schema"] == "loopmath.check-change/1" and st["check"] == chk
    cmp = st["comparison"]
    for k in ("n_pairs", "gain", "p_better", "cost_base", "cost_cand", "cpar_base", "cpar_cand", "verdict",
              "next_pair_value_usd", "reasons"):
        assert k in cmp
    assert cmp["n_pairs"] == 2 and cmp["budget_usd"] == 20.0
    assert cmp["spent_usd"] == pytest.approx(4 * 0.5)
    if cmp["verdict"] == "more":
        assert st["next"]["pair"] == "p03"
    code, text, _ = lm(capsys, "check-change", "status", chk)
    assert code == 0 and "P(better)" in text and len(text.splitlines()) <= 25


def test_status_text_lists_the_next_pairs_steps(home, tmp_path, capsys):
    """The tester's #350 and #352: status said `check-change start lists the steps`, and start lists p01's only."""
    _, out, _ = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                   str(tasks_file(tmp_path, 4)), "--budget-usd", "20", "--json")
    chk = out["check"]
    record(capsys, home, chk, "base", "fail")
    record(capsys, home, chk, "cand", "pass")
    code, st, err = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 0 and st["comparison"]["verdict"] == "more" and st["next"]["pair"] == "p02", err
    code, text, err = lm(capsys, "check-change", "status", chk)
    lines = text.splitlines()
    at = lines.index(next(x for x in lines if x.startswith("next: pair p02, run ")))
    assert lines[at + 1:at + 1 + len(st["next"]["commands"])] == [f"  {c}" for c in st["next"]["commands"]], text
    assert f"  referee: {st['next']['referee']}" in lines and "start lists the steps" not in text
    assert sum(f"loopmath run record --check {chk} --arm" in x for x in lines) == 2


def test_a_run_out_of_the_planned_order_is_kept_with_a_warning_and_status_marks_its_pair(home, tmp_path, capsys):
    """The tester's #352 and #353: p03 was planned cand then base, and base was recorded first in silence."""
    from loopmath.compare import checks

    _, out, _ = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                   str(tasks_file(tmp_path, 4)), "--budget-usd", "20", "--json")
    chk = out["check"]
    store = Store(home)
    check = checks.load(store, chk)
    check["pairs"][0]["order"], check["pairs"][1]["order"] = ["base", "cand"], ["cand", "base"]
    checks.save(store, check)

    def rec(arm, pair):
        return lm(capsys, "run", "record", "--check", chk, "--arm", arm, "--pair", pair, *SOLO, "--cwd",
                  f"implement={home.parent / 'wt'}", "--since", "1h", "--verified", "tests=pass", "--no-commits",
                  "--no-fit", "--json")

    code, run, err = rec("base", "p01")  # as planned
    assert code == 0 and "warning" not in err, err
    code, run, err = rec("base", "p02")  # planned second
    assert code == 0 and run["run"], err
    assert (f"warning: pair p02 of {chk} is planned cand then base, which balances which arm runs first; this base "
            f"run comes first, out of that order. It is recorded and counts, and status marks the pair") in err, err
    pair = checks.load(store, chk)["pairs"][1]
    assert pair["runs"]["base"] == run["run"] and pair["ran"] == ["base"]
    code, st, err = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 0 and st["recorded"]["base"] == 2, err
    assert st["out_of_order"] == [{"pair": "p02", "planned": ["cand", "base"], "ran": ["base"]}]
    code, text, _ = lm(capsys, "check-change", "status", chk)
    assert "run out of the planned order: p02 (planned cand then base, ran base first)" in text.splitlines(), text
    code, _, err = rec("cand", "p02")
    assert code == 0 and "warning" not in err, err
    assert checks.load(store, chk)["pairs"][1]["ran"] == ["base", "cand"]
    code, text, _ = lm(capsys, "check-change", "status", chk)
    assert "run out of the planned order: p02 (planned cand then base, ran base then cand)" in text.splitlines()
    code, st, _ = lm(capsys, "check-change", "status", chk, "--json")
    assert st["recorded"] == {"base": 2, "cand": 1} and len(st["out_of_order"]) == 1


def test_status_equals_compare_paired_on_the_same_runs(home, tmp_path, capsys):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--n", "3", "--json")
    chk = out["check"]
    for v in (("pass", "fail"), ("pass", "pass"), ("fail", "pass")):
        record(capsys, home, chk, "base", v[0], "--type", "bug_fix", "--repo", "acme/app")
        record(capsys, home, chk, "cand", v[1])
    _, st, _ = lm(capsys, "check-change", "status", chk, "--json")
    docs = [Store(home).run_doc(r["run"]) for r in Store(home).runs()]
    base = [d for d in docs if d["run"]["ext"][compare.CHECK_EXT]["arm"] == "base"]
    cand = [d for d in docs if d["run"]["ext"][compare.CHECK_EXT]["arm"] == "cand"]
    direct = compare.paired(base, cand, rule="tests").to_dict()
    for k in ("n_pairs", "gain", "p_better", "cost_base", "cost_cand", "cpar_base", "cpar_cand"):
        assert st["comparison"][k] == direct[k]


def test_from_runs_compares_labelled_runs_already_in_the_store(home, tmp_path, capsys):
    for i, (vb, vc) in enumerate((("pass", "pass"), ("fail", "pass"), ("pass", "fail"), ("fail", "pass"))):
        task = tmp_path / f"t{i}.json"
        task.write_text(json.dumps({"id": f"tsk_hand{i:02d}", "type": "bug_fix", "repo": "acme/app"}))
        for label, v in (("old", vb), ("new", vc)):
            code, _, err = lm(capsys, "run", "record", "--task-file", str(task), *SOLO, "--label", f"config={label}",
                              "--cwd", f"implement={tmp_path / 'wt'}", "--since", "1h", "--verified", f"tests={v}",
                              "--no-commits", "--no-fit", "--json")
            assert code == 0, err
    code, out, err = lm(capsys, "check-change", "from-runs", "--base-label", "config=old", "--cand-label",
                        "config=new", "--json")
    assert code == 0, err
    assert out["comparison"]["n_pairs"] == 4
    assert out["base"]["runs"] == 4 and out["cand"]["runs"] == 4
    code, _, err = lm(capsys, "check-change", "from-runs", "--base-label", "config=old", "--cand-label",
                      "config=old", "--json")
    assert code != 0 and "both" in err


def test_run_record_label_writes_run_labels(home, tmp_path, capsys):
    code, out, err = lm(capsys, "run", "record", "--type", "bug_fix", "--repo", "acme/app", *SOLO,
                        "--label", "config=new", "--label", "team=core", "--cwd", f"implement={tmp_path / 'wt'}",
                        "--since", "1h", "--verified", "tests=pass", "--no-commits", "--no-fit", "--json")
    assert code == 0, err
    assert Store(home).run_doc(out["run"])["run"]["labels"] == {"config": "new", "team": "core"}


def test_check_flags_are_refused_where_they_would_be_ignored(home, tmp_path, capsys):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--n", "1", "--json")
    chk = out["check"]
    code, _, err = lm(capsys, "run", "record", "--arm", "base", "--type", "bug_fix", "--repo", "acme/app", *SOLO,
                      "--cwd", f"implement={tmp_path / 'wt'}", "--since", "1h", "--no-commits", "--no-fit", "--json")
    assert code != 0 and "--check" in err
    code, _, err = lm(capsys, "run", "record", "--run", "run_x", "--check", chk, "--arm", "base",
                      "--cwd", f"implement={tmp_path / 'wt'}", "--no-commits", "--no-fit", "--json")
    assert code != 0 and "--check" in err
    code, _, err = lm(capsys, "run", "record", "--label", "config", "--type", "bug_fix", "--repo", "acme/app", *SOLO,
                      "--cwd", f"implement={tmp_path / 'wt'}", "--since", "1h", "--no-commits", "--no-fit", "--json")
    assert code != 0 and "KEY=VALUE" in err
    record(capsys, home, chk, "base", "pass", "--type", "bug_fix", "--repo", "acme/app")
    code, _, err = lm(capsys, "run", "record", "--check", chk, "--arm", "base", *SOLO, "--cwd",
                      f"implement={tmp_path / 'wt'}", "--since", "1h", "--verified", "tests=pass", "--no-commits",
                      "--no-fit", "--json")
    assert code != 0 and "every pair" in err
    code, _, err = lm(capsys, "run", "record", "--check", chk, "--arm", "cand", "--type", "bug_fix", "--repo",
                      "acme/app", *SOLO, "--cwd", f"implement={tmp_path / 'wt'}", "--since", "1h", "--no-commits",
                      "--no-fit", "--json")
    assert code != 0 and "has its task" in err  # the pair's task comes from its first run


def test_status_answers_before_any_run_and_with_one_arm_only(home, tmp_path, capsys):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "old", "--cand", "new", "--n", "2", "--json")
    chk = out["check"]
    for _ in range(2):
        code, st, err = lm(capsys, "check-change", "status", chk, "--json")
        assert code == 0, err
        assert st["comparison"]["verdict"] == "more" and st["next"]["pair"] == "p01"
        code, text, err = lm(capsys, "check-change", "status", chk)
        assert code == 0 and "none yet" in text and "next pair worth" not in text, err  # no price, no value line
        record(capsys, home, chk, "base", "pass", "--type", "bug_fix", "--repo", "acme/app")
    assert st["comparison"]["cost_cand"] is None and st["recorded"] == {"base": 1, "cand": 0}


TASK = ("--type", "bug_fix", "--repo", "acme/app")  # a check made with --n names no task


def status_both_ways(capsys, chk):
    code, st, err = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 0, err
    code, text, err = lm(capsys, "check-change", "status", chk)
    assert code == 0, err
    return st, text


def test_status_answers_with_open_runs_only_and_with_uncosted_runs(home, tmp_path, capsys, monkeypatch):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "old", "--cand", "new", "--n", "2", "--json")
    chk = out["check"]
    code, run, err = lm(capsys, "run", "start", "--check", chk, "--arm", "base", *SOLO, *TASK, "--json")
    assert code == 0, err
    st, text = status_both_ways(capsys, chk)
    assert st["open"] == [run["run"]] and st["comparison"]["verdict"] == "more" and "open runs" in text
    monkeypatch.setattr(finish_mod, "default_settle", lambda: lambda doc: (doc, {"unmatched": {}}))  # no log
    _, out, _ = lm(capsys, "check-change", "start", "--base", "old", "--cand", "new", "--n", "2", "--json")
    chk = out["check"]
    record(capsys, home, chk, "base", "pass", *TASK)
    record(capsys, home, chk, "cand", "pass")  # the second arm joins the pair's task
    st, text = status_both_ways(capsys, chk)
    assert st["comparison"]["cost_base"] is None and st["comparison"]["counts"]["no_cost"] == {"base": 1, "cand": 1}
    assert st["comparison"]["outcomes_only"] is True  # no dollar cost: the cost is unknown, not yet to come
    assert "cost per run: base unknown; candidate unknown" in text and "none yet" not in text, text


def test_a_fully_recorded_score_check_with_failed_judges_reaches_its_cap(home, tmp_path, capsys):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "old", "--cand", "new", "--n", "2",
                   "--rule", "score:quality", "--json")
    chk = out["check"]
    st, text = status_both_ways(capsys, chk)  # a score check before any scored pair
    assert st["comparison"]["verdict"] == "more" and "none yet" in text
    for _ in range(2):
        record(capsys, home, chk, "base", "pass", "--score", "quality=", *TASK)  # the judge gave no score
        record(capsys, home, chk, "cand", "pass", "--score", "quality=")
    st, text = status_both_ways(capsys, chk)
    c = st["comparison"]
    assert c["n_pairs"] == 0 and c["counts"]["pairs"] == 2 and st["next"] is None
    assert c["verdict"] == "reject" and any("cap of 2 pairs is reached" in r for r in c["reasons"])


WORKER = """
import sys, time
from pathlib import Path
from loopmath import cli
from loopmath.compare import checks
me, other, folder, argv = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
orig = checks.claim
def claim_then_wait(store, args):  # hold the claimed pair open a while, as a slow start would
    out = orig(store, args)
    (folder / me).write_text("claimed")
    deadline = time.monotonic() + 3
    while not (folder / other).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    return out
checks.claim = claim_then_wait
raise SystemExit(cli.main(argv))
"""


def two_starts(tmp_path, first, second):
    """Two `run start` processes, the second launched once the first has claimed its pair."""
    folder = tmp_path / f"barrier-{time.monotonic_ns()}"
    folder.mkdir()
    env = dict(os.environ, PYTHONPATH=str(Path(compare.__file__).resolve().parents[2]))
    procs = [subprocess.Popen([sys.executable, "-c", WORKER, "first", "second", str(folder), *first],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)]
    deadline = time.monotonic() + 60
    while not (folder / "first").exists():
        assert time.monotonic() < deadline and procs[0].poll() is None, procs[0].communicate()
        time.sleep(0.01)
    procs.append(subprocess.Popen([sys.executable, "-c", WORKER, "second", "first", str(folder), *second],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env))
    return [(p.wait(timeout=120), *p.communicate()) for p in procs]


def test_two_starts_at_once_share_the_pairs_slate_and_one_arm_is_never_registered_twice(home, tmp_path, capsys):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "old", "--cand", "new", "--tasks",
                   str(tasks_file(tmp_path, 3)), "--json")
    chk = out["check"]
    start = ("run", "start", "--check", chk, *SOLO, "--json")
    (cb, ob, eb), (cc, oc, ec) = two_starts(tmp_path, [*start, "--arm", "base"], [*start, "--arm", "cand"])
    assert cb == 0 and cc == 0, (eb, ec)
    ob, oc = json.loads(ob), json.loads(oc)
    pair = compare.checks.load(Store(home), chk)["pairs"][0]
    assert ob["check"]["pair"] == oc["check"]["pair"] == "p01"
    assert ob["slate"] == oc["slate"] == pair["slate"] and pair["runs"] == {"base": ob["run"], "cand": oc["run"]}
    (c1, o1, e1), (c2, o2, e2) = two_starts(tmp_path, [*start, "--arm", "base"], [*start, "--arm", "base"])
    assert c1 == 0 and c2 == 0, (e1, e2)
    assert {json.loads(o1)["check"]["pair"], json.loads(o2)["check"]["pair"]} == {"p02", "p03"}
    (c1, _, e1), (c2, _, e2) = two_starts(tmp_path, [*start, "--arm", "cand", "--pair", "p02"],
                                          [*start, "--arm", "cand", "--pair", "p02"])
    assert c1 == 0 and c2 != 0 and "already has its cand run" in e2, (e1, e2)
    assert len(list((home / "runs").glob("*.ocp.json"))) == 5  # the refused start opened no run


def test_a_start_that_fails_after_the_claim_leaves_the_pair_for_the_next_start(home, tmp_path, capsys):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "old", "--cand", "new", "--tasks",
                   str(tasks_file(tmp_path, 2)), "--json")
    chk = out["check"]
    code, _, err = lm(capsys, "run", "start", "--check", chk, "--arm", "base", "--workflow",
                      str(tmp_path / "missing.toml"), "--source", "designed", "--json")
    assert code != 0, err
    code, run, err = lm(capsys, "run", "start", "--check", chk, "--arm", "base", *SOLO, "--json")
    assert code == 0 and run["check"]["pair"] == "p01", err


def test_from_runs_refuses_imported_scores_on_different_scales(home, tmp_path, capsys):
    from loopmath import harness_runs as HR
    from tests.compare.test_compare_paired import scale_rows

    rows = tmp_path / "rows.jsonl"
    for b, c in zip(*scale_rows(12)):  # the reviewer's N2 probe, through adapt and a no-fit import
        HR.append(rows, b)
        HR.append(rows, c)
    assert lm(capsys, "adapt", "harness-runs", str(rows), "--out", str(tmp_path / "ocp"))[0] == 0
    code, _, err = lm(capsys, "run", "import", str(tmp_path / "ocp"), "--finish", "--no-fit", "--json")
    assert code == 0, err
    code, out, err = lm(capsys, "check-change", "from-runs", "--base-label", "arm=base", "--cand-label",
                        "arm=candidate", "--rule", "score:judge", "--max-pairs", "12", "--json")
    assert code != 0 and "mix score scales (fraction: 12 base and 0 candidate runs; linear: 0 base and 12" in str(out)


def test_horizon_runs_is_kept_by_the_check_and_read_by_status_and_from_runs(home, tmp_path, capsys):
    code, _, err = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--n", "2", "--horizon-runs", "0",
                      "--json")
    assert code != 0 and "--horizon-runs" in err
    _, out, _ = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                   str(tasks_file(tmp_path, 3)), "--horizon-runs", "500", "--json")
    chk = out["check"]
    assert out["settings"]["horizon_runs"] == 500
    for arm, v in (("base", "fail"), ("cand", "pass")):
        record(capsys, home, chk, arm, v)
    _, st, _ = lm(capsys, "check-change", "status", chk, "--json")
    assert st["comparison"]["horizon_runs"] == 500
    sel = ("--base-label", f"{chk}:base", "--cand-label", f"{chk}:cand", "--json")
    _, fr, err = lm(capsys, "check-change", "from-runs", *sel)
    assert fr["comparison"]["horizon_runs"] == 50, err  # the default, unless asked
    _, fr, err = lm(capsys, "check-change", "from-runs", *sel, "--horizon-runs", "500")
    assert fr["comparison"]["horizon_runs"] == 500, err


def test_status_names_the_calibrated_threshold_and_from_runs_reads_a_check_arm(home, tmp_path, capsys):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                   str(tasks_file(tmp_path, 2)), "--json")
    chk = out["check"]
    for arm, v in (("base", "fail"), ("cand", "pass"), ("base", "pass"), ("cand", "pass")):
        record(capsys, home, chk, arm, v)
    _, st, _ = lm(capsys, "check-change", "status", chk, "--json")
    cmp = st["comparison"]
    assert cmp["max_pairs"] == 2 and cmp["adopt_threshold"] >= 0.95 and cmp["bound"]["cap"] == 6
    assert cmp["verdict"] in ("adopt", "reject") and st["next"] is None  # two pairs is the cap: no more looks
    code, text, _ = lm(capsys, "check-change", "status", chk)
    assert f"adopt at {cmp['adopt_threshold']:.3f}" in text
    _, fr, err = lm(capsys, "check-change", "from-runs", "--base-label", f"{chk}:base", "--cand-label", f"{chk}:cand",
                    "--max-pairs", "2", "--json")
    assert fr["comparison"]["p_better"] == cmp["p_better"] and fr["base"]["runs"] == 2, err


# ---------------------------------------------------------------- the CLI tester's round 1 (h19-cli-r1)
def import_rows(capsys, tmp_path, *arms, name="ocp"):
    """Rows through `adapt harness-runs` and a no-fit `run import`, as the tester stored them."""
    from loopmath import harness_runs as HR

    rows = tmp_path / f"{name}.jsonl"
    for arm in arms:
        for r in arm:
            HR.append(rows, r)
    assert lm(capsys, "adapt", "harness-runs", str(rows), "--out", str(tmp_path / name))[0] == 0
    code, _, err = lm(capsys, "run", "import", str(tmp_path / name), "--finish", "--no-fit", "--json")
    assert code == 0, err


ARMS = ("--base-label", "arm=base", "--cand-label", "arm=candidate")


def test_from_runs_reads_the_runs_own_rule_by_default_and_says_when_no_pair_is_scored(home, tmp_path, capsys):
    from tests.compare.test_compare_paired import verdict_rows

    import_rows(capsys, tmp_path, *verdict_rows(3))  # the tester's #30: accepted verdicts, as plugin-evals.jsonl
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--json")
    assert code == 0, err
    assert out["rule"] == "accept" and out["comparison"]["n_pairs"] == 3 and out["comparison"]["p_better"] is not None
    code, text, err = lm(capsys, "check-change", "from-runs", *ARMS)
    assert code == 0 and "mode accept (the runs' rule: accepted)" in text.splitlines()[0], err  # the tester's #238
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "tests", "--json")
    c = out["comparison"]
    assert code == 0 and c["n_pairs"] == 0 and c["verdict"] == "more", err
    assert c["p_better"] is None and c["gain"] is None and c["next_pair_value_usd"] is None
    assert c["pair_price_usd"] is None  # no scored pair, no next-pair price
    code, text, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "tests")
    assert code == 0 and "no pair is scored under the rule tests" in text, err
    assert "accepted (3 of 3 pairs; the runs' own rule, also --rule accept), review (3 of 3 pairs)" in text, text
    assert "accept (3 of 3" not in text  # the tester's #239: one name per rule
    assert "gain: none yet" in text and "gain: mean" not in text and "next pair worth" not in text
    assert "price $" not in text and "USD" not in text
    assert "P(better) ?" in text


def test_from_runs_refuses_runs_judged_by_different_rules(home, tmp_path, capsys):
    from tests.compare.test_compare_paired import scale_rows, verdict_rows

    judged = [dict(r, id="judge-cand", task_id="task-x", pair="px") for r in scale_rows(1)[1]]  # the default rule
    import_rows(capsys, tmp_path, *verdict_rows(3), judged)
    code, _, err = lm(capsys, "check-change", "from-runs", *ARMS)
    assert code == 1 and "judged by different rules (accepted: 3 base and 3 candidate runs; tests: 0 base and 1 " \
                          "candidate runs)" in err and "--rule accepted" in err and "--rule tests" in err, err
    code, _, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "accepted")
    assert code == 0, err


def test_a_check_with_base_runs_judged_by_tests_and_candidate_runs_by_review_exits_1(home, tmp_path, capsys,
                                                                                      monkeypatch):
    from loopmath.store import fitjob

    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})
    _, out, _ = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                   str(tasks_file(tmp_path, 6)), "--budget-usd", "50", "--json")
    chk = out["check"]
    cand = []
    for i in range(3):
        record(capsys, home, chk, "base", "pass" if i else "fail", "--label", "arm=base")
        code, run, err = lm(capsys, "run", "record", "--check", chk, "--arm", "cand", *SOLO, "--cwd",
                            f"implement={home.parent / 'wt'}", "--since", "1h", "--verified", "review=accept",
                            "--no-commits", "--no-fit", "--label", "arm=candidate", "--json")
        assert code == 0, err
        cand.append(run["run"])
    assert lm(capsys, "outcome", "--rule", "review", *cand)[0] == 0
    msg = ("the runs are judged by different rules (review: 0 base and 3 candidate runs; tests: 3 base and 0 "
           "candidate runs); compare them under one rule: --rule review or --rule tests")
    for argv in (("status", chk), ("from-runs", *ARMS)):
        code, out, err = lm(capsys, "check-change", *argv)
        assert code == 1 and msg in err and "P(better)" not in out, (argv, err)
    for rule in ("review", "tests"):
        code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", rule, "--json")
        assert code == 0 and out["comparison"]["n_pairs"] == 0, (rule, err)


def test_from_runs_refuses_two_different_rules_of_one_name(home, capsys):
    """Two rules named quality, one requiring tests and one accepted, are two rules; read as one, a check adopts."""
    from loopmath import harness_runs as HR
    from loopmath.types import DEFAULT_RULE

    def doc(arm, i, requires):
        d = HR.to_ocp(HR.row(f"t{i}", id=f"{arm}-{i}", pair=f"p{i}", arm=arm, outcome={"accepted": True},
                             cost={"usd": 0.1}))
        d["run"]["acceptance_rule"] = dict(DEFAULT_RULE.to_dict(), name="quality", requires=[requires])
        proto = d["run"]["signals"][0]
        d["run"]["signals"] = [dict(proto, name=n, value=v) for n, v in (("tests", "fail"), ("accepted", "pass"))]
        return d

    store = Store(home)
    for i in range(12):
        store.import_run(doc("base", i, "tests"))
        store.import_run(doc("candidate", i, "accepted"))
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS)  # not an adopt at P(better) 1
    assert code == 1 and "P(better)" not in out, out
    assert ("the runs are judged by 2 different rules named quality (quality, requires accepted: 0 base and 12 "
            "candidate runs; quality, requires tests: 12 base and 0 candidate runs); compare them under one rule: "
            "--rule with one of tests, review, accepted") in err, err
    for rule in ("tests", "accepted"):
        code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", rule, "--json")
        assert code == 0 and out["comparison"]["n_pairs"] == 12, (rule, err)


def test_from_runs_names_the_score_rule_of_the_selected_signal_not_an_older_one(home, capsys):
    """An older judge score says higher and the selected one lower: the default must name score:judge:lower,
    the rule that scores all four pairs, not say that no other rule scores one."""
    from loopmath import harness_runs as HR
    from loopmath.store.ids import new_id

    store = Store(home)
    for i in range(4):
        for arm in ("base", "candidate"):
            out = {"score": 0.8 if arm == "base" else 0.4, "name": "judge", "scale": "fraction", "better": "lower",
                   "observed_at": "2026-10-18T12:00:00-07:00"}
            d = HR.to_ocp(HR.row(f"t{i}", id=f"{arm}-{i}", pair=f"p{i}", arm=arm, outcome=out, cost={"usd": 0.1}))
            now = d["run"]["signals"][0]
            old = dict(now, id=new_id("sig"), value=0.9, better="higher", observed_at="2026-10-17T12:00:00-07:00")
            d["run"]["signals"] = [old, now]
            store.import_run(d)
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--json")
    assert code == 0 and out["comparison"]["n_pairs"] == 0, err
    assert "the rules that score them: score:judge:lower (4 of 4 pairs)" in out["comparison"]["reasons"][0], out
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "score:judge:lower", "--json")
    assert code == 0 and out["comparison"]["n_pairs"] == 4, err
    code, _, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "nosuchrule")
    assert code == 1 and "score names present: judge (--rule score:judge:lower)" in err, err


def test_from_runs_refuses_an_unknown_rule_or_score_name_and_names_what_is_there(home, tmp_path, capsys):
    from tests.compare.test_compare_paired import verdict_rows

    import_rows(capsys, tmp_path, *verdict_rows(3))
    code, _, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "nosuchrule")  # the tester's #186
    assert code == 1 and "nosuchrule" in err, err  # one name per rule, and only those that score a run (CLI r3 N3)
    assert "the rules that score these runs: accepted (the runs' own rule, also accept), review;" in err, err
    code, _, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "score:nosuch")  # the tester's #187
    assert code == 1 and "no selected run has a score named nosuch" in err and "score names present: none" in err, err
    for rule in ("accepted", "tests", "review", "accept"):
        code, _, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", rule)
        assert code == 0, (rule, err)


def test_a_stop_that_keeps_the_base_says_why_in_text_and_json(home, tmp_path, capsys):
    from tests.compare.test_compare_paired import verdict_rows

    import_rows(capsys, tmp_path, *verdict_rows(3))  # 6 runs at $0.30 each
    sel = (*ARMS, "--rule", "accepted", "--budget-usd", "0.5")  # the tester's #116: P(better) undecided
    code, out, err = lm(capsys, "check-change", "from-runs", *sel, "--json")
    assert code == 0 and (out["comparison"]["verdict"], out["comparison"]["stop"]) == ("reject", "budget"), err
    code, text, err = lm(capsys, "check-change", "from-runs", *sel)
    assert text.splitlines()[1].startswith("stop: budget spent, keep the base  P(better)"), text
    assert "verdict: reject" not in text
    _, out, _ = lm(capsys, "check-change", "start", "--base", "a", "--cand", "b", "--n", "2", "--budget-usd", "0",
                   "--json")  # the tester's #107 and #113: a zero budget is spent before the first run
    st, text = status_both_ways(capsys, out["check"])
    assert st["comparison"]["stop"] == "budget" and "stop: budget spent, keep the base" in text


def test_start_refuses_a_missing_path_the_same_arm_a_negative_budget_and_a_floor_out_of_range(home, tmp_path,
                                                                                             capsys):
    from tests.compare.test_compare_paired import verdict_rows

    (tmp_path / "a-file").write_text("x")
    code, _, err = lm(capsys, "check-change", "start", "--base", str(tmp_path / "nope"), "--cand", "b", "--n", "2")
    assert code == 2 and "no folder" in err, err  # the tester's #104: a missing path exits 2 (CLI r3 N2)
    bad = ((["--base", str(tmp_path / "a-file"), "--cand", "b"], "not a folder"),
           (["--base", "x", "--cand", "x"], "the same arm"),  # #105
           (["--base", "a", "--cand", "b", "--budget-usd", "-5"], "--budget-usd"),  # #108
           (["--base", "a", "--cand", "b", "--budget-usd", "nan"], "--budget-usd"),
           (["--base", "a", "--cand", "b", "--floor", "-1"], "floor"),
           (["--base", "a", "--cand", "b", "--objective", "gain", "--floor", "1"], "floor"))
    for argv, said in bad:
        code, _, err = lm(capsys, "check-change", "start", *argv, "--n", "2")
        assert code == 1 and said in err, (argv, err)
    assert not (home / "checks").exists()  # no refused start wrote a check
    trees = tmp_path / "trees"
    (trees / "a").mkdir(parents=True)
    code, _, err = lm(capsys, "check-change", "start", "--base", str(trees / "a"), "--cand",
                      str(trees / "a" / ".." / "a"), "--n", "2")
    assert code == 1 and "the same arm" in err, err
    code, _, err = lm(capsys, "check-change", "start", "--base", "feature/x", "--cand", "main", "--n", "2")
    assert code == 0, err  # a label with a slash inside is a label
    import_rows(capsys, tmp_path, *verdict_rows(3))
    code, _, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "accepted", "--floor", "-1")  # #185
    assert code == 1 and "floor" in err, err


def test_start_prints_the_harness_flag_for_tree_arms_and_the_run_records_that_harness(home, tmp_path, capsys):
    trees = {}
    for arm, text in (("base", "be brief\n"), ("cand", "explain each step\n")):
        trees[arm] = tmp_path / "trees" / arm
        trees[arm].mkdir(parents=True)
        (trees[arm] / "CLAUDE.md").write_text(text)
    code, out, err = lm(capsys, "check-change", "start", "--base", str(trees["base"]), "--cand", str(trees["cand"]),
                        "--n", "2", "--json")  # the tester's #95
    assert code == 0, err
    chk, joined = out["check"], "\n".join(out["next"]["commands"])
    for arm in ("base", "cand"):
        line = next(c for c in out["next"]["commands"] if f"--arm {arm} " in c)
        assert f"--harness {out[arm]['path']}" in line, joined
    run = record(capsys, home, chk, "base", "pass", *TASK, "--harness", out["base"]["path"])
    ext = Store(home).run_doc(run["run"])["run"]["ext"]
    assert ext["dev.loopmath.harness"]["digest"] == out["base"]["harness"]
    _, out, _ = lm(capsys, "check-change", "start", "--base", "old", "--cand", "new", "--n", "2", "--json")
    assert not any("--harness" in c for c in out["next"]["commands"])  # label arms have no tree


def test_from_runs_names_the_rows_of_two_runs_on_one_pair_key_and_the_fix(home, tmp_path, capsys):
    from tests.compare.test_compare_paired import scale_rows

    base, cand = scale_rows(1, scales=("fraction", "fraction"), scores=(0.5, 0.7))
    dup = dict(base[0], id="base-0-dup", outcome=dict(base[0]["outcome"], score=0.6))  # the tester's #131
    import_rows(capsys, tmp_path, base + [dup], cand)
    code, _, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "score:judge")
    assert code == 1 and "row base-0," in err and "row base-0-dup)" in err and "drop one of them" in err, err


def test_status_of_a_check_of_n_pairs_says_its_own_cap(home, tmp_path, capsys):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "a", "--cand", "b", "--n", "4", "--json")  # #95, #96
    assert out["rule"] == "accept"  # the runs' own rule, read when they are recorded
    st, text = status_both_ways(capsys, out["check"])
    assert "up to this check's cap of 4 pairs" in text and "up to the cap of 6 pairs" not in text, text
    assert st["comparison"]["max_pairs"] == 4


def _uncosted_check(capsys, home, tmp_path, monkeypatch, n, *start):
    monkeypatch.setattr(finish_mod, "default_settle", lambda: lambda doc: (doc, {"unmatched": {}}))  # no log
    _, out, _ = lm(capsys, "check-change", "start", "--base", "a", "--cand", "b", "--tasks",
                   str(tasks_file(tmp_path, n)), *start, "--json")
    return out["check"]


def test_a_check_with_no_dollar_cost_decides_on_the_outcomes_and_never_keeps_the_base_for_it(home, tmp_path, capsys,
                                                                                              monkeypatch):
    chk = _uncosted_check(capsys, home, tmp_path, monkeypatch, 6)  # the CLI tester's #425 and #426
    record(capsys, home, chk, "base", "fail")
    record(capsys, home, chk, "cand", "pass")
    _, text = status_both_ways(capsys, chk)
    assert ", 1 scored pair\n" in text, text
    for _ in range(5):
        record(capsys, home, chk, "base", "fail")
        record(capsys, home, chk, "cand", "pass")
    st, text = status_both_ways(capsys, chk)
    c = st["comparison"]
    assert (c["verdict"], c["stop"], c["objective"], c["outcomes_only"]) == ("adopt", "decided", "gain", True), c
    assert c["p_better"] > c["adopt_threshold"] and c["n_pairs"] == 6
    assert text.splitlines()[1].startswith("verdict: adopt  P(better) 0.9"), text
    assert "cost per run: base unknown; candidate unknown" in text and "6 scored pairs" in text
    assert "the cost is unknown (12 of 12 runs have no dollar cost), so the answer is on the outcomes alone" in text
    assert "keep the base" not in text


def test_a_check_with_a_budget_and_a_run_with_no_dollar_cost_refuses_and_still_shows_the_check(home, tmp_path,
                                                                                               capsys, monkeypatch):
    chk = _uncosted_check(capsys, home, tmp_path, monkeypatch, 2, "--budget-usd", "20")
    run = record(capsys, home, chk, "base", "pass")["run"]
    code, text, err = lm(capsys, "check-change", "status", chk)
    assert code == 1 and text.startswith(f"check {chk}: b (cand) against a (base)"), (text, err)
    assert f"the check's --budget-usd needs every run's dollar cost, and 1 of 1 runs has no dollar cost ({run})" in err
    code, out, _ = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 1 and out["ok"] is False and out["check"] == chk and out["recorded"] == {"base": 1, "cand": 0}
    assert "--budget-usd needs every run's dollar cost" in out["error"]


def test_from_runs_with_no_dollar_cost_refuses_cpar_by_name_and_otherwise_decides_on_the_gain(home, tmp_path, capsys,
                                                                                               monkeypatch):
    from tests.compare.test_compare_paired import _no_cost, verdict_rows

    monkeypatch.setattr(finish_mod, "default_settle", lambda: lambda doc: (doc, {"unmatched": {}}))  # no log
    base, cand = verdict_rows(6, ((False,), (True,)))
    import_rows(capsys, tmp_path, _no_cost(base), _no_cost(cand))
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "accepted", "--max-pairs", "6", "--json")
    assert code == 0 and (out["comparison"]["verdict"], out["comparison"]["objective"]) == ("adopt", "gain"), err
    code, _, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "accepted", "--objective", "cpar")
    assert code == 1 and "--objective cpar, the cost per accepted result, needs every run's cost" in err, err
    assert "use --objective gain" in err


def test_from_runs_takes_the_gates_selector_forms_and_lists_the_arms_when_none_match(home, capsys):
    from store_helpers import finished_doc

    a, b, c = ("sha256:" + "ab" * 32, "sha256:" + "ac" * 32, "sha256:" + "f3" * 32)
    st = Store(home)
    for i, (digest, arm, verdict) in enumerate([(a, "base", "fail"), (a, "base", "fail"), (b, "candidate", "pass"),
                                                (b, "candidate", "pass"), (c, "other", "pass")]):
        doc = finished_doc(f"run_d{i}", usd=0.5)
        doc["run"]["task"]["id"] = f"tsk_{i % 2}"
        doc["run"]["labels"] = {"arm": arm}
        doc["run"]["ext"] = {"dev.loopmath.harness": {"schema": "loopmath.harness/0.1", "digest": digest,
                                                      "source": "hand"}}
        doc["run"]["signals"] = [{"id": f"sig_d{i}", "kind": "verdict", "name": "tests", "value": verdict,
                                  "observed_at": "2026-09-20T10:05:00-07:00", "tier": "verified"}]
        st.import_run(doc, check=False)
    code, whole, err = lm(capsys, "check-change", "from-runs", "--base-label", a, "--cand-label", b, "--json")
    assert code == 0 and whole["comparison"]["n_pairs"] == 2, err
    for sel in ((a[:15], b[:15]), ("digest:" + a, "arm:candidate"), ("arm=base", "digest:" + b[:12])):  # #421, #423
        code, out, err = lm(capsys, "check-change", "from-runs", "--base-label", sel[0], "--cand-label", sel[1],
                            "--json")
        assert code == 0 and out["comparison"]["pairs"] == whole["comparison"]["pairs"], (sel, err)
    code, _, err = lm(capsys, "check-change", "from-runs", "--base-label", "sha256:a", "--cand-label", c)
    assert code == 1 and f"--base-label sha256:a matches 2 digests: {a}, {b}; give more of one" in err, err
    code, out, err = lm(capsys, "check-change", "from-runs", "--base-label", "arm=nope", "--cand-label", b, "--json")
    assert code == 1 and out["ok"] is False and "--base-label arm=nope matches no finished run in" in out["error"]
    assert "arms: arm=base (2 runs), arm=candidate (2 runs), arm=other (1 run); digests: sha256:abababababab..." \
        in out["error"], out["error"]


def test_from_runs_refuses_an_empty_digest_prefix_and_heads_with_the_digest_a_prefix_resolved_to(home, capsys):
    from store_helpers import finished_doc

    a = "sha256:" + "ab" * 32
    st = Store(home)
    for i, (arm, verdict) in enumerate([("base", "fail"), ("base", "fail"), ("candidate", "pass"),
                                        ("candidate", "pass")]):
        doc = finished_doc(f"run_e{i}", usd=0.5)
        doc["run"]["task"]["id"] = f"tsk_{i % 2}"
        doc["run"]["labels"] = {"arm": arm}
        if arm == "base":  # one digest in the store: an empty prefix would select it (the tester's #470)
            doc["run"]["ext"] = {"dev.loopmath.harness": {"schema": "loopmath.harness/0.1", "digest": a,
                                                          "source": "hand"}}
        doc["run"]["signals"] = [{"id": f"sig_e{i}", "kind": "verdict", "name": "tests", "value": verdict,
                                  "observed_at": "2026-09-20T10:05:00-07:00", "tier": "verified"}]
        st.import_run(doc, check=False)
    for empty in ("sha256:", "digest:sha256:"):
        code, out, err = lm(capsys, "check-change", "from-runs", "--base-label", empty, "--cand-label",
                            "arm:candidate", "--json")
        assert code == 1 and out["ok"] is False, (empty, out, err)
        assert f"--base-label {empty} has no hex after sha256:; give a harness digest, or a prefix of one" in out["error"]
    code, text, err = lm(capsys, "check-change", "from-runs", "--base-label", a[:15], "--cand-label", "arm:candidate")
    assert code == 0 and text.splitlines()[0].startswith(f"arm:candidate (2 runs) against {a} (2 runs), "), text
    code, out, err = lm(capsys, "check-change", "from-runs", "--base-label", a[:15], "--cand-label", "arm:candidate",
                        "--json")
    assert out["base"] == {"label": a[:15], "runs": 2, "digest": a} and "digest" not in out["cand"], out
