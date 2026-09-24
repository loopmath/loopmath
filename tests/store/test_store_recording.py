"""Recording commands end to end through the CLI: run start, attempt, artifact, outcome, finish, import."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from store_helpers import EXPLORE_CFG, REC_ID, USUAL_CFG, cli, fake_settle, finished_doc, install_rec, task, usual_config

from loopmath.store import Store
from loopmath.store import finish as finish_mod
from loopmath.store import fit_state, fitjob
from loopmath.store import runs as R
from loopmath.store.fitjob import run_job
from loopmath.store.home import Conflict
from loopmath.store.ids import new_id, now_iso
from loopmath.types import DEFAULT_RULE, Signal

REAL_SETTLE = finish_mod.default_settle  # lane 2's settle_run; the `home` fixture swaps in a fake
LOGMATCH_FIXTURES = Path(__file__).resolve().parents[1] / "logmatch" / "fixtures"
CC_SESSION = "00000000-0000-4000-8000-000000000001"  # lane 2's Claude Code fixture session


@pytest.fixture
def spawned(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(str(home)) or {"started": True, "pid": 0})
    return calls


@pytest.fixture
def home(tmp_path, monkeypatch, spawned):
    h = tmp_path / "lm"
    install_rec(h)
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle())
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "0f0e0d0c-1111-2222-3333-444455556666")
    return h


def _start(capsys, home, *extra):
    code, out, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "loopmath/loopmath",
                         "--title", "Add --since", "--config", EXPLORE_CFG, "--rec", REC_ID, "--source", "exploration",
                         *extra, "--json")
    assert code == 0, err
    return out


def test_run_start_writes_skeleton_index_row_and_before_receipt(capsys, home):
    out = _start(capsys, home, "--base-commit", "672fd0d")
    assert list(out)[0] == "schema" and out["schema"] == "loopmath.run.start/1"
    assert out["config"] == EXPLORE_CFG and out["pieces"] == ["implement", "review"]
    doc = json.loads((home / "runs" / f"{out['run']}.ocp.json").read_text())
    assert doc["ocp"] == "0.3" and doc["privacy"] == {"profile": "metadata_only"}
    run = doc["run"]
    assert run["configuration"]["source"] == "exploration" and run["configuration"]["rec"] == REC_ID
    assert run["provenance"]["kind"] == "designed" and run["acceptance_rule"]["name"] == "tests"
    assert run["task"]["base_commit"] == "672fd0d"
    assert [n["id"] for n in doc["nodes"]] == ["implement", "review"]
    assert all("gate" not in n for n in doc["nodes"])
    rows = [json.loads(line) for line in (home / "runs" / "index.jsonl").read_text().splitlines()]
    assert rows[-1]["run"] == out["run"] and rows[-1]["state"] == "open" and rows[-1]["task_type"] == "feature"
    rct = out["receipt"]["receipt"]
    receipt = json.loads((home / "receipts" / f"{rct}.json").read_text())
    assert receipt["before"]["config"] == EXPLORE_CFG and receipt["after"] is None and receipt["fit"] == "fit_20260923160000"


def test_run_start_resolves_config_from_stored_run_and_reports_unknown(capsys, home):
    first = _start(capsys, home)
    (home / "recs" / f"{REC_ID}.json").unlink()  # D13: a config seen in a stored run still resolves
    code, out, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "loopmath/loopmath",
                         "--config", EXPLORE_CFG, "--source", "alternative", "--json")
    assert code == 0, err
    assert out["config"] == first["config"]
    code, _, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "r", "--config", "cfg_000000000000",
                       "--source", "usual")
    assert code == 2 and "not in any stored recommendation or run" in err


def test_run_start_takes_the_settings_in_a_workflow_file(capsys, home, tmp_path):
    """D81: a workflow file's `[settings.<piece>]` are the run's settings, `--set` overrides per piece and is
    needed only for pieces the file leaves unset. The file alone gives the id `workflows validate` prints."""
    from loopmath.types import Setting
    from loopmath.workflows import format as wf_format
    from loopmath.workflows.ids import make_config

    wf = wf_format.catalog()["plan_implement_review"]
    low = {p.id: Setting(harness="codex", model="gpt-6-sol", effort="low") for p in wf.pieces}
    full = tmp_path / "full.toml"
    full.write_text(wf_format.dump_workflow(wf, low), encoding="utf-8")
    code, checked, err = cli(capsys, home, "workflows", "validate", str(full), "--json")
    assert code == 0 and checked["valid"], err

    def start(path, *sets):
        return cli(capsys, home, "run", "start", "--type", "bug_fix", "--repo", "r", "--workflow", str(path),
                   *[a for s in sets for a in ("--set", s)], "--source", "user_edit", "--json")

    code, out, err = start(full)
    assert code == 0, err
    assert out["config"] == checked["config"] and out["config_requested"] is None
    code, out, err = start(full, "review=claude-code:claude-opus-5-5:high")
    assert code == 0, err
    high = Setting(harness="claude-code", model="claude-opus-5-5", effort="high")
    assert out["config"] == make_config(wf, {**low, "review": high}).id != checked["config"]
    part = tmp_path / "part.toml"
    part.write_text(wf_format.dump_workflow(wf, {"plan": low["plan"]}), encoding="utf-8")
    code, _, err = start(part)
    assert code == 1 and "piece(s) implement, review:" in err and "--set" in err
    code, out, err = start(part, "implement=codex:gpt-6-sol:low", "review=codex:gpt-6-sol:low")
    assert code == 0, err
    assert out["config"] == checked["config"]


def test_run_start_takes_a_configuration_file(capsys, home, tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(usual_config().to_dict()))
    code, out, err = cli(capsys, home, "run", "start", "--type", "bug_fix", "--repo", "r", "--config", str(path),
                         "--source", "usual", "--json")
    assert code == 0, err
    assert out["config"] == USUAL_CFG and out["config_requested"] is None and "canonical" not in err
    stale = {**usual_config().to_dict(), "id": "cfg_0123456789ab"}  # a producer with an older id scheme
    path.write_text(json.dumps(stale))
    code, out, err = cli(capsys, home, "run", "start", "--type", "bug_fix", "--repo", "r", "--config", str(path),
                         "--source", "usual", "--json")
    assert code == 0, err
    assert out["config"] == USUAL_CFG and out["config_requested"] == "cfg_0123456789ab"
    assert "recorded under its canonical id " + USUAL_CFG in err


def test_run_start_with_rec_and_no_config_names_what_the_rec_offers(capsys, home):
    """Dogfood: recommend's text names no configuration ids, so the error names them."""
    code, _, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "r", "--rec", REC_ID,
                       "--source", "usual")
    assert code == 1 and f"({REC_ID} offers goal " in err
    assert f"usual {USUAL_CFG}" in err and f"exploration best_value {EXPLORE_CFG}" in err


def test_run_start_argument_errors(capsys, home):
    code, _, err = cli(capsys, home, "run", "start", "--type", "nonsense", "--repo", "r", "--config", EXPLORE_CFG,
                       "--source", "usual")
    assert code == 1 and "task type" in err
    code, _, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "r", "--source", "usual")
    assert code == 1 and "--config" in err
    code, _, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "r", "--config", EXPLORE_CFG,
                       "--source", "usual", "--slate", "slt_nothere")
    assert code == 2 and "--new-slate" in err


def test_attempts_session_self_and_errors(capsys, home, monkeypatch):
    run = _start(capsys, home)["run"]
    code, out, err = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "review", "--harness", "claude-code",
                         "--model", "claude-opus-5-5", "--session", "self", "--json")
    assert code == 0, err
    assert out["session"] == "0f0e0d0c-1111-2222-3333-444455556666"
    code, _, err = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
                       "--model", "gpt-6-sol", "--session", "self")
    assert code == 1 and "codex" in err  # D26: never guessed for Codex
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID")
    code, _, err = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "review", "--harness", "claude-code",
                       "--model", "claude-opus-5-5", "--session", "self")
    assert code == 1 and "CLAUDE_CODE_SESSION_ID" in err
    code, _, err = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "nope", "--harness", "codex",
                       "--model", "m", "--cwd", "/tmp")
    assert code == 1 and "no piece" in err
    code, _, err = cli(capsys, home, "run", "attempt", "--run", "run_missing", "--piece", "implement",
                       "--harness", "codex", "--model", "m", "--cwd", "/tmp")
    assert code == 2
    code, _, err = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
                       "--model", "m")
    assert code == 1 and "--cwd" in err
    doc = Store(home).run_doc(run)
    att = doc["attempts"][0]
    assert att["effort"] == "high"  # from the configuration's setting when --effort is absent
    assert att["ext"]["dev.loopmath.match"]["session_from"] == "self"


def test_an_end_before_the_start_exits_2(capsys, home):
    """D87: `--ended-at` before the attempt's `started_at` was accepted without a word."""
    run = _start(capsys, home)["run"]
    _, att, _ = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
                    "--model", "m", "--cwd", str(home), "--started-at", "2026-09-23T19:00:25-07:00", "--json")
    code, out, err = cli(capsys, home, "run", "attempt", "--run", run, "--end", att["attempt"], "--status", "done",
                         "--ended-at", "2026-09-23T19:00:00-07:00", "--json")
    assert code == 2 and out["ok"] is False
    assert "--ended-at 2026-09-23T19:00:00-07:00 is before attempt " + att["attempt"] in err
    assert "started (2026-09-23T19:00:25-07:00)" in err
    assert Store(home).run_doc(run)["attempts"][0]["status"] != "done"  # nothing written
    code, out, err = cli(capsys, home, "run", "attempt", "--run", run, "--end", att["attempt"], "--status", "done",
                         "--ended-at", "2026-09-23T19:00:25-07:00", "--json")
    assert code == 0 and out["ended_at"] == "2026-09-23T19:00:25-07:00", err


def test_full_flow_finish_costs_receipt_and_refit(capsys, home, spawned):
    run = _start(capsys, home)["run"]
    _, a1, _ = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
                   "--model", "gpt-6-sol", "--cwd", str(home), "--json")
    _, a2, _ = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "review", "--harness", "claude-code",
                   "--model", "claude-opus-5-5", "--session", "self", "--json")
    code, ended, err = cli(capsys, home, "run", "attempt", "--run", run, "--end", a1["attempt"], "--status", "done",
                           "--json")
    assert code == 0, err
    assert ended["ended_at"] and ended["ended_at"] == Store(home).run_doc(run)["attempts"][0]["ended_at"]
    code, art, err = cli(capsys, home, "run", "artifact", "--run", run, "--kind", "commit", "--path", "abc1234def",
                         "--by", a1["attempt"], "--read-by", a2["attempt"], "--json")
    assert code == 0, err
    assert art["version"] == 1
    code, sig, err = cli(capsys, home, "outcome", "--run", run, "--signal", "tests=pass", "--tier", "verified",
                         "--source", "ci", "--json")
    assert code == 0, err
    assert sig["state"] == "pending"
    assert (home / "signals" / f"{run}.jsonl").exists()
    code, res, err = cli(capsys, home, "run", "finish", "--run", run, "--json")
    assert code == 0, err
    assert res["schema"] == "loopmath.run.finish/1"
    assert res["cost"]["usd"] == pytest.approx(1.0) and res["cost"]["tokens"] == 3000
    assert res["matched"] == {"verified": 2, "heuristic": 0, "unmatched": []}
    assert res["evidence"]["z"] == 1.0 and res["fit_started"] is True and spawned == [str(home)]
    assert not (home / "signals" / f"{run}.jsonl").exists()
    doc = Store(home).run_doc(run)
    assert doc["run"]["ext"]["dev.loopmath.store"]["state"] == "finished" and doc["run"]["ended_at"]
    assert [s["name"] for s in doc["run"]["signals"]] == ["tests"]
    assert doc["run"]["receipt"]["before"]["rec"] == REC_ID and doc["run"]["receipt"]["after"]["cost_usd"] == 1.0
    assert [e["detail"] for e in doc["events"] if e["type"] == "receipt_written"] == [res["receipt"]]
    review = [a for a in doc["attempts"] if a["id"] == a2["attempt"]][0]
    assert review["status"] == "settled_unverified"  # never settled by the orchestrator
    receipt = json.loads((home / "receipts" / f"{res['receipt']}.json").read_text())
    assert receipt["after"]["z"] == 1.0 and receipt["scored"]["cost_in_interval"] is True
    row = Store(home).index_rows()[run]
    assert row["state"] == "finished" and row["cost_usd"] == pytest.approx(1.0)
    # a finished run takes no more attempts and cannot finish twice
    code, _, err = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
                       "--model", "m", "--cwd", "/tmp")
    assert code == 1 and "its attempts can no longer change" in err
    code, _, err = cli(capsys, home, "run", "artifact", "--run", run, "--kind", "commit", "--path", "fedcba9876",
                       "--by", a1["attempt"])  # a later commit, so late events find the run (D17)
    assert code == 0, err
    code, _, err = cli(capsys, home, "run", "finish", "--run", run)
    assert code == 1 and "already finished" in err


def test_finish_lists_the_validation_warnings(capsys, home):
    """Dogfood: finish printed "2 warning(s)" and never which."""
    run = _start(capsys, home)["run"]
    _, att, _ = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
                    "--model", "m", "--cwd", str(home), "--json")
    cli(capsys, home, "run", "artifact", "--run", run, "--kind", "notes", "--path", "n.md", "--by", att["attempt"])
    code, out, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit")
    assert code == 0, err
    assert "validation: ok, 1 warning(s)\n  warning W200 $['artifacts'][0]['kind']['value']: " in out
    assert "'notes' is outside the recommended vocabulary" in out and "not checked" not in out


def test_unmatched_cost_is_unknown_not_zero(capsys, home, monkeypatch, spawned):
    monkeypatch.setattr(finish_mod, "default_settle", REAL_SETTLE)  # lane 2 over empty log folders (conftest)
    run = _start(capsys, home)["run"]
    cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex", "--model", "m",
        "--cwd", "/tmp")
    code, res, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--json")
    assert code == 0, err
    assert res["cost"]["usd"] is None and res["matched"]["unmatched"][0]["reason"]
    receipt = json.loads((home / "receipts" / f"{res['receipt']}.json").read_text())
    assert receipt["scored"]["cost_in_interval"] is None and receipt["after"]["cost"]["usd"] is None
    assert spawned == []


def test_outcome_values_are_checked(capsys, home):
    run = _start(capsys, home)["run"]
    for args, needle in [(("--signal", "tests=maybe"), "verdict"),
                         (("--signal", "q=abc", "--kind", "score"), "number"),
                         (("--signal", "q=1.5", "--kind", "score", "--scale", "fraction"), "[0, 1]"),
                         (("--signal", "incident=", "--kind", "event"), "reference"),
                         (("--signal", "novalue"), "NAME=VALUE")]:
        code, _, err = cli(capsys, home, "outcome", "--run", run, *args)
        assert code == 1 and needle in err, (args, err)
    code, out, err = cli(capsys, home, "outcome", "--run", run, "--signal", "perf=", "--kind", "score", "--json")
    assert code == 0, err
    assert out["value"] is None  # a declared, unmeasured score
    code, _, err = cli(capsys, home, "outcome", "--run", run, "--signal", "tests=pass", "--at-attempt", "att_nope")
    assert code == 2


def test_slates_join_check_base_commit_and_copy_preferences(capsys, home):
    a = _start(capsys, home, "--new-slate", "--base-commit", "672fd0d")
    slate = a["slate"]
    code, _, err = cli(capsys, home, "run", "start", "--task-file", "/dev/null", "--type", "feature", "--repo", "x",
                       "--config", EXPLORE_CFG, "--source", "usual", "--slate", slate)
    assert code == 1  # /dev/null is not a JSON task
    task_id = Store(home).run_doc(a["run"])["run"]["task"]["id"]
    code, _, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "loopmath/loopmath",
                       "--config", EXPLORE_CFG, "--source", "usual", "--slate", slate, "--base-commit", "fffffff")
    assert code == 1 and "same --task-file" in err  # task flags make a new task id; the error says what to do
    b_task = home / "task.json"
    b_task.write_text(json.dumps({"id": task_id, "type": "feature", "repo": "loopmath/loopmath"}))
    code, _, err = cli(capsys, home, "run", "start", "--task-file", str(b_task), "--config", EXPLORE_CFG,
                       "--source", "usual", "--slate", slate, "--base-commit", "fffffff")
    assert code == 1 and "base commit" in err
    code, b, err = cli(capsys, home, "run", "start", "--task-file", str(b_task), "--config", USUAL_CFG, "--rec", REC_ID,
                       "--source", "usual", "--slate", slate, "--base-commit", "672fd0d", "--json")
    assert code == 0, err
    store = Store(home)
    for run in (a["run"], b["run"]):
        assert store.run_doc(run)["run"]["slate"]["members"] == [a["run"], b["run"]]
    code, pref, err = cli(capsys, home, "outcome", "--slate", slate, "--prefer", b["run"], "--judge", "referee",
                          "--blinded", "--json")
    assert code == 0, err
    assert pref["members"] == [a["run"], b["run"]]
    for run in (a["run"], b["run"]):
        doc = store.run_doc(run)
        assert doc["run"]["preferences"][0]["winner"] == b["run"] and doc["run"]["slate"]["blinded"] is True
    code, _, err = cli(capsys, home, "outcome", "--slate", slate, "--prefer", "run_other")
    assert code == 1 and "not a member" in err


def test_import_stores_replaces_and_finishes(capsys, home, tmp_path):
    run = _start(capsys, home)["run"]
    doc = Store(home).run_doc(run)
    doc["run"]["ext"] = {}  # as an orchestrator would write it
    doc["producer"] = {"name": "herdr-dagr", "version": "1"}
    doc["attempts"] = [{"id": "att_x1", "node": "implement", "n": 1, "status": "done", "harness": "codex",
                        "model": {"raw": "gpt-6-sol"}, "started_at": "2026-09-23T10:00:00-07:00",
                        "ended_at": "2026-09-23T10:30:00-07:00"}]
    path = tmp_path / "x.ocp.json"
    path.write_text(json.dumps(doc))
    code, out, err = cli(capsys, home, "run", "import", str(path), "--json")
    assert code == 0, err
    assert out["run"] == run and out["finished"] is None and out["state"] == "open"
    stored = Store(home).run_doc(run)
    assert stored["producer"]["name"] == "herdr-dagr" and stored["run"]["ext"]["dev.loopmath.store"]["state"] == "open"
    code, out, err = cli(capsys, home, "run", "import", str(path), "--finish", "--json")
    assert code == 0, err
    assert out["finished"]["cost"]["usd"] == pytest.approx(0.5)
    stored = Store(home).run_doc(run)
    assert stored["run"]["ext"]["dev.loopmath.store"]["state"] == "finished"
    assert "receipt" not in stored["run"]  # run.receipt only in loopmath's own documents
    assert len([r for r in Store(home).index_rows().values()]) == 1
    future = tmp_path / "future.ocp.json"
    future.write_text(json.dumps({**doc, "ocp": "9.9"}))
    code, _, err = cli(capsys, home, "run", "import", str(future))
    assert code == 1 and "0.3" in err
    old = tmp_path / "old.ocp.json"
    old.write_text(json.dumps({**doc, "ocp": "0.1"}))
    code, out, err = cli(capsys, home, "run", "import", str(old), "--json")
    assert (code == 0 and out["migrated_from"] == "0.1") or (code == 1 and "migration" in err)  # lane 1 migrator or stub
    code, _, err = cli(capsys, home, "run", "import", str(tmp_path / "none.json"))
    assert code == 2


def test_import_of_a_run_loopmath_finished_keeps_it_finished(capsys, home, tmp_path):
    run = _start(capsys, home)["run"]
    cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex", "--model", "m",
        "--cwd", str(tmp_path))
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit")
    assert code == 0, err
    other = tmp_path / "other"
    code, out, err = cli(capsys, other, "run", "import", str(Store(home).run_path(run)), "--finish", "--json")
    assert code == 0, err
    assert out["state"] == "finished" and out["finished"] is None and "--finish skipped" in err
    assert Store(other).index_rows()[run]["state"] == "finished"
    assert Store(other).run_doc(run)["run"]["ended_at"] == Store(home).run_doc(run)["run"]["ended_at"]


def test_store_import_run_api_marks_finished(tmp_path):
    """Analyst D4: import_run(doc, finished=True) for onboard history."""
    store = Store(tmp_path / "lm")
    doc = finished_doc("run_hist1", source="habit", started="2026-09-01T10:00:00-07:00")
    del doc["run"]["started_at"]
    assert store.import_run(doc) == "run_hist1"
    stored = store.run_doc("run_hist1")
    assert stored["run"]["ended_at"] == "2026-09-01T10:00:00-07:00"
    assert [e["type"] for e in stored["events"]] == ["run_finished"]
    assert "events" not in doc  # the caller's document is not changed
    assert store.index_rows()["run_hist1"]["source"] == "habit"


def test_run_end_covers_events_after_the_last_attempt():
    """An artifact recorded after the last attempt ended moves the run's end, so events stay ascending (W150)."""
    doc = finished_doc("run_w150", started="2026-09-23T10:00:00-07:00")
    doc["events"] = [{"at": "2026-09-23T10:00:00-07:00", "type": "attempt_settled"},
                     {"at": "2026-09-23T10:05:00-07:00", "type": "artifact_written"}]
    finish_mod._close(doc, "2026-09-23T11:00:00-07:00")
    assert doc["run"]["ended_at"] == "2026-09-23T10:05:00-07:00"
    assert doc["events"][-1] == {"at": "2026-09-23T10:05:00-07:00", "type": "run_finished"}


def test_a_verdict_recorded_before_finish_is_not_late(capsys, home):
    """Dogfood: the tests verdict came after the attempt ended and before finish; lane 12's runs view marked it
    late because the run's end left out the signals folded in at finish."""
    from loopmath.store.ids import parse_ts
    from loopmath.views.runs import detail_from_doc

    run = _start(capsys, home)["run"]
    _, att, _ = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
                    "--model", "m", "--cwd", str(home), "--started-at", "2099-01-01T10:00:00-08:00", "--json")
    cli(capsys, home, "run", "attempt", "--run", run, "--end", att["attempt"], "--status", "done",
        "--ended-at", "2099-01-01T10:01:00-08:00")
    cli(capsys, home, "outcome", "--run", run, "--signal", "tests=pass", "--tier", "verified",
        "--observed-at", "2099-01-01T10:30:00-08:00")  # later than the clock
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit")
    assert code == 0, err
    doc = Store(home).run_doc(run)
    assert doc["run"]["ended_at"] == "2099-01-01T10:30:00-08:00"
    finished = next(e for e in doc["events"] if e["type"] == "run_finished")
    assert parse_ts(finished["at"]) == parse_ts(doc["run"]["ended_at"])
    shown = detail_from_doc(doc, doc["run"]["signals"], {"run": run})["signals"]
    assert [s.get("late") for s in shown] == [None]


def _finished_with_receipt(capsys, home):
    run = _start(capsys, home)["run"]
    cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex", "--model", "m",
        "--cwd", str(home))
    code, res, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--json")
    assert code == 0, err
    return run, res["receipt"]


def test_refit_stamps_fit_after_on_receipts_and_run_files(capsys, home):
    """Spec 01 section 2.8: after the refit, `after.fit_after` (and `moved` when the fit predicts the run)."""
    run, rct = _finished_with_receipt(capsys, home)

    def fake_fit(h, **opts):
        path = h / "fits" / "fit_20260923170000"
        path.mkdir(parents=True)
        (path / "meta.json").write_text("{}")
        return path

    assert run_job(home, fit_fn=fake_fit)["ran"] == 1
    receipt = json.loads((home / "receipts" / f"{rct}.json").read_text())
    assert receipt["after"]["fit_after"] == "fit_20260923170000" and receipt["fit_after_at"]
    assert Store(home).run_doc(run)["run"]["receipt"]["after"]["fit_after"] == "fit_20260923170000"
    assert fit_state(home)["job"]["receipts"]["stamped"] == 1
    assert run_job(home, fit_fn=lambda h, **o: h / "fits" / "fit_20260923180000")["ran"] == 1
    again = json.loads((home / "receipts" / f"{rct}.json").read_text())
    assert again["after"]["fit_after"] == "fit_20260923170000"  # stamped once, by the first fit after the run


def test_refit_records_how_the_prediction_moved(capsys, home, monkeypatch):
    run, rct = _finished_with_receipt(capsys, home)
    before = json.loads((home / "receipts" / f"{rct}.json").read_text())["before"]
    moved_to = SimpleNamespace(p_success=SimpleNamespace(mean=0.5), cost=SimpleNamespace(usd=SimpleNamespace(mean=2.0)))
    monkeypatch.setattr(finish_mod, "_prediction_after", lambda belief, doc: moved_to)
    out = finish_mod.stamp_fit_after(Store(home), "fit_x", belief=None)
    assert out == {"stamped": [rct], "moved": [rct], "skipped": []}
    moved = json.loads((home / "receipts" / f"{rct}.json").read_text())["after"]["moved"]
    assert moved == {"p_success": {"from": before["p_success"]["mean"], "to": 0.5},
                     "cost_usd": {"from": before["cost"]["usd"]["mean"], "to": 2.0}}
    assert Store(home).run_doc(run)["run"]["receipt"]["after"]["moved"] == moved


def test_prediction_after_reads_the_run_configuration(capsys, home):
    run, _ = _finished_with_receipt(capsys, home)
    seen = []
    belief = SimpleNamespace(predict=lambda task, cfg, rule: seen.append((task.type, cfg.id, rule.name)) or "p")
    assert finish_mod._prediction_after(belief, Store(home).run_doc(run)) == "p"
    assert seen == [("feature", EXPLORE_CFG, "tests")]


def _settle_by_attempt(costs):
    """Lane 2 stand-in: attempt i gets costs[i]; a cost without "usd" is tokens with no dollars (an unpriced model)."""

    def settle(doc):
        for a, c in zip(doc.get("attempts") or [], costs):
            a["cost"] = {"input_tokens": c["tokens"], "output_tokens": 0, "basis": "measured",
                         **({"usd": c["usd"]} if "usd" in c else {})}
        return doc, {"unmatched": {}}

    return settle


def test_token_only_attempt_is_unknown_dollars_not_zero(capsys, home, monkeypatch):
    """Review of 87cf8fc, finding 1: tokens without dollars never read as $0 in finish, index, receipt or budget."""
    monkeypatch.setattr(finish_mod, "default_settle", lambda: _settle_by_attempt([{"tokens": 1500}]))
    run = _start(capsys, home)["run"]
    cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex", "--model", "m",
        "--cwd", str(home))
    code, res, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--json")
    assert code == 0, err
    assert res["cost"] == {"usd": None, "usd_known": None, "tokens": 1500, "attempts_costed": 0,
                           "attempts_not_costed": 1}
    row = Store(home).index_rows()[run]
    assert row["cost_usd"] is None and row["attempts_not_costed"] == 1 and row["tokens"] == 1500
    receipt = json.loads((home / "receipts" / f"{res['receipt']}.json").read_text())
    assert receipt["after"]["cost"]["usd"] is None and receipt["scored"]["cost_in_interval"] is None
    assert Store(home).run_doc(run)["run"]["receipt"]["after"]["cost_usd"] is None
    code, b, err = cli(capsys, home, "budget", "--json")
    assert code == 0, err
    assert b["spent"]["usd"] == 0 and b["spent"]["attempts_not_costed"] == 1 and b["spent"]["runs_not_costed"] == 1
    code, text, err = cli(capsys, home, "budget")
    assert code == 0 and "1 attempt(s) in 1 run(s) have no dollars" in text
    code, rep, err = cli(capsys, home, "report", "--json")
    assert code == 0, err
    assert rep["exploration"]["runs"][0]["usd"] is None and rep["exploration"]["runs_not_costed"] == 1
    code, _, err = cli(capsys, home, "report", "--html", str(home / "r.html"))
    assert code == 0, err
    assert "unknown ($0.00 known)" in (home / "r.html").read_text()


def test_known_and_unknown_dollars_mixed(capsys, home, monkeypatch):
    monkeypatch.setattr(finish_mod, "default_settle",
                        lambda: _settle_by_attempt([{"tokens": 1000, "usd": 0.5}, {"tokens": 700}]))
    run = _start(capsys, home)["run"]
    for piece in ("implement", "review"):
        cli(capsys, home, "run", "attempt", "--run", run, "--piece", piece, "--harness", "codex", "--model", "m",
            "--cwd", str(home))
    code, res, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--json")
    assert code == 0, err
    assert res["cost"] == {"usd": None, "usd_known": 0.5, "tokens": 1700, "attempts_costed": 1,
                           "attempts_not_costed": 1}
    row = Store(home).index_rows()[run]
    assert row["cost_usd"] is None and row["cost_usd_known"] == 0.5 and row["attempts_not_costed"] == 1
    receipt = json.loads((home / "receipts" / f"{res['receipt']}.json").read_text())
    assert receipt["after"]["cost"] == {"usd": None, "tokens": 1700}  # the $0.50 lower bound is not the cost
    assert receipt["scored"]["cost_in_interval"] is None and receipt["scored"]["tokens_in_interval"] is not None
    assert Store(home).run_doc(run)["run"]["receipt"]["after"]["cost_usd"] is None
    code, b, err = cli(capsys, home, "budget", "--json")
    assert b["spent"]["usd"] == pytest.approx(0.5) and b["spent"]["attempts_not_costed"] == 1
    code, text, err = cli(capsys, home, "run", "finish", "--run", run)
    assert code == 1  # already finished; the text form of a fresh finish is checked below
    run2 = _start(capsys, home)["run"]
    for piece in ("implement", "review"):
        cli(capsys, home, "run", "attempt", "--run", run2, "--piece", piece, "--harness", "codex", "--model", "m",
            "--cwd", str(home))
    code, text, err = cli(capsys, home, "run", "finish", "--run", run2, "--no-fit")
    assert code == 0, err
    assert "1 attempt(s) without dollars; $0.50 known" in text


def _shared_settle(session):
    """Lane 2 stand-in for D87: attempts 1 and 2 name one session (summary `shared`); attempt 3 is a heuristic
    match; attempt 4 has no logmatch tier, so its basis decides."""

    def settle(doc):
        atts = doc["attempts"]
        pair = [atts[0]["id"], atts[1]["id"]]
        for a in atts[:2]:
            a["cost"] = {"input_tokens": 500, "output_tokens": 50, "usd": 0.25, "basis": "measured",
                         "ext": {"dev.loopmath.logmatch": {"tier": "verified", "session": session}}}
        atts[2]["cost"] = {"input_tokens": 100, "output_tokens": 10, "usd": 0.1, "basis": "allocated",
                           "ext": {"dev.loopmath.logmatch": {"tier": "heuristic"}}}
        atts[3]["cost"] = {"input_tokens": 100, "output_tokens": 10, "usd": 0.1, "basis": "measured"}
        return doc, {"unmatched": [], "shared": [{"session": session, "attempts": pair}]}

    return settle


def test_finish_names_a_session_that_two_attempts_share(capsys, home, monkeypatch):
    """D87: a sent-back round 2 resumes the session round 1 named; lane 2 reports it shared, finish says so."""
    session = "0a0b0c0d-1111-2222-3333-444455556666"
    monkeypatch.setattr(finish_mod, "default_settle", lambda: _shared_settle(session))

    def started():
        run = _start(capsys, home)["run"]
        for piece, rnd in (("implement", "1"), ("implement", "2"), ("review", "1"), ("review", "2")):
            code, _, err = cli(capsys, home, "run", "attempt", "--run", run, "--piece", piece, "--round", rnd,
                               "--harness", "claude-code", "--model", "m", "--cwd", str(home), "--json")
            assert code == 0, err
        return run, [a["id"] for a in Store(home).run_doc(run)["attempts"]]

    run, atts = started()
    code, res, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--json")
    assert code == 0, err
    assert res["matched"] == {"verified": 3, "heuristic": 1, "unmatched": []}
    assert res["shared"] == [{"session": session, "attempts": atts[:2], "split": False}]  # whole costs: lane 2's fallback
    run, atts = started()
    code, text, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit")
    assert code == 0, err
    assert "matched: 3 verified, 1 heuristic, 0 unmatched" in text
    assert (f"  session {session} is shared by attempts {atts[0]} and {atts[1]}; its cost is counted once in the run total"
            in text)  # D100: not "split" while the attempts keep whole costs
    from loopmath.store.commands import _finish_lines

    three = {**res, "shared": [{"session": "S", "attempts": ["A", "B", "C"], "split": True}]}
    assert "  session S is shared by attempts A, B and C; its cost is split" in _finish_lines(three)


def test_a_shared_session_is_split_only_when_its_attempts_carry_allocated_shares():
    """D100: lane 2's split marks the attempts `allocated` with `shared_session`; the fallback keeps them whole."""
    mark = {"session": "S", "attempts": ["a1", "a2"]}

    def doc(basis):
        return {"attempts": [{"id": f"a{i}", "cost": {"input_tokens": 5, "usd": 0.1, "basis": basis,
                                                       "ext": {"dev.loopmath.logmatch": {"tier": "verified", "shared_session": mark}}}}
                             for i in (1, 2)]}

    summary = {"shared": [mark]}
    assert finish_mod._shared_sessions(summary, doc("allocated")) == [{**mark, "split": True}]
    assert finish_mod._shared_sessions(summary, doc("measured")) == [{**mark, "split": False}]


def test_match_counts_reads_the_logmatch_tier_before_the_basis():
    """D88: lane 2's split of a shared verified session has basis `allocated` but tier verified."""
    def att(i, basis, tier=None):
        ext = {"ext": {"dev.loopmath.logmatch": {"tier": tier}}} if tier else {}
        return {"id": f"att_{i}", "cost": {"input_tokens": 10, "usd": 0.1, "basis": basis, **ext}}

    doc = {"attempts": [att(1, "allocated", "verified"), att(2, "allocated", "verified"), att(3, "allocated", "heuristic"),
                        att(4, "allocated"), att(5, "measured"), att(6, "reported")]}
    assert finish_mod.match_counts(doc) == {"verified": 3, "heuristic": 2, "unmatched": [], "reported": 1}


def test_finish_recomputes_when_a_signal_arrives_after_the_evidence(capsys, home, monkeypatch):
    """Review of 87cf8fc, finding 2: an open-run signal does not bump the rev, so finish must see it itself."""
    run = _start(capsys, home)["run"]
    cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex", "--model", "m",
        "--cwd", str(home))
    code, _, err = cli(capsys, home, "outcome", "--run", run, "--signal", "tests=pass", "--tier", "verified",
                       "--source", "ci", "--json")
    assert code == 0, err
    real = finish_mod.validate
    seen: list[list[str]] = []

    def racing_validate(doc):
        seen.append([str(s.get("value")) for s in doc["run"].get("signals") or []])
        if len(seen) == 1:  # evidence and receipt are computed; CI now reports a failure
            assert Store(home).add_signal(run, Signal(id=new_id("sig"), run=run, kind="verdict", name="tests",
                                                      value="fail", tier="verified", source={"kind": "ci"},
                                                      observed_at=now_iso())) == "pending"
        return real(doc)

    monkeypatch.setattr(finish_mod, "validate", racing_validate)
    res = finish_mod.finish_run(Store(home), run, no_fit=True)
    assert seen == [["pass"], ["pass", "fail"]]  # the new signal went through validation before the write
    doc = Store(home).run_doc(run)
    assert [s["value"] for s in doc["run"]["signals"]] == ["pass", "fail"]
    assert res["evidence"]["z"] == 0.0
    receipt = json.loads((home / "receipts" / f"{res['receipt']}.json").read_text())
    assert receipt["after"]["z"] == 0.0
    inline = doc["run"]["receipt"]["after"]["signals"]
    assert inline == receipt["after"]["signals"] and len(inline) == 2
    assert [e["detail"] for e in doc["events"] if e["type"] == "receipt_written"] == [res["receipt"]]
    assert not (home / "signals" / f"{run}.jsonl").exists()
    assert len(Store(home).receipts_for(run)) == 1


def test_finish_gives_up_after_bounded_retries(tmp_path):
    store = Store(tmp_path / "lm")
    run = store.new_run(task(), usual_config(), source="usual", rec=None, slate=None, rule=DEFAULT_RULE, base_commit=None)
    store.add_attempt(run, {"piece": "implement", "harness": "codex", "model": "m", "cwd": "/tmp"})
    passes = []

    def always_moving(doc):
        passes.append(1)
        store.add_attempt(run, {"piece": "review", "harness": "codex", "model": "m", "cwd": "/tmp"})
        return fake_settle()(doc)

    with pytest.raises(Conflict):
        finish_mod.finish_run(store, run, settle=always_moving, no_fit=True)
    assert len(passes) == finish_mod.FINISH_TRIES
    assert not R.is_finished(store.run_doc(run))


def _fail_after_commit(monkeypatch, capsys, home, *, via_outcome=True):
    """Review of d351d8e: a failing verdict lands the moment `Store.finish` returns, before `finish_run` does."""
    real = Store.finish
    states = []

    def finish_then_signal(self, run, doc, **kw):
        real(self, run, doc, **kw)
        if via_outcome:  # the real `outcome` path: late signal, flag, re-score
            code, out, err = cli(capsys, home, "outcome", "--run", run, "--signal", "tests=fail", "--tier", "verified",
                                 "--source", "ci", "--json")
            assert code == 0, err
            states.append(out["state"])
        else:  # an `outcome` interrupted after the signal was written: the receipt is only flagged
            states.append(self.add_signal(run, Signal(id=new_id("sig"), run=run, kind="verdict", name="tests",
                                                      value="fail", tier="verified", observed_at=now_iso())))

    monkeypatch.setattr(Store, "finish", finish_then_signal)
    return states


def _pass_run(capsys, home):
    run = _start(capsys, home)["run"]
    cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex", "--model", "m",
        "--cwd", str(home))
    code, _, err = cli(capsys, home, "outcome", "--run", run, "--signal", "tests=pass", "--tier", "verified",
                       "--source", "ci", "--json")
    assert code == 0, err
    return run


def test_a_late_signal_right_after_the_commit_rescores_the_receipt(capsys, home, monkeypatch):
    run = _pass_run(capsys, home)
    states = _fail_after_commit(monkeypatch, capsys, home)
    res = finish_mod.finish_run(Store(home), run, no_fit=True)
    assert states == ["late"] and res["evidence"]["z"] == 1.0  # finish reports the run as it committed it
    doc = Store(home).run_doc(run)
    receipt = json.loads((home / "receipts" / f"{res['receipt']}.json").read_text())
    assert receipt["after"]["z"] == 0.0 and not receipt["rescore"] and receipt["rescored_at"]
    assert receipt["after"]["signals"] == [s["id"] for s in doc["run"]["signals"]] and len(doc["run"]["signals"]) == 2
    assert doc["run"]["receipt"]["after"]["signals"] == receipt["after"]["signals"]


def test_an_interrupted_rescore_is_reconciled_by_the_background_job(capsys, home, monkeypatch):
    run = _pass_run(capsys, home)
    states = _fail_after_commit(monkeypatch, capsys, home, via_outcome=False)
    res = finish_mod.finish_run(Store(home), run, no_fit=True)
    assert states == ["late"]
    receipt = json.loads((home / "receipts" / f"{res['receipt']}.json").read_text())
    assert receipt["rescore"] is True and receipt["after"]["z"] == 1.0  # flagged, never overwritten
    code, out, err = cli(capsys, home, "status", "--json")
    assert "receipts_rescore" in {h["code"] for h in out["health"]}
    assert run_job(home, fit_fn=lambda h, **o: None)["ran"] == 1
    receipt = json.loads((home / "receipts" / f"{res['receipt']}.json").read_text())
    assert receipt["rescore"] is False and receipt["after"]["z"] == 0.0 and len(receipt["after"]["signals"]) == 2
    assert Store(home).run_doc(run)["run"]["receipt"]["after"]["signals"] == receipt["after"]["signals"]
    assert fit_state(home)["job"]["rescored"] == {"rescored": 1, "failed": []}
    assert finish_mod.rescore_run(Store(home), run)["rescored"] == []  # nothing flagged: a second pass is a no-op
    assert json.loads((home / "receipts" / f"{res['receipt']}.json").read_text()) == receipt
    code, out, err = cli(capsys, home, "status", "--json")
    assert "receipts_rescore" not in {h["code"] for h in out["health"]}


def test_a_rescore_after_the_refit_keeps_fit_after(capsys, home):
    run, rct = _finished_with_receipt(capsys, home)

    def fake_fit(h, **opts):
        path = h / "fits" / "fit_20260923170000"
        path.mkdir(parents=True)
        return path

    assert run_job(home, fit_fn=fake_fit)["ran"] == 1
    code, out, err = cli(capsys, home, "outcome", "--run", run, "--signal", "tests=fail", "--tier", "verified",
                         "--source", "ci", "--json")
    assert code == 0 and out["state"] == "late" and out["late"]["rescored"] == [rct]
    receipt = json.loads((home / "receipts" / f"{rct}.json").read_text())
    assert receipt["after"]["fit_after"] == "fit_20260923170000" and receipt["after"]["z"] == 0.0
    inline = Store(home).run_doc(run)["run"]["receipt"]["after"]
    assert inline["fit_after"] == "fit_20260923170000" and inline["signals"] == receipt["after"]["signals"]


def test_finish_settles_with_lane_2_on_real_session_logs(capsys, home, monkeypatch):
    """Post-merge: `run finish` calls lane 2's `settle_run` itself, here over lane 2's fixture session logs."""
    monkeypatch.setattr(finish_mod, "default_settle", REAL_SETTLE)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(LOGMATCH_FIXTURES / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(LOGMATCH_FIXTURES / "codex"))
    run = _start(capsys, home)["run"]
    code, att, err = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "claude-code",
                         "--model", "claude-opus-5-5", "--session", CC_SESSION, "--json")
    assert code == 0, err
    code, res, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--json")
    assert code == 0, err
    assert res["matched"] == {"verified": 1, "heuristic": 0, "unmatched": []}
    assert res["validation"]["errors"] == [] and res["validation"]["checker"] == "loopmath.ocp"
    doc = Store(home).run_doc(run)
    attempt = doc["attempts"][0]
    assert attempt["id"] == att["attempt"] and attempt["cost"]["basis"] == "measured"
    assert attempt["cost"]["ext"]["dev.loopmath.logmatch"]["tier"] == "verified"
    assert len(attempt["cost"]["ext"]["dev.loopmath.logmatch"]["children"]) == 7  # the session's sub-agents
    cost = R.run_cost(doc)
    assert res["cost"]["tokens"] == cost["tokens"] == 162 + 3_689_023 + 641_706 + 89_793  # lane 2's fixture totals
    assert res["cost"]["usd"] == cost["usd"] and cost["usd"] > 0
    assert Store(home).index_rows()[run]["tokens"] == cost["tokens"]
    receipt = json.loads((home / "receipts" / f"{res['receipt']}.json").read_text())
    assert receipt["after"]["cost"]["tokens"] == cost["tokens"]


def test_ocp_configuration_with_rec_gets_its_receipt_through_lane_4_and_6(capsys, home, tmp_path):
    """A configuration read as an OCP object (a file or a stored run) is turned back into one by lane 4."""
    first = _start(capsys, home)
    cfg = Store(home).run_doc(first["run"])["run"]["configuration"]
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({k: v for k, v in cfg.items() if k not in ("source", "rec")}))
    code, out, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "loopmath/loopmath",
                         "--config", str(path), "--rec", REC_ID, "--source", "exploration", "--json")
    assert code == 0, err
    assert out["config"] == EXPLORE_CFG and out["receipt"]["edited"] is False
    receipt = json.loads((home / "receipts" / f"{out['receipt']['receipt']}.json").read_text())
    assert receipt["before"]["config"] == EXPLORE_CFG and receipt["rec"] == REC_ID


def test_outcome_q_config_reaches_lane_5(capsys, home):
    """`outcome.q.<tier>` in config.toml is passed to lane 5's outcome function."""
    code, _, err = cli(capsys, home, "config", "set", "outcome.q.verified", "0.9")
    assert code == 0, err
    run = _pass_run(capsys, home)
    code, res, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--json")
    assert code == 0, err
    assert res["evidence"]["z"] == 1.0 and res["evidence"]["tier"] == "verified" and res["evidence"]["q"] == 0.9
