"""0.2.4 store fixes: small costs, UTC times, spend (labelling, onboard history), background refit
exclusions, store receipt checks, artifact and outcome summaries. Synthetic stores only."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from loopmath import cli as loopmath_cli
from loopmath.output import fmt_time, fmt_usd
from loopmath.store import Store
from loopmath.store import finish as finish_mod
from loopmath.store import fitjob
from loopmath.store.budget import spend
from loopmath.store.home import StoreError
from loopmath.store.ids import now_iso, utc_iso

from store_helpers import EXPLORE_CFG, REC_ID, cli, fake_settle, finished_doc, install_rec


def test_small_costs_print_with_one_significant_digit_and_zero_as_dollar_zero():
    assert [fmt_usd(x) for x in (0, 0.0, 0.0016, 0.00004, 0.0049, 0.0096, 0.01, 2.74, 1234.5)] == [
        "$0", "$0", "$0.002", "$0.00004", "$0.005", "$0.01", "$0.01", "$2.74", "$1,234.50"]
    assert fmt_usd(None) == "n/a" and fmt_usd(True) == "n/a" and fmt_usd(-0.002) == "-$0.002"
    assert all(fmt_usd(x) not in ("$0.00", "$0") for x in (1e-9, 0.001, 0.004999, 0.0099))
    # rounding up to the next power of ten keeps one significant digit
    assert [fmt_usd(x) for x in (0.00096, 0.000999, 0.000096, 0.0000096)] == ["$0.001", "$0.001", "$0.0001", "$0.00001"]


def test_store_runs_are_oldest_first_by_instant_across_offsets(tmp_path):
    store = Store(tmp_path / "home")  # an upgraded store holds local-offset and UTC strings side by side
    for rid, started in (("run_late", "2026-09-27T20:00:00-07:00"), ("run_early", "2026-09-28T01:00:00Z"),
                         ("run_first", "2026-09-27T17:59:00-07:00")):
        store.import_run(finished_doc(rid, usd=0.01, started=started))
    assert [r["run"] for r in store.runs()] == ["run_first", "run_early", "run_late"]


def test_times_are_stored_in_utc_and_shown_in_local_time_with_the_zone():
    stamp = now_iso()
    assert stamp.endswith("Z") and len(stamp) == 20
    assert utc_iso("2026-09-27T19:56:09-07:00") == "2026-09-28T02:56:09Z"
    assert utc_iso("2026-09-28T02:56:09.250000+00:00") == "2026-09-28T02:56:09.250Z"
    assert utc_iso(None) is None and utc_iso("not a time") == "not a time"
    local = datetime(2026, 9, 27, 19, 56, tzinfo=timezone.utc).astimezone()
    zone = local.strftime("%Z") or local.strftime("%z")
    assert fmt_time("2026-09-27T19:56:00Z") == local.strftime("%Y-%m-%d %H:%M") + f" {zone}"
    assert fmt_time(None) == "n/a" and fmt_time("garbage") == "garbage"


def _now_stamp() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def test_status_shows_the_period_spend_with_labelling_and_onboard_history_on_its_own_line(tmp_path, capsys):
    store = Store(tmp_path / "lm")
    stamp = _now_stamp()
    store.import_run(finished_doc("run_u1", source="usual", usd=2.0, started=stamp))
    store.import_run(finished_doc("run_h1", source="habit", usd=50.0, started=stamp))  # written by onboard
    store.import_run(finished_doc("run_c1", source="habit", usd=0.059, started=stamp, history=False))  # a loop run
    spd = store.add_spend("labeling", usd=0.0016, tokens=900, detail={"labeler": "claude:haiku", "calls": 3})
    assert spd.startswith("spd_") and store.spend_records()[0]["at"].endswith("Z")
    s = spend(store, "month")
    assert s["usd"] == pytest.approx(2.0 + 0.059 + 0.0016) and s["runs"] == 2
    assert s["labeling"]["usd"] == pytest.approx(0.0016) and s["labeling"]["calls"] == 3
    assert s["history_runs"] == 1 and s["history_usd"] == pytest.approx(50.0)
    code, out, err = cli(capsys, store.home, "status")
    assert code == 0, err
    lines = out.splitlines()
    spent = next(i for i, ln in enumerate(lines) if ln.startswith("spend"))
    assert "$2.06" in lines[spent]
    assert "onboard labelling $0.002" in lines[spent + 1] and "runs $2.06 over 2 runs" in lines[spent + 1]
    assert lines[spent + 2].strip().startswith("history from onboard") and "$50.00 over 1 run" in lines[spent + 2]
    code, out, err = cli(capsys, store.home, "budget", "--json")
    assert out["spent"]["labeling"]["usd"] == pytest.approx(0.0016) and out["spent"]["since"].endswith("Z")
    with pytest.raises(StoreError):
        store.add_spend("labeling", usd=-1)


def test_a_loop_run_with_source_habit_counts_in_spend_and_report(tmp_path, capsys):
    store = Store(tmp_path / "lm")
    stamp = _now_stamp()
    store.import_run(finished_doc("run_h1", source="habit", usd=5.0, started=stamp))
    store.import_run(finished_doc("run_c1", source="habit", usd=0.5, started=stamp, history=False))
    rows = store.index_rows()
    assert rows["run_h1"]["history"] is True and rows["run_c1"]["history"] is False
    code, rep, err = cli(capsys, store.home, "report", "--json")
    assert code == 0, err
    assert rep["runs"]["history"] == 1 and rep["cost_per_accepted"]["by_source"]["habit"]["runs"] == 1
    code, text, err = cli(capsys, store.home, "report")
    assert "1 of them from onboard history" in text


def test_an_old_index_row_without_the_flag_reads_the_run(tmp_path):
    store = Store(tmp_path / "lm")
    stamp = _now_stamp()
    store.import_run(finished_doc("run_h1", source="habit", usd=5.0, started=stamp))
    store.import_run(finished_doc("run_c1", source="habit", usd=0.5, started=stamp, history=False))
    lines = [json.loads(x) for x in store.index_path.read_text().splitlines()]
    for row in lines:
        row.pop("history", None)  # as 0.2.3 wrote them
    store.index_path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    s = spend(store, "month")
    assert s["history_runs"] == 1 and s["runs"] == 1 and s["usd"] == pytest.approx(0.5)


def _fit(home, fit_id, options):
    folder = home / "fits" / fit_id
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"fit": fit_id, "options": options}), encoding="utf-8")
    link = home / "fits" / "latest"
    if link.is_symlink():
        link.unlink()
    os.symlink(fit_id, link)


def test_background_refits_keep_the_last_fits_exclusions_and_status_says_so(tmp_path, capsys, monkeypatch):
    home = tmp_path / "lm"
    Store(home).ensure()
    _fit(home, "fit_20260928070000", {"no_prior": False, "without": ["shared"], "full": False})
    assert fitjob.exclusions(home) == {"no_prior": False, "without": ["shared"], "fit": "fit_20260928070000"}
    seen = []

    def fake_fit(h, **opts):
        seen.append(opts)
        _fit(h, "fit_20260928070100", {"no_prior": opts["no_prior"], "without": opts["without"], "full": opts["full"]})
        return h / "fits" / "fit_20260928070100"

    assert fitjob.run_job(home, fit_fn=fake_fit, inherit=True)["ran"] == 1
    assert seen == [{"no_prior": False, "without": ["shared"], "full": False}]
    job = json.loads((home / "fits" / "job.json").read_text())
    assert job["opts"]["as_fit"] == "fit_20260928070000" and job["opts"]["without"] == ["shared"]
    assert job["started_at"].endswith("Z")
    cmds = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            cmds.append(cmd)
            self.pid = 4242

    monkeypatch.setattr(fitjob.subprocess, "Popen", FakePopen)
    assert fitjob.spawn_fit(home)["started"] is True  # a store refit: decided when it runs
    assert fitjob.spawn_fit(home, without=[])["started"] is True  # fit --background: its own flags
    assert "--as-last-fit" in cmds[0] and "--as-last-fit" not in cmds[1]
    code, out, err = cli(capsys, home, "status")
    assert code == 0, err
    assert "sources left out of fits: shared" in out and "background refits keep this" in out
    code, out, err = cli(capsys, home, "status", "--json")
    assert out["fit"]["background_exclusions"]["without"] == ["shared"]


def _receipt(store, run, usd, lo, hi, *, after=True, inside=None):
    rc = {"id": f"rct_{run[4:]}", "rec": None, "run": run, "fit": "fit_1",
          "before": {"config": "cfg_aaaaaaaaaaaa", "p_success": {"mean": 0.7, "lo": 0.6, "hi": 0.8, "level": 0.8},
                     "cost": {"usd": {"mean": (lo + hi) / 2, "lo": lo, "hi": hi, "level": 0.8},
                              "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}},
                     "ell": {"usd": {"mean": 1, "lo": 1, "hi": 1, "level": 0.8},
                             "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}},
                     "rounds": {"mean": 1, "lo": 1, "hi": 1, "level": 0.8}, "per_piece": {}}}
    if after:
        rc["after"] = {"cost": {"usd": usd, "tokens": 110}, "z": 1.0, "q": 0.95}
        rc["scored"] = {"cost_in_interval": (lo <= usd <= hi) if inside is None else inside}
    store.write_receipt(rc)
    return rc["id"]


def test_verify_receipts_checks_the_store_without_an_argument(tmp_path, capsys, monkeypatch):
    home = tmp_path / "lm"
    store = Store(home)
    stamp = _now_stamp()
    store.import_run(finished_doc("run_a1", usd=1.0, started=stamp))
    store.import_run(finished_doc("run_a2", usd=3.0, started=stamp, open_=True), finished=False)
    _receipt(store, "run_a1", 1.0, 0.5, 2.5)
    _receipt(store, "run_a2", None, 0.5, 2.5, after=False)
    monkeypatch.chdir(tmp_path)  # no e2-receipts.jsonl here: the store is what is checked
    code = loopmath_cli.main(["verify-receipts", "--home", str(home)])
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("OK: 2 receipt(s)") and "1 scored, 1 waiting" in out
    bad = _receipt(store, "run_a1", 1.0, 1.5, 2.5, inside=True)  # 1.0 is outside 1.5 to 2.5
    (home / "receipts" / "rct_zz.json").write_text(json.dumps({"id": "rct_other", "run": "run_a1"}))
    code = loopmath_cli.main(["verify-receipts", "--home", str(home)])
    out = capsys.readouterr().out
    assert code == 1 and out.startswith("FAIL: 2 of 3 receipt(s)")
    assert "rct_zz: its id 'rct_other' is not its file name" in out and f"{bad}: cost_in_interval is True" in out
    code = loopmath_cli.main(["verify-receipts", "e2-receipts.jsonl"])  # a research ledger: named explicitly
    assert code == 1 and "cannot read ledger" in capsys.readouterr().out


def test_verify_receipts_passes_receipts_that_run_start_and_run_finish_wrote(tmp_path, capsys, monkeypatch):
    home = tmp_path / "lm"
    install_rec(home)
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle(usd=0.0016))
    for n in range(2):
        code, out, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "acme/web", "--config",
                             EXPLORE_CFG, "--rec", REC_ID, "--source", "exploration", "--json")
        assert code == 0, err
        if n == 0:
            cli(capsys, home, "run", "attempt", "--run", out["run"], "--piece", "implement", "--harness", "codex",
                "--model", "m", "--cwd", str(tmp_path))
            code, fin, err = cli(capsys, home, "run", "finish", "--run", out["run"], "--no-fit", "--json")
            assert code == 0 and fin["receipt"], err
    code = loopmath_cli.main(["verify-receipts", "--home", str(home)])
    out = capsys.readouterr().out
    assert code == 0 and "OK: 2 receipt(s)" in out and "1 scored, 1 waiting" in out, out


def test_verify_receipts_on_an_empty_store(tmp_path, capsys):
    code = loopmath_cli.main(["verify-receipts", "--home", str(tmp_path / "none")])
    assert code == 0 and "OK: no receipts yet" in capsys.readouterr().out


def test_run_artifact_and_outcome_print_a_one_line_summary(tmp_path, capsys):
    home = tmp_path / "lm"
    store = Store(home)
    store.import_run(finished_doc("run_o1", started=_now_stamp(), open_=True), finished=False)
    code, _, err = cli(capsys, home, "run", "attempt", "--run", "run_o1", "--piece", "n", "--harness", "codex",
                       "--model", "m", "--cwd", str(tmp_path))
    assert code == 0, err
    att = [a["id"] for a in store.run_doc("run_o1")["attempts"] if a["id"] != "att_run_o1"][0]
    code, out, err = cli(capsys, home, "run", "artifact", "--run", "run_o1", "--kind", "commit",
                         "--path", "d0296af1234567890abcdef", "--by", att)
    assert code == 0, err
    line = out.strip()
    assert line.startswith("art_") and line.endswith(f": commit d0296af123 on run run_o1, by {att}")
    code, out, err = cli(capsys, home, "outcome", "--run", "run_o1", "--signal", "tests=pass", "--tier", "verified")
    assert code == 0, err
    assert out.strip().startswith("sig_") and out.strip().endswith(": verdict tests=pass (verified) on run run_o1")
    code, out, err = cli(capsys, home, "outcome", "--run", "run_o1", "--signal", "perf=", "--kind", "score")
    assert out.strip().endswith(": score perf=not measured yet (reported) on run run_o1")
