"""The runs view (spec 06 section 1): the fixture page, and `loopmath runs` on a store of v0.3 runs (lane 12)."""

from __future__ import annotations

import copy
import json
import re
import time
from datetime import datetime, timedelta

import pytest

from loopmath.cli import main
from loopmath.store import ids
from loopmath.views import common, runs

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")
T = "2026-09-2{}T{}:00-07:00"

CLICK_EVERY_ROW = """(async () => {
  const out = { rows: document.querySelectorAll('tr.row').length, opened: 0, graphs: 0, timelines: 0, raw: 0 };
  for (const run of [...document.querySelectorAll('tr.row')].map(tr => tr.dataset.run)) {
    document.querySelector('tr.row[data-run="' + CSS.escape(run) + '"]').click();
    const d = document.getElementById('detail');
    if (!d.hidden) out.opened++;
    if (d.querySelector('#d-wf svg')) out.graphs++;
    if (d.querySelector('#d-rg svg')) out.timelines++;
    const raw = d.querySelector('#d-raw');
    if (raw) { raw.open = true; raw.dispatchEvent(new Event('toggle')); if (raw.querySelector('pre').textContent.length > 2) out.raw++; }
    for (const el of d.querySelectorAll('[data-n], [data-gate], [data-att]')) el.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 5, clientY: 5 }));
  }
  for (const [f, v] of [['outcome', 'accepted'], ['slate', '(any)'], ['source', 'habit']]) {
    const el = document.querySelector('[data-f="' + f + '"]'); el.value = v; el.dispatchEvent(new Event('change', { bubbles: true }));
    out['filter_' + f] = document.querySelectorAll('tr.row').length;
    document.querySelector('[data-reset]').click();
  }
  out.lede = document.getElementById('lede').textContent;
  return out;
})()"""


# ---------------------------------------------------------------- the fixture
def test_fixture_page_round_trips(fixture_json):
    data = fixture_json("view-runs.json")
    page = runs.render(data)
    assert common.extract_data(page) == data
    assert not re.search(r"https?://", page)
    assert chr(0x2014) not in page  # no em dashes


def test_fixture_page_runs_without_errors(fixture_json, probe, tmp_path):
    data = fixture_json("view-runs.json")
    path = tmp_path / "runs.html"
    path.write_text(runs.render(data), encoding="utf-8")
    out = probe(path, CLICK_EVERY_ROW)
    assert out["exceptions"] == [] and out["console_errors"] == [] and out["network"] == []
    r = out["result"]
    assert r["rows"] == 4 and r["opened"] == 4
    with_detail = {data["selected"]["run"]} | set(data.get("details") or {})  # D12: details for every row
    assert r["graphs"] == len(with_detail)
    assert r["filter_outcome"] == 2 and r["filter_slate"] == 2 and r["filter_source"] == 1


def test_fixture_without_receipts_or_signals(fixture_json, probe, tmp_path):
    data = fixture_json("view-runs.json")
    for row in data["runs"]:
        row["receipt"] = {"predicted": None, "cost_in_interval": None, "surprise": None}
        row["z"] = row["q"] = row["tier"] = row["score"] = None
    data["selected"]["signals"] = []
    path = tmp_path / "bare.html"
    path.write_text(runs.render(data), encoding="utf-8")
    out = probe(path, CLICK_EVERY_ROW)
    assert out["exceptions"] == [] and out["console_errors"] == []
    assert "None had a prediction on record" in out["result"]["lede"]


def test_empty_view_renders(probe, tmp_path):
    data = {"schema": runs.SCHEMA, "generated_at": "2026-09-23T12:00:00-07:00", "filters": {}, "runs": [], "selected": None}
    path = tmp_path / "empty.html"
    path.write_text(runs.render(data), encoding="utf-8")
    out = probe(path, "document.getElementById('lede').textContent")
    assert out["exceptions"] == [] and out["result"].startswith("No runs recorded yet")


# ---------------------------------------------------------------- a store of v0.3 runs
@pytest.fixture
def store(tmp_path, v03):
    s = v03.signal
    tests_pass = s("sig_a_tests", "verdict", "tests", "pass", T.format(0, "11:05"))
    docs = [
        # usual run, two repair rounds, a receipt in receipts/
        v03.run_doc("run_a", attempts=[
            v03.attempt("a1", "implement", 1, T.format(0, "10:00"), T.format(0, "10:10"), 1.0),
            v03.attempt("a2", "review", 1, T.format(0, "10:10"), T.format(0, "10:20"), 0.5, result="reject"),
            v03.attempt("a3", "implement", 2, T.format(0, "10:20"), T.format(0, "10:30"), 0.8),
            v03.attempt("a4", "review", 2, T.format(0, "10:30"), T.format(0, "10:40"), 0.4, result="accept"),
        ], signals=[tests_pass], artifacts=[{"id": "d1", "path": "x.diff", "vertex": "diff", "version": 1},
                                            {"id": "d2", "path": "x.diff", "vertex": "diff", "version": 2, "supersedes": "d1"}]),
        # a pair in one slate, with a preference
        *[v03.run_doc(rid, started=T.format(1, hm), ended=T.format(1, "15:00"), source=src,
                      slate={"id": "slt_1", "members": ["run_b", "run_c"], "base_commit": "abc", "isolated": True, "blinded": True},
                      preferences=[{"id": "prf_1", "slate": "slt_1", "winner": "run_b", "members": ["run_b", "run_c"],
                                    "judge": {"kind": "referee", "model": "gpt-6-astra", "blinded": True},
                                    "observed_at": T.format(1, "16:00"), "tier": "reported"}],
                      signals=[s(f"sig_{rid}", "verdict", "tests", verdict, T.format(1, "14:50"))])
          for rid, hm, src, verdict in (("run_b", "14:00", "alternative", "pass"), ("run_c", "14:01", "exploration", "fail"))],
        # a logged habit run: no rule, no signals, OCP receipt absent
        v03.run_doc("run_d", started=T.format(2, "09:00"), ended=T.format(2, "09:30"), source=None, task_type="bug_fix",
                    repo="acme/web", workflow=v03.SOLO, provenance={"kind": "logged", "chooser": "habit"}),
        # an open run with no attempts yet
        v03.run_doc("run_e", started=T.format(3, "08:00"), ended=None, attempts=[]),
        # accepted by tests, then reverted inside the window (a late event, appended in signals/)
        v03.run_doc("run_f", started=T.format(0, "12:00"), ended=T.format(0, "13:00"),
                    signals=[s("sig_f_tests", "verdict", "tests", "pass", T.format(0, "12:55"))]),
        # a score rule, reached
        v03.run_doc("run_g", started=T.format(2, "10:00"), ended=T.format(2, "11:00"), workflow=v03.SOLO,
                    rule={"name": "perf>=2400", "definition": "perf at least 2400", "requires": [],
                          "score": {"name": "heldout_perf", "target": 2400, "better": "higher", "scale": "linear"}},
                    signals=[s("sig_g", "score", "heldout_perf", 2500, T.format(2, "10:59"), unit="perf", better="higher")],
                    receipt={"before": {"rec": "rec_1", "predicted": {"p_success": {"mean": 0.6, "lo": 0.4, "hi": 0.8},
                                                                       "cost_usd": {"mean": 1.0, "lo": 0.5, "hi": 2.0},
                                                                       "tokens": {"mean": 20000, "lo": 10000, "hi": 40000}}}}),
    ]
    receipt = {"id": "rct_1", "rec": "rec_0", "run": "run_a", "fit": "fit_1",
               "before": {"config": docs[0]["run"]["configuration"]["id"], "p_success": {"mean": 0.7, "lo": 0.6, "hi": 0.8, "level": 0.8},
                          "cost": {"usd": {"mean": 2.0, "lo": 1.5, "hi": 3.0, "level": 0.8},
                                   "tokens": {"mean": 40000, "lo": 30000, "hi": 60000, "level": 0.8}},
                          "rounds": {"mean": 1.5, "lo": 1.0, "hi": 2.0, "level": 0.8}},
               "scored": {"cost_in_interval": True, "surprise": 0.2}}
    late = [v03.signal("sig_f_revert", "event", "revert", "commit 123", "2026-09-22T09:00:00-07:00", "reported")]
    return v03.write_store(tmp_path / "home", docs, receipts=[receipt], appended={"run_f": late})


def _json(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr()
    return code, (json.loads(captured.out) if "--json" in argv and captured.out.strip() else None), captured


def test_a_row_without_an_outcome_blanks_tier_and_q(tmp_path, v03):
    # D63: Evidence always has a tier (asserted, q 0.7, when nothing was used); a row whose z is null shows none.
    s = v03.signal
    docs = [v03.run_doc("run_none"),
            v03.run_doc("run_err", signals=[s("sig_e", "verdict", "tests", "error", "2026-09-20T10:59:00-07:00", "reported")])]
    for doc in docs:
        ev = runs.outcome(doc, doc["run"]["signals"], NOW)  # lane 5's outcome_evidence
        assert (ev["z"], ev["tier"], ev["q"]) == (None, "asserted", 0.7), doc["run"]["id"]
    root = v03.write_store(tmp_path / "home", docs)
    for row in runs.build_view(root, now=NOW)["runs"]:
        assert (row["z"], row["tier"], row["q"]) == (None, None, None), row["run"]


def test_rows_from_v03_documents(store, v03):
    data = runs.build_view(store, now=NOW)
    rows = {r["run"]: r for r in data["runs"]}
    assert list(data)[0] == "schema" and data["schema"] == "loopmath.view.runs/1"
    assert [r["run"] for r in data["runs"]][0] == "run_e"  # newest first
    a = rows["run_a"]
    assert a["rounds"] == 2 and a["cost"] == {"usd": 2.7, "tokens": 86000}
    assert (a["z"], a["q"], a["tier"], a["source"], a["state"]) == (1.0, 0.98, "verified", "usual", "finished")
    ir_id = v03.config_id(v03.WORKFLOW, v03.default_settings(v03.WORKFLOW))
    assert re.fullmatch(r"cfg_[0-9a-f]{12}", ir_id)
    assert a["config"] == {"id": ir_id, "label": "implement_review: gpt-6-astra/xhigh, gpt-6-astra/xhigh",
                           "workflow": "implement_review"}
    assert a["receipt"]["cost_in_interval"] is True and a["receipt"]["surprise"] == 0.2
    assert a["receipt"]["predicted"]["p_success"]["mean"] == 0.7
    assert rows["run_b"]["slate"] == rows["run_c"]["slate"] == "slt_1"
    assert rows["run_b"]["preference"]["winner"] == "run_b" and rows["run_c"]["z"] == 0.0
    d = rows["run_d"]
    # D7: no verdict is inferred for a habit run, so z is unknown; D63: the row then blanks tier and q,
    # though Evidence says asserted, 0.7.
    assert d["source"] == "habit" and d["z"] is None and d["tier"] is None and d["q"] is None
    assert d["receipt"]["predicted"] is None
    assert rows["run_e"]["state"] == "open" and rows["run_e"]["cost"] == {"usd": None, "tokens": None}
    assert rows["run_e"]["rounds"] is None
    assert rows["run_f"]["z"] == 0.0 and rows["run_f"]["tier"] == "reported"  # the late revert, appended
    g = rows["run_g"]
    assert g["z"] == 1.0 and g["score"] == 2500 and g["score_meta"]["unit"] == "perf" and g["score_meta"]["target"] == 2400
    assert g["receipt"]["cost_in_interval"] is True and g["receipt"]["predicted"]["cost"]["usd"]["mean"] == 1.0
    assert "notes" not in data  # every file read, and outcomes come from lane 5, so nothing to note


def test_json_command_and_filters(store, capsys):
    code, out, _ = _json(capsys, ["runs", "--home", str(store), "--json"])
    assert code == 0 and list(out)[0] == "schema" and len(out["runs"]) == 7 and "details" not in out
    _, out, _ = _json(capsys, ["runs", "--home", str(store), "--json", "--type", "bug_fix"])
    assert [r["run"] for r in out["runs"]] == ["run_d"] and out["filters"] == {"type": "bug_fix"}
    _, out, _ = _json(capsys, ["runs", "--home", str(store), "--json", "--repo", "acme/web"])
    assert [r["run"] for r in out["runs"]] == ["run_d"]
    _, out, _ = _json(capsys, ["runs", "--home", str(store), "--json", "--slate", "slt_1"])
    assert sorted(r["run"] for r in out["runs"]) == ["run_b", "run_c"]
    _, out, _ = _json(capsys, ["runs", "--home", str(store), "--json", "--since", "2026-09-22"])
    assert sorted(r["run"] for r in out["runs"]) == ["run_d", "run_e", "run_g"]
    code, _, captured = _json(capsys, ["runs", "--home", str(store), "--since", "last tuesday"])
    assert code == 1 and "--since" in captured.err


def test_since_is_read_by_the_store_and_minutes_or_months_is_refused(store, capsys):
    # D109: the store's one --since reader; `m` could be minutes or months, so it exits 2 and lists nothing
    assert runs.parse_since is ids.parse_since
    for ambiguous in ("3m", "90M", "1.5 m"):
        for extra in ([], ["--json"]):
            code, _, captured = _json(capsys, ["runs", "--home", str(store), *extra, "--since", ambiguous])
            assert code == 2 and "ambiguous" in captured.err and captured.out == ""
    data = runs.build_view(store, {"since": "36H"}, now=NOW)  # any case and a decimal, as the store reads it
    assert data["filters"]["since_at"] == (NOW - timedelta(hours=36)).isoformat(timespec="seconds")
    assert runs.build_view(store, {"since": "1.5d"}, now=NOW)["filters"]["since_at"] == data["filters"]["since_at"]


def test_one_run(store, capsys):
    code, out, _ = _json(capsys, ["runs", "--home", str(store), "--json", "--run", "run_f"])
    assert code == 0 and [r["run"] for r in out["runs"]] == ["run_f"]
    sel = out["selected"]
    assert sel["run"] == "run_f" and sel["doc"]["run"]["id"] == "run_f" and sel["graph"]["nodes"]
    assert [s["id"] for s in sel["signals"]] == ["sig_f_tests", "sig_f_revert"]
    assert sel["signals"][1]["late"] is True and "late" not in sel["signals"][0]
    code, _, captured = _json(capsys, ["runs", "--home", str(store), "--run", "run_a"])
    lines = captured.out.splitlines()
    assert code == 0 and lines[0].startswith("run run_a") and len(lines) <= 25
    assert any("predicted  success 70% (60% to 80%)" in line for line in lines)
    code, _, captured = _json(capsys, ["runs", "--home", str(store), "--run", "run_zzz"])
    assert code == 2 and "run_zzz" in captured.err


def test_html_paths_and_streams(store, capsys, tmp_path):
    target = tmp_path / "out" / "r.html"
    code = main(["runs", "--home", str(store), "--html", str(target)])
    captured = capsys.readouterr()
    assert code == 0 and captured.out.strip() == str(target)
    data = common.extract_data(target.read_text(encoding="utf-8"))
    assert sorted(data["details"]) == sorted(r["run"] for r in data["runs"])
    assert data["details"]["run_a"]["workflow"]["gates"][0]["results"][0]["value"] == "reject"
    code = main(["runs", "--home", str(store), "--html", str(target), "--json"])
    captured = capsys.readouterr()
    assert code == 0 and captured.err.strip() == str(target) and json.loads(captured.out)["schema"] == runs.SCHEMA
    code = main(["runs", "--home", str(store), "--html"])
    captured = capsys.readouterr()
    written = captured.out.strip()
    assert code == 0 and re.search(r"/views/runs-\d{8}-\d{6}\.html$", written) and written.startswith(str(store))


def test_store_page_runs_without_errors(store, probe, tmp_path):
    path = tmp_path / "store.html"
    path.write_text(runs.render(runs.build_view(store, details=True, now=NOW)), encoding="utf-8")
    out = probe(path, CLICK_EVERY_ROW)
    assert out["exceptions"] == [] and out["console_errors"] == [] and out["network"] == []
    r = out["result"]
    assert r["rows"] == 7 and r["opened"] == 7 and r["graphs"] == 7 and r["raw"] == 7
    assert r["timelines"] == 6  # the open run has no attempts, so no attempt timeline


def test_details_past_the_cap_show_the_command(store, probe, tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "DETAIL_CAP", 2)
    data = runs.build_view(store, details=True, now=NOW)
    assert len(data["details"]) == 2
    path = tmp_path / "capped.html"
    path.write_text(runs.render(data), encoding="utf-8")
    out = probe(path, """(() => { document.querySelector('tr.row[data-run="run_a"]').click();
      return document.querySelector('#detail .cmd code').textContent; })()""")
    assert out["exceptions"] == [] and out["result"] == "loopmath runs --run run_a --html"


def test_empty_store(tmp_path, capsys):
    code = main(["runs", "--home", str(tmp_path / "nothing")])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0 and lines[0] == "No runs recorded yet."
    assert "loopmath run import FILE.ocp.json" in lines[1] and not (tmp_path / "nothing").exists()


def test_filters_that_match_nothing_are_named(store, capsys):
    code = main(["runs", "--home", str(store), "--type", "nope", "--repo", "x"])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0 and lines == ["No runs match --type nope, --repo x.",
                                   "Run `loopmath runs` without filters to see every recorded run."]
    data = runs.build_view(store, {"since": "1d"}, now=datetime.fromisoformat("2030-01-01T12:00:00-07:00"))
    assert runs.summary_lines(data)[0] == "No runs match --since 1d (since 2029-12-31 12:00)."


def test_the_table_shows_whole_run_ids_that_run_accepts(tmp_path, v03, capsys):
    long_id = "run_" + "x" * 40
    v03.write_store(tmp_path / "home", [v03.run_doc(long_id), v03.run_doc("run_b", started="2026-09-01T10:00:00-07:00")])
    main(["runs", "--home", str(tmp_path / "home")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split()[:2] == ["run", "started"]
    ids = [line.split()[0] for line in lines[2:]]
    assert sorted(ids) == sorted([long_id, "run_b"])
    code = main(["runs", "--home", str(tmp_path / "home"), "--run", ids[0]])
    assert code == 0 and capsys.readouterr().out.startswith(f"run {ids[0]}")


def test_page_names_command_line_filters_that_match_nothing(store, probe, tmp_path):
    path = tmp_path / "none.html"
    path.write_text(runs.render(runs.build_view(store, {"type": "nope", "since": "90d"}, now=NOW)), encoding="utf-8")
    out = probe(path, "document.getElementById('lede').textContent")
    assert out["exceptions"] == [] and out["result"].startswith("No runs match --type nope, --since 90d from the command line.")


def test_summary_is_at_most_25_lines(tmp_path, v03, capsys):
    docs = [v03.run_doc(f"run_{i:03d}", started=f"2026-09-{1 + i % 20:02d}T10:{i % 60:02d}:00-07:00") for i in range(60)]
    v03.write_store(tmp_path / "home", docs)
    main(["runs", "--home", str(tmp_path / "home")])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) <= 25 and lines[0].startswith("60 runs") and any(line.startswith("... ") for line in lines)


def test_five_hundred_runs(tmp_path, v03, probe):
    base = v03.run_doc("run_x", attempts=[
        v03.attempt("x1", "implement", 1, T.format(0, "10:00"), T.format(0, "10:10"), 1.0),
        v03.attempt("x2", "review", 1, T.format(0, "10:10"), T.format(0, "10:20"), 0.5, result="accept")],
        signals=[v03.signal("sig_x", "verdict", "tests", "pass", T.format(0, "10:30"))])
    docs = []
    for i in range(500):
        doc = copy.deepcopy(base)
        doc["run"]["id"] = f"run_{i:03d}"
        doc["run"]["started_at"] = f"2026-08-{1 + i % 28:02d}T{i % 24:02d}:00:00-07:00"
        doc["run"]["signals"][0]["id"] = f"sig_{i:03d}"
        docs.append(doc)
    home = v03.write_store(tmp_path / "home", docs)
    t0 = time.perf_counter()
    data = runs.build_view(home, details=True, now=NOW)
    page = runs.render(data)
    build_s = time.perf_counter() - t0
    assert len(data["runs"]) == 500 and len(data["details"]) == runs.DETAIL_CAP
    path = tmp_path / "500.html"
    path.write_text(page, encoding="utf-8")
    out = probe(path, """(async () => { const t0 = performance.now(); document.querySelector('tr.row').click();
      return { rows: document.querySelectorAll('tr.row').length, click_ms: performance.now() - t0,
               nav_ms: performance.getEntriesByType('navigation')[0].loadEventEnd }; })()""")
    assert out["exceptions"] == []
    assert out["result"]["rows"] == 500
    assert out["result"]["nav_ms"] < 2000 and out["result"]["click_ms"] < 500, (out, build_s)


ROW_ORDER = "[...document.querySelectorAll('tr.row[data-run]')].map(tr => tr.dataset.run)"


@pytest.mark.parametrize("since", ["90d", "2w", "2026-09-19"])
def test_since_is_applied_once_by_the_builder(store, probe, tmp_path, since):
    data = runs.build_view(store, {"since": since}, details=True, now=NOW)
    assert data["runs"] and data["filters"]["since"] == since and "since_at" in data["filters"]
    page = tmp_path / "since.html"
    page.write_text(runs.render(data), encoding="utf-8")
    out = probe(page, ROW_ORDER)
    assert out["exceptions"] == []
    assert out["result"] == [r["run"] for r in data["runs"]]  # every row the builder kept is shown


def test_fixture_with_a_relative_since_shows_every_row(fixture_json, probe, tmp_path):
    data = fixture_json("view-runs.json")
    data["filters"] = {"since": "90d"}
    page = tmp_path / "since-fixture.html"
    page.write_text(runs.render(data), encoding="utf-8")
    assert len(probe(page, ROW_ORDER)["result"]) == len(data["runs"])


def test_since_through_the_command_matches_its_json(store, capsys, tmp_path, probe):
    page = tmp_path / "cmd-since.html"
    code, out, _ = _json(capsys, ["runs", "--home", str(store), "--since", "36500d", "--json", "--html", str(page)])
    assert code == 0 and out["runs"]
    assert probe(page, ROW_ORDER)["result"] == [r["run"] for r in out["runs"]]


def test_mixed_utc_offsets_order_by_instant(tmp_path, v03, probe):
    s = v03.signal
    rule = {"name": "perf>=20", "definition": "perf at least 20", "requires": [],
            "score": {"name": "perf", "target": 20, "better": "higher", "scale": "linear"}}
    # 17:00Z comes before 11:00-07:00 (18:00Z), though its text sorts after it
    sigs = [s("sig_late", "score", "perf", 30, "2026-09-20T11:00:00-07:00"),
            s("sig_early", "score", "perf", 10, "2026-09-20T17:00:00Z")]
    docs = [v03.run_doc("run_x", started="2026-09-20T10:00:00-07:00", ended="2026-09-20T11:30:00-07:00", rule=rule, signals=sigs),
            v03.run_doc("run_y", started="2026-09-20T12:00:00+00:00", ended="2026-09-20T12:30:00+00:00")]
    root = v03.write_store(tmp_path / "home", docs)
    data = runs.build_view(root, details=True, now=NOW)
    x = next(r for r in data["runs"] if r["run"] == "run_x")
    assert x["score"] == 30 and x["z"] == 1.0
    assert [r["run"] for r in data["runs"]] == ["run_x", "run_y"]  # 17:00Z is newer than 12:00Z
    assert [g["id"] for g in data["details"]["run_x"]["signals"]] == ["sig_early", "sig_late"]
    signal_ids = [sig["id"] for sig in common.run_signals(docs[0])]
    assert signal_ids == ["sig_early", "sig_late"]
    page = tmp_path / "offsets.html"
    page.write_text(runs.render(data), encoding="utf-8")
    out = probe(page, ROW_ORDER)
    assert out["exceptions"] == [] and out["result"] == ["run_x", "run_y"]



SID = "00000000-0000-4000-8000-00000000c0de"


def _claude_session(root, turns):
    """One Claude Code session log with the given (time, request, input, output) turns."""
    path = root / "projects" / "-work-repo" / f"{SID}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({
        "type": "assistant", "sessionId": SID, "timestamp": f"2026-09-20T{ts}Z", "cwd": "/work/repo", "requestId": req,
        "message": {"role": "assistant", "model": "claude-opus-5-5", "content": [],
                    "usage": {"input_tokens": tin, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": tout}},
    }) + "\n" for ts, req, tin, tout in turns), encoding="utf-8")


def test_a_session_named_whole_and_as_self_counts_once_in_the_run_and_the_totals(tmp_path, v03, capsys):
    """D101, D103: the run's cost is the store's `run_cost` over lane 2's split, not a sum of whole sessions."""
    from loopmath.logmatch.match import LogRoots
    from loopmath.logmatch.settle import settle_attempt, settle_run

    _claude_session(tmp_path, [("17:00:00", "r0", 100_000, 1_000), ("17:10:00", "r1", 1_000, 50), ("17:20:00", "r2", 70_000, 700)])
    roots = LogRoots(claude=[tmp_path / "projects"])
    start, end = "2026-09-20T17:05:00Z", "2026-09-20T17:15:00Z"

    def att(aid, **extra):
        a = v03.attempt(aid, "impl", 1, start, end, 0.0)
        a.pop("cost")
        return dict(a, harness="claude-code", session=SID, **extra)

    whole_only = settle_attempt(att("w"), roots=roots)[0]["cost"]
    self_ext = {"ext": {"dev.loopmath.match": {"session_from": "self"}}}
    settled, _ = settle_run({"ocp": "0.3", "run": {"id": "run_split"}, "attempts": [att("run_split.a1"), att("run_split.a2", **self_ext)]}, roots=roots)
    assert settled["attempts"][0]["cost"]["basis"] == "allocated"  # lane 2 split the session
    v03.write_store(tmp_path / "home", [v03.run_doc("run_split", attempts=settled["attempts"])])
    row = runs.build_view(tmp_path / "home", now=NOW)["runs"][0]
    tokens = sum(whole_only[f] for f in ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens"))
    assert row["cost"]["usd"] == pytest.approx(whole_only["usd"], abs=1e-6) and row["cost"]["tokens"] == tokens  # once, not twice
    main(["runs", "--home", str(tmp_path / "home")])
    assert capsys.readouterr().out.splitlines()[0].startswith(f"1 runs, {common.fmt_money(row['cost']['usd'], tokens)}")


def test_a_run_with_an_unpriced_attempt_has_unknown_dollars_as_the_store_says(tmp_path, v03):
    """D101: one definition of a run's cost. An unpriced attempt makes the dollars unknown, never a partial sum."""
    from loopmath.store.runs import run_cost

    doc = v03.run_doc("run_part", attempts=[v03.attempt("run_part.a1", "impl", 1, "2026-09-20T10:00:00-07:00", "2026-09-20T10:30:00-07:00", 1.25),
                                            v03.attempt("run_part.a2", "impl", 2, "2026-09-20T10:30:00-07:00", "2026-09-20T11:00:00-07:00", 0.0)])
    doc["attempts"][1]["cost"]["usd"] = None  # tokens but no price (an unpriced model)
    v03.write_store(tmp_path / "home", [doc])
    row = runs.build_view(tmp_path / "home", now=NOW)["runs"][0]
    store = run_cost(doc)
    assert row["cost"] == {"usd": store["usd"], "tokens": store["tokens"]} and row["cost"]["usd"] is None


def test_a_predicted_cost_above_its_interval_keeps_its_mean_with_the_note(tmp_path, v03, capsys, probe):
    """D107: in the terminal and on the page; the embedded JSON keeps the numbers as they are."""
    note = "the average is pulled up by rare very large outcomes"
    doc = v03.run_doc("run_t")
    receipt = {"id": "rct_t", "rec": "rec_t", "run": "run_t", "fit": "fit_t",
               "before": {"config": doc["run"]["configuration"]["id"], "p_success": {"mean": 0.7, "lo": 0.6, "hi": 0.8, "level": 0.8},
                          "cost": {"usd": {"mean": 4.0, "lo": 0.5, "hi": 3.0, "level": 0.8},
                                   "tokens": {"mean": 40000, "lo": 30000, "hi": 60000, "level": 0.8}}}}
    home = v03.write_store(tmp_path / "home", [doc], receipts=[receipt])
    main(["runs", "--home", str(home), "--run", "run_t"])
    out = capsys.readouterr().out
    assert f"cost $4.00 ($0.50 to $3.00); {note}" in out and "success 70% (60% to 80%)," in out  # only where it applies
    path = tmp_path / "t.html"
    data = runs.build_view(home, run="run_t", details=True, now=NOW)
    path.write_text(runs.render(data), encoding="utf-8")
    assert common.extract_data(path.read_text(encoding="utf-8")) == json.loads(json.dumps(data))
    got = probe(path, "[...document.querySelectorAll('#detail .tail')].map(e => e.textContent)")  # --run opens its detail
    assert got["exceptions"] == [] and got["result"] == [note]
