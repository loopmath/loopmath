"""The harness2 reader: synthetic harness2 runs folders (tests/harness_runs/h2fixture.py, written from
harness2's code with file:line) become `loopmath.harness-runs/0.1` rows. harness2 itself is never imported."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "harness_runs"))
import h2fixture as F  # noqa: E402

from loopmath import harness, harness_runs as HR  # noqa: E402
from loopmath.adapters import harness2 as H2  # noqa: E402

H2NS = "dev.loopmath.adapter.harness2"


def _by_run(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for row in rows:
        rid = row["tool"]["run_id"]
        assert rid not in out, f"{rid} imported twice"
        out[rid] = row
    return out


@pytest.fixture
def jb(tmp_path) -> F.Built:
    return F.build(tmp_path / "runs")


def test_no_harness2_module_is_imported():
    assert not [m for m in sys.modules if m == "harness2" or m.startswith("harness2.")]


def test_every_live_rollout_is_one_checked_row_and_bases_once(jb):
    got = H2.read(jb.root)
    assert got.format == "harness2/method-2"
    rows = _by_run(got.rows)
    assert set(rows) == {r.run_id for r in jb.rollouts}
    for row in got.rows:
        assert HR.check_row(row) == [], row
        assert row["schema"] == HR.SCHEMA
    for rid in rows:   # a base listed in both modes' ledgers is still one row
        assert sum(1 for r in got.rows if r["tool"]["run_id"] == rid) == 1


@pytest.mark.parametrize("root_of", ["runs", "bench", "adapt"])
def test_any_level_of_the_runs_folder_reads_the_same(jb, root_of):
    path = {"runs": jb.root, "bench": jb.root / "jb", "adapt": jb.adapt}[root_of]
    assert _by_run(H2.read(path).rows).keys() == {r.run_id for r in jb.rollouts}


def test_a_folder_with_no_harness2_runs_is_refused_by_name(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(H2.FormatError, match="no harness2"):
        H2.read(tmp_path / "empty")


def test_arms_tool_ids_and_stages(jb):
    rows = _by_run(H2.read(jb.root).rows)
    for r in jb.rollouts:
        row = rows[r.run_id]
        assert row["task_id"] == r.task_id
        assert row["arm"] == r.arm
        tool = row["tool"]
        assert tool["name"] == "harness2" and tool["version"] == "0.1.0"
        assert (tool["run_id"], tool["task_id"], tool["stage"]) == (r.run_id, r.task_id, r.label)
    adopted = {(r.task_id, r.mode) for r in jb.rollouts if r.arm == "adopted"}
    assert adopted == {(t, m) for t in ("finance/task1", "finance/task2") for m in ("agg", "seq")}


def test_records_carry_digest_parent_and_fingerprint(jb):
    rows = _by_run(H2.read(jb.root).rows)
    for r in jb.rollouts:
        rec = rows[r.run_id]["harness"]
        assert rec["schema"] == harness.SCHEMA and rec["source"] == "harness2"
        assert rec["digest"] == harness.digest(r.tree, layout="harness2")
        if r.parent is None:
            assert rec.get("parent") is None
        else:
            assert rec["parent"] == harness.digest(r.parent, layout="harness2")
        assert rec["tool"]["fingerprint"] == F.fingerprint(r.tree)
        assert rec["tool"]["fingerprint_match"] is True


def test_a_tree_changed_after_its_rollout_gets_no_record(tmp_path):
    """The marker's fingerprint says which tree ran (core/runmark.py:64-81); a scored rollout whose tree was
    edited later must not take the new tree's factors."""
    b = F.build(tmp_path / "runs")
    adopted = next(r for r in b.rollouts if r.arm == "adopted" and r.score is not None)
    (adopted.tree / ".claude").mkdir(exist_ok=True)
    (adopted.tree / ".claude" / "settings.json").write_text('{"enabledPlugins": {"late@acme": true}}\n',
                                                            encoding="utf-8")
    got = H2.read(b.root)
    rows = _by_run(got.rows)
    row = rows[adopted.run_id]
    assert "harness" not in row
    assert row["ext"][H2.NS]["tree_changed"] is True
    assert row["outcome"]["score"] == pytest.approx(adopted.score)       # the score still stands
    assert got.counts["fingerprint_mismatch"] == 1
    assert any(adopted.run_id in n and "changed after the rollout ran" in n for n in got.notes)
    others = [r for r in b.rollouts if r.run_id != adopted.run_id]
    assert all("harness" in rows[r.run_id] and "tree_changed" not in rows[r.run_id]["ext"][H2.NS] for r in others)
    doc = HR.to_ocp(row)
    assert harness.EXT_KEY not in doc["run"].get("ext", {})


def test_a_parent_tree_changed_after_it_ran_leaves_its_children_without_a_parent(tmp_path):
    b = F.build(tmp_path / "runs")
    c1 = next(r for r in b.rollouts if r.run_id == "jb-oc-seq2/finance/task1/c1")
    (c1.tree / "NOTES.md").write_text("edited after the round\n", encoding="utf-8")
    got = H2.read(b.root)
    rows = _by_run(got.rows)
    assert "harness" not in rows[c1.run_id]
    for label in ("inv2", "dom2", "c2"):
        row = rows[f"jb-oc-seq2/finance/task1/{label}"]
        assert row["harness"].get("parent") is None and row["ext"][H2.NS]["parent_changed"] is True
        assert row["harness"]["tool"]["fingerprint_match"] is True
    assert rows["jb-oc-seq2/finance/task1/inv1"]["harness"]["parent"]     # round 1 edits base, untouched


def test_a_sequential_round_edits_the_previous_compose(jb):
    rows = _by_run(H2.read(jb.root).rows)
    c1 = rows["jb-oc-seq2/finance/task1/c1"]["harness"]["digest"]
    assert rows["jb-oc-seq2/finance/task1/inv2"]["harness"]["parent"] == c1
    assert rows["jb-oc-seq2/finance/task1/c2"]["harness"]["parent"] == c1
    base = rows["jb-oc-agg2/finance/task1/base1"]["harness"]["digest"]
    assert rows["jb-oc-agg2/finance/task1/agg2"]["harness"]["parent"] == base


def test_scores_come_from_the_judge_files_and_name_the_judge(jb):
    rows = _by_run(H2.read(jb.root).rows)
    for r in jb.rollouts:
        out = rows[r.run_id].get("outcome")
        if not r.judged:
            assert out is None, r.run_id
            continue
        assert out["score"] == pytest.approx(r.score)
        assert out["judge"] == "judge-model-x" and out["name"] == "judge" and out["scale"] == "fraction"
    scored = [r for r in jb.rollouts if r.score is not None]
    assert {r.arm for r in scored} == {"base", "adopted"}


def test_a_failed_judge_stays_unscored_never_zero(tmp_path):
    b = F.build(tmp_path / "runs", variants={"judge-failed", "base-unscored"})
    rows = _by_run(H2.read(b.root).rows)
    failed = b.by_label("finance/task1", "agg", "agg1")
    assert rows[failed.run_id]["outcome"]["score"] is None
    assert rows[failed.run_id]["outcome"]["judge"] == "judge-model-x"
    base2 = b.by_label("finance/task1", "agg", "base2")
    assert rows[base2.run_id]["outcome"]["score"] is None


def test_no_judge_files_means_no_outcomes(tmp_path):
    b = F.build(tmp_path / "runs", variants={"no-judge"})
    assert all(row.get("outcome") is None for row in H2.read(b.root).rows)


def test_opencode_cost_is_recovered_from_the_trajectory(jb):
    rows = _by_run(H2.read(jb.root).rows)
    for r in jb.rollouts:
        cost = rows[r.run_id]["cost"]
        assert cost["recovered"] is True and cost["usd_source"] == "opencode"
        assert cost["usd"] == pytest.approx(r.usd)
        assert cost["tokens"] == r.tokens


def test_rollout_usd_is_never_read(jb):
    for summary in jb.adapt.rglob("summary.json"):
        doc = json.loads(summary.read_text())
        for mode in doc["modes"].values():
            mode["ledger"]["rollout_usd"] = 99.0
            mode["ledger"]["usd"] = 99.0
        summary.write_text(json.dumps(doc))
    rows = _by_run(H2.read(jb.root).rows)
    assert all(rows[r.run_id]["cost"]["usd"] == pytest.approx(r.usd) for r in jb.rollouts)


def test_every_retry_file_counts(tmp_path):
    b = F.build(tmp_path / "runs", variants={"retry"})
    r = b.by_label("finance/task1", "agg", "base1")
    cost = _by_run(H2.read(b.root).rows)[r.run_id]["cost"]
    assert cost["files"] == 2 and cost["tokens"] == r.tokens and cost["usd"] == pytest.approx(r.usd)


def test_zero_step_costs_leave_usd_for_pricing(tmp_path):
    b = F.build(tmp_path / "runs", variants={"zero-cost"})
    for row in H2.read(b.root).rows:
        assert row["cost"]["usd"] is None and row["cost"]["tokens"]["input"] > 0


def test_lines_that_do_not_parse_are_counted_not_fatal(tmp_path):
    b = F.build(tmp_path / "runs", variants={"garbage-lines"})
    rows = _by_run(H2.read(b.root).rows)
    for r in b.rollouts:
        assert rows[r.run_id]["cost"]["tokens"] == r.tokens
        assert rows[r.run_id]["cost"]["skipped_lines"] == 2


@pytest.mark.parametrize("bench", ["jb", "lab"])
def test_codex_usage_is_summed_and_left_for_pricing(tmp_path, bench):
    b = F.build(tmp_path / "runs", bench=bench, substrate="cx")
    rows = _by_run(H2.read(b.root).rows)
    for r in b.rollouts:
        cost = rows[r.run_id]["cost"]
        assert cost["tokens"] == r.tokens and cost["recovered"] is True
        assert cost["usd"] is None
        assert rows[r.run_id]["agent"] == "codex"


def test_model_and_effort(tmp_path):
    jb = F.build(tmp_path / "jb", executor="gemini-3.5-flash")
    assert {row["model"] for row in H2.read(jb.root).rows} == {"gemini-3.5-flash"}
    assert {row["agent"] for row in H2.read(jb.root).rows} == {"opencode"}
    lab = F.build(tmp_path / "lab", bench="lab", substrate="cx", executor="gpt-5.4")
    rows = H2.read(lab.root).rows
    assert {(row["model"], row["effort"]) for row in rows} == {("gpt-5.4", "medium")}


def test_no_config_reads_with_model_unknown(tmp_path):
    b = F.build(tmp_path / "runs", variants={"no-config"})
    got = H2.read(b.root)
    assert got.format == "harness2/method-2"
    assert {row.get("model") for row in got.rows} == {None}


def test_editor_and_selector_spend_stay_out_of_the_rollout_cost(jb):
    rows = _by_run(H2.read(jb.root).rows)
    inv1 = rows["jb-oc-agg2/finance/task1/inv1"]
    assert inv1["ext"][H2NS]["editor_usd"] == pytest.approx(0.035)
    adopted = rows["jb-oc-agg2/finance/task1/agg1"]
    assert adopted["ext"][H2NS]["selector_usd"] == pytest.approx(0.012)
    assert inv1["cost"]["usd"] != pytest.approx(0.035)


def test_unknown_method_version_is_refused_by_name(tmp_path):
    b = F.build(tmp_path / "runs", method_version=3)
    with pytest.raises(H2.FormatError, match="method_version 3"):
        H2.read(b.root)


def test_unknown_marker_version_is_refused_by_name(jb):
    mark = next(jb.root.rglob(".adapt_complete"))
    doc = json.loads(mark.read_text())
    doc["version"] = 2
    mark.write_text(json.dumps(doc))
    with pytest.raises(H2.FormatError, match=r"\.adapt_complete version 2"):
        H2.read(jb.root)


def test_a_summary_missing_keys_is_refused_naming_them(jb):
    summary = next(jb.adapt.rglob("summary.json"))
    doc = json.loads(summary.read_text())
    del doc["selections"], doc["base"]
    summary.write_text(json.dumps(doc))
    with pytest.raises(H2.FormatError, match=r"base.*selections|selections.*base"):
        H2.read(jb.root)


def test_legacy_arm_keys_read_as_harness2_reads_them(tmp_path):
    new = H2.read(F.build(tmp_path / "new").root)
    old = H2.read(F.build(tmp_path / "old", variants={"legacy-arms"}).root)
    assert "legacy" in old.format
    assert {r["tool"]["run_id"]: r["arm"] for r in old.rows} == {r["tool"]["run_id"]: r["arm"] for r in new.rows}


def test_bare_label_bases_import_without_cost_or_score(tmp_path):
    b = F.build(tmp_path / "runs", variants={"bare-base"})
    got = H2.read(b.root)
    rows = _by_run(got.rows)
    base = rows[b.by_label("finance/task1", "agg", "base1").run_id]
    assert base.get("outcome") is None
    assert got.counts["bare_base"] == 4


def test_base_failed_has_only_bases(tmp_path):
    b = F.build(tmp_path / "runs", variants={"base-failed"})
    rows = H2.read(b.root).rows
    assert {row["arm"] for row in rows} == {"base"} and len(rows) == 4


def test_a_crashed_mode_is_skipped_and_counted(tmp_path):
    b = F.build(tmp_path / "runs", variants={"mode-crashed"})
    got = H2.read(b.root)
    assert not any("-seq2/" in row["tool"]["run_id"] for row in got.rows)
    assert got.counts["mode_failed"] == 2
    assert any("seq" in s["reason"] for s in got.skipped)


def test_a_fallback_selection_adopts_nothing(tmp_path):
    b = F.build(tmp_path / "runs", variants={"fallback"})
    rows = H2.read(b.root).rows
    assert not [r for r in rows if r["arm"] == "adopted" and "-seq2/" in r["tool"]["run_id"]]
    assert len([r for r in rows if r["arm"] == "adopted"]) == 2


def test_a_dead_rollout_is_skipped_with_its_reason(tmp_path):
    b = F.build(tmp_path / "runs", variants={"dead-rollout"})
    got = H2.read(b.root)
    dead = [r.run_id for r in b.rollouts if r.dead]
    assert dead and not set(dead) & {row["tool"]["run_id"] for row in got.rows}
    assert {s["run_id"] for s in got.skipped} >= set(dead)
    assert all(s["reason"].startswith("unavailable") for s in got.skipped)
    assert got.counts["unavailable"] == len(dead) and got.counts["failed"] == 0


@pytest.mark.parametrize("substrate", ["oc", "cx"])
def test_a_rollout_with_no_marker_is_a_failed_row_with_incomplete_cost(tmp_path, substrate):
    """A known failed execution stays in, with its unknown cost kept unknown (spec/harness-runs.md section 2)."""
    b = F.build(tmp_path / "runs", substrate=substrate, variants={"failed-rollout"})
    got = H2.read(b.root)
    rows = _by_run(got.rows)
    failed = [r for r in b.rollouts if r.failed]
    assert len(failed) == 1 and got.counts["failed"] == 1
    row = rows[failed[0].run_id]
    assert row["status"] == "failed"
    cost = row["cost"]
    assert cost["complete"] is False and cost["usd"] is None
    assert ("1 step started" if substrate == "oc" else "1 turn started") in cost["incomplete"]
    assert cost.get("partial_usd") == failed[0].partial_usd
    doc = HR.to_ocp(row)
    att = doc["attempts"][0]
    assert att["status"] == "failed" and att["outcome"]["result"] == "failed"
    assert "cost" not in att
    marks = att["ext"][HR.NS]["cost"]
    assert marks["complete"] is False and marks["partial_tokens"] == failed[0].tokens
    others = [row for rid, row in rows.items() if rid != failed[0].run_id]
    assert all(row["status"] == "done" and row["cost"].get("complete", True) for row in others)


def test_two_experiments_with_the_same_native_ids_never_collide(tmp_path):
    """harness2's run ids leave out the executor (core/schedule.py:111-127), so two roots share them."""
    a = F.build(tmp_path / "a" / "runs", executor="model-a")
    b = F.build(tmp_path / "b" / "runs", executor="model-b")
    assert {r.run_id for r in a.rollouts} == {r.run_id for r in b.rollouts}
    ra, rb = H2.read(a.root), H2.read(b.root)
    exp_a, exp_b = {row["experiment"] for row in ra.rows}, {row["experiment"] for row in rb.rows}
    assert len(exp_a) == len(exp_b) == 1 and exp_a != exp_b
    ids_a, ids_b = {HR.row_id(row) for row in ra.rows}, {HR.row_id(row) for row in rb.rows}
    assert len(ids_a) == len(ra.rows) == 28 and not ids_a & ids_b
    runs_a, runs_b = {HR.run_id_for(row) for row in ra.rows}, {HR.run_id_for(row) for row in rb.rows}
    assert len(runs_a | runs_b) == 56


def test_moving_a_root_keeps_its_experiment_and_row_ids(jb, tmp_path):
    before = H2.read(jb.root)
    moved = tmp_path / "moved" / "runs"
    moved.parent.mkdir()
    shutil.move(str(jb.root), str(moved))
    after = H2.read(moved)
    assert [HR.row_id(r) for r in after.rows] == [HR.row_id(r) for r in before.rows]
    assert after.experiments == before.experiments


def test_an_experiment_label_tells_identical_repeats_apart(tmp_path):
    one = F.build(tmp_path / "one" / "runs")
    two = F.build(tmp_path / "two" / "runs")
    assert {r["experiment"] for r in H2.read(one.root).rows} == {r["experiment"] for r in H2.read(two.root).rows}
    rep1 = {r["experiment"] for r in H2.read(one.root, experiment="rep1").rows}
    rep2 = {r["experiment"] for r in H2.read(two.root, experiment="rep2").rows}
    assert rep1 != rep2 and all(e.startswith("rep1-h2-") for e in rep1)
    assert {r["ext"][H2NS]["experiment_label"] for r in H2.read(one.root, experiment="rep1").rows} == {"rep1"}
    with pytest.raises(ValueError, match="--experiment"):
        H2.read(one.root, experiment="/abs/path")


def test_a_stale_judge_row_scores_nothing(tmp_path):
    """The summary now adopts agg2; judged.json still scores agg1 from the earlier selection."""
    b = F.build(tmp_path / "runs", variants={"stale-judge"})
    got = H2.read(b.root)
    rows = _by_run(got.rows)
    agg2, agg1 = b.by_label("finance/task1", "agg", "agg2"), b.by_label("finance/task1", "agg", "agg1")
    assert agg2.arm == "adopted" and agg1.arm == "candidate"
    assert rows[agg2.run_id].get("outcome") is None and rows[agg1.run_id].get("outcome") is None
    assert "stale judge row" in rows[agg2.run_id]["ext"][H2NS]["unscored_reason"]
    assert got.counts["stale_judge"] == 1
    assert any("stale judge row" in note for note in got.notes)


def test_a_base_judge_row_must_name_the_same_run(jb):
    path = next(jb.adapt.rglob("base_judged.json"))
    rows = json.loads(path.read_text())
    rows[0]["run_id"] = rows[0]["run_id"] + "-earlier"
    path.write_text(json.dumps(rows))
    got = _by_run(H2.read(jb.root).rows)
    assert got[rows[0]["run_id"][: -len("-earlier")]].get("outcome") is None
    assert sum(1 for row in got.values() if row["arm"] == "base" and row.get("outcome")) == 3


@pytest.mark.parametrize("field,value", [("label", "agg2"), ("k", 3), ("task_id", "finance/task2"), ("mode", "seq")])
def test_a_judge_row_must_agree_on_label_task_mode_and_k(jb, field, value):
    path = next(jb.adapt.rglob("judged.json"))
    rows = json.loads(path.read_text())
    target = next(r for r in rows if r["task_id"] == "finance/task1" and r["mode"] == "agg")
    target[field] = value
    path.write_text(json.dumps(rows))
    got = _by_run(H2.read(jb.root).rows)
    assert got[target["run_id"]].get("outcome") is None
    assert got[target["run_id"]]["ext"][H2NS]["unscored_reason"]


@pytest.mark.parametrize("executor", ["gemini-3.5-flash", "gpt-5.4"])
def test_a_partial_reported_cost_never_becomes_the_rollout_usd(tmp_path, executor):
    """One step reports $0.0125, the other reports none; the whole token total is priced or unknown."""
    b = F.build(tmp_path / "runs", executor=executor, variants={"mixed-cost"})
    for row in H2.read(b.root).rows:
        cost = row["cost"]
        assert cost["usd"] is None and cost["partial_usd"] == pytest.approx(0.0125)
        assert cost.get("complete", True) is True
        att = HR.to_ocp(row)["attempts"][0]
        marks = att["ext"][HR.NS]["cost"]
        assert marks["partial_usd"] == pytest.approx(0.0125)
        priced, reason = HR.price_tokens(executor, cost["tokens"])
        if priced is None:
            assert "usd" not in att["cost"] and "no price" in marks["unpriced"]
        else:
            assert att["cost"]["usd"] == pytest.approx(round(priced, 6))
            assert att["cost"]["usd"] != pytest.approx(0.0125)
            assert marks["usd_source"] == "loopmath-prices"


def test_a_moved_runs_folder_is_re_anchored(jb, tmp_path):
    before = _by_run(H2.read(jb.root).rows)
    moved = tmp_path / "elsewhere" / "runs"
    moved.parent.mkdir()
    shutil.move(str(jb.root), str(moved))
    after = _by_run(H2.read(moved).rows)
    assert {k: (v["cost"], v.get("outcome"), v["harness"]["digest"]) for k, v in after.items()} == \
           {k: (v["cost"], v.get("outcome"), v["harness"]["digest"]) for k, v in before.items()}


def test_rows_hold_no_absolute_path(jb):
    text = json.dumps(H2.read(jb.root).rows)
    assert str(jb.root) not in text and "/private/" not in text


def test_reading_writes_nothing(jb):
    before = sorted((p, p.stat().st_mtime_ns) for p in jb.root.rglob("*"))
    H2.read(jb.root)
    assert sorted((p, p.stat().st_mtime_ns) for p in jb.root.rglob("*")) == before


def test_the_reader_registers_no_ade_adapter():
    from loopmath.adapters.registry import list_adapters

    names = [a if isinstance(a, str) else getattr(a, "name", "") for a in list_adapters()]
    assert "harness2" not in names and "harness-runs" not in names


def test_experiment_id_is_the_id_read_gives(jb):
    """For a writer of the same rollouts (a wrapper around harness2): one rollout, one run in a store."""
    exp = {row["experiment"] for row in H2.read(jb.root).rows}
    for runs in (jb.root, jb.root / "jb", jb.adapt):
        for namespace in ("jb-oc-agg2", "jb-oc-seq2"):
            assert {H2.experiment_id(runs, namespace)} == exp
    labelled = {row["experiment"] for row in H2.read(jb.root, experiment="rep2").rows}
    assert {H2.experiment_id(jb.root, "jb-oc-agg2", label="rep2")} == labelled
    with pytest.raises(H2.FormatError, match="jb-oc-agg3"):
        H2.experiment_id(jb.root, "jb-oc-agg3")


# ---------------------------------------------------------------- the folder being read is the boundary

def _summary_path(b, task="finance/task1"):
    return b.adapt / F.ns(b.bench, b.substrate, "agg", b.k, None) / task.replace("/", "__") / "summary.json"


def _outside_rollout(tmp_path):
    """A synthetic rollout folder outside the runs root, with a stream and a marker of its own."""
    out = tmp_path / "elsewhere" / "rollout"
    out.mkdir(parents=True)
    (out / "codex_session.jsonl").write_text(
        json.dumps({"type": "turn.started"}) + "\n" + json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 777, "cached_input_tokens": 0, "output_tokens": 7}}) + "\n", encoding="utf-8")
    (out / ".adapt_complete").write_text(json.dumps({"version": 1, "run_id": str(out), "harness": "base"}),
                                         encoding="utf-8")
    return out


@pytest.mark.parametrize("bad", ["absolute", "dotdot"])
def test_a_base_run_id_outside_the_runs_root_is_refused(tmp_path, bad):
    b = F.build(tmp_path / "runs", bench="lab", substrate="cx", executor="gpt-5.4")
    outside = _outside_rollout(tmp_path)
    path = _summary_path(b)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["base"][0]["run_id"] = str(outside) if bad == "absolute" else "../../../../elsewhere/rollout"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(H2.FormatError, match="base run_id .*relative"):
        H2.read(b.root)


@pytest.mark.parametrize("field", ["task_id", "label", "mode"])
def test_traversing_ids_in_a_summary_are_refused_by_name(tmp_path, field):
    b = F.build(tmp_path / "runs")
    path = _summary_path(b)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if field == "task_id":
        doc["task_id"] = "/abs/elsewhere"
    elif field == "label":
        doc["modes"]["agg"]["ledger"]["rollouts"].append("../../x")
    else:
        doc["modes"]["../x"] = doc["modes"].pop("seq")
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(H2.FormatError, match={"task_id": "task_id", "label": "rollout label", "mode": "mode"}[field]):
        H2.read(b.root)


def test_a_stream_that_links_outside_the_root_is_refused(tmp_path):
    b = F.build(tmp_path / "runs", bench="lab", substrate="cx", executor="gpt-5.4")
    outside = _outside_rollout(tmp_path)
    r = next(r for r in b.rollouts if r.arm == "base")
    stream = next(b.root.glob(f"lab/results/cx/{r.run_id}/codex_session.jsonl"))
    stream.unlink()
    stream.symlink_to(outside / "codex_session.jsonl")
    with pytest.raises(H2.FormatError, match="resolves outside"):
        H2.read(b.root)


def test_a_tree_entry_that_links_outside_the_root_is_refused(tmp_path):
    b = F.build(tmp_path / "runs")
    secret = tmp_path / "elsewhere.md"
    secret.write_text("not part of the harness\n", encoding="utf-8")
    r = next(r for r in b.rollouts if r.arm == "adopted")
    (r.tree / "linked.md").symlink_to(secret)
    with pytest.raises(H2.FormatError, match="tree entry .*resolves outside"):
        H2.read(b.root)


# ---------------------------------------------------------------- WorkBuddy, mixed roots, benches not read

WB_TASKS = ("quarterly-budget-variance-analysis-for-finance", "travel-policy-memo")   # the first is over 32 characters


@pytest.fixture
def wb(tmp_path) -> F.Built:
    return F.build(tmp_path / "runs", bench="wb-office", tasks=WB_TASKS)


def _marker(trial: Path) -> dict:
    return json.loads((trial / ".adapt_complete").read_text(encoding="utf-8"))


@pytest.mark.parametrize("bench,substrate,stream", [("wb-office", "oc", "agent/oc-output.txt"),
                                                    ("wb-code", "cx", "agent/codex-output.jsonl")])
def test_workbuddy_rows_have_outcome_and_tree_and_name_their_unknown_cost(tmp_path, bench, substrate, stream):
    b = F.build(tmp_path / "runs", bench=bench, substrate=substrate, tasks=WB_TASKS)
    got = H2.read(b.root)
    rows = _by_run(got.rows)
    assert set(rows) == {r.run_id for r in b.rollouts} and len(rows) == 28
    for r in b.rollouts:
        row = rows[r.run_id]
        assert HR.check_row(row) == [], row
        assert row["harness"]["digest"] == harness.digest(r.tree, layout="harness2")
        assert (row.get("outcome") or {}).get("score") == r.score
        assert "cost" not in row and stream in row["ext"][H2NS]["cost_unread"]
        assert row["model"] == b.executor and row["task"]["repo"] == f"harness2-{bench}"
        assert row["ext"][H2NS]["bench"] == bench
    scored = [rows[r.run_id]["outcome"] for r in b.rollouts if r.score is not None]
    assert len(scored) == 8 and {o["judge"] for o in scored} == {f"{bench}-verifier"}
    assert any(o["score"] != o["n_passed"] / o["n_criteria"] for o in scored)   # the unrounded raw_score
    assert got.counts["cost_unread"] == 28 and any(stream in n and "cost_unread" in n for n in got.notes)


def test_workbuddy_bases_map_to_their_recorded_run_dir_not_their_marker(wb):
    rows = _by_run(H2.read(wb.root).rows)
    bases = [r for r in wb.rollouts if r.arm == "base"]
    for r in bases:
        first = r.run_id.rsplit("/", 1)[0] + "/base1"
        assert _marker(r.run_dir)["run_id"] == first          # every trial of the batch names base1
        assert rows[r.run_id]["ended_at"] == H2._iso(_marker(r.run_dir)["at"])
    assert len({rows[r.run_id]["ended_at"] for r in bases}) == len(bases)
    by_name = sorted(r.run_dir.name for r in bases if r.task_id == WB_TASKS[1])
    assert [r.run_dir.name for r in bases if r.task_id == WB_TASKS[1]] == by_name[::-1]   # base1 sorts last


@pytest.mark.parametrize("root_of", ["runs", "wb", "subset", "adapt"])
def test_any_level_of_a_workbuddy_folder_reads_the_same(wb, root_of):
    path = {"runs": wb.root, "wb": wb.root / "wb", "subset": wb.home, "adapt": wb.adapt}[root_of]
    assert _by_run(H2.read(path).rows).keys() == {r.run_id for r in wb.rollouts}


def test_workbuddy_dirs_recorded_on_another_machine_are_re_anchored(tmp_path, wb):
    moved = F.build(tmp_path / "moved" / "runs", bench="wb-office", tasks=WB_TASKS, variants={"moved"})
    summary = json.loads(next(moved.adapt.glob("*agg2/*/summary.json")).read_text(encoding="utf-8"))
    assert summary["base"][0]["run_dir"].startswith(F.RAN_AT + "/wb/office/rollouts/")
    assert summary["selections"]["agg"]["chosen_run_dir"].startswith(F.RAN_AT + "/wb/office/rollouts/")
    def strip(rows):
        return {k: (v.get("outcome"), v["harness"]["digest"], v.get("ended_at")) for k, v in _by_run(rows).items()}
    assert strip(H2.read(moved.root).rows) == strip(H2.read(wb.root).rows)


@pytest.mark.parametrize("bad", ["outside", "dotdot", "missing", "other-task", "empty"])
def test_a_recorded_workbuddy_run_dir_is_read_only_inside_its_bench(tmp_path, wb, bad):
    outside = _outside_rollout(tmp_path)
    path = _summary_path(wb, WB_TASKS[0])
    doc = json.loads(path.read_text(encoding="utf-8"))
    rel = Path(doc["base"][0]["run_dir"]).relative_to(wb.home)
    other = next(r for r in wb.rollouts if r.arm == "base" and r.task_id == WB_TASKS[1])
    doc["base"][0]["run_dir"] = {"outside": str(outside), "dotdot": f"/x/wb/office/rollouts/../../{outside.name}/t__a1",
                                 "missing": str(wb.adapt / "missing"), "other-task": str(other.run_dir),
                                 "empty": ""}[bad]
    path.write_text(json.dumps(doc), encoding="utf-8")
    got = H2.read(wb.root)
    rid = doc["base"][0]["run_id"]
    assert rid not in _by_run(got.rows) and str(rel).startswith("rollouts/")
    reason = next(s["reason"] for s in got.skipped if s["run_id"] == rid)
    assert reason.startswith("unavailable: its recorded run_dir")
    assert str(outside) not in json.dumps(got.rows)


def test_a_long_recorded_run_dir_is_cut_at_the_front_with_an_ellipsis(tmp_path, wb):
    path = _summary_path(wb, WB_TASKS[0])
    doc = json.loads(path.read_text(encoding="utf-8"))
    long_dir = str(wb.adapt / ("x" * 90) / "t__a1")
    doc["base"][0]["run_dir"] = long_dir
    path.write_text(json.dumps(doc), encoding="utf-8")
    got = H2.read(wb.root)
    reason = next(s["reason"] for s in got.skipped if s["run_id"] == doc["base"][0]["run_id"])
    shown = reason.split("(", 1)[1].split(")", 1)[0]
    assert shown == repr("..." + long_dir[-77:]) and len(shown) == 82, reason


def test_a_recorded_workbuddy_trial_that_links_outside_is_refused(tmp_path, wb):
    outside = _outside_rollout(tmp_path)
    trial = next(r for r in wb.rollouts if r.arm == "base").run_dir
    shutil.rmtree(trial)
    trial.symlink_to(outside, target_is_directory=True)
    with pytest.raises(H2.FormatError, match="resolves outside"):
        H2.read(wb.root)


def test_a_workbuddy_stage_is_found_by_its_marker_and_a_rerun_by_its_tree(tmp_path):
    b = F.build(tmp_path / "runs", bench="wb-office", tasks=WB_TASKS, variants={"rerun-trial"})
    r = b.by_label(WB_TASKS[0], "agg", "inv1")
    reruns = [t for t in (r.run_dir.parent.parent).glob("*/*") if _marker(t)["run_id"] == r.run_id]
    assert len(reruns) == 2
    row = _by_run(H2.read(b.root).rows)[r.run_id]
    assert row["harness"]["tool"]["fingerprint_match"] is True
    assert row["ended_at"] == H2._iso(_marker(r.run_dir)["at"])
    twin = r.run_dir.parent.parent / "2026-10-19__07-00-00" / r.run_dir.name
    shutil.copytree(r.run_dir, twin)          # two trials on the same tree: which one ran is not knowable
    got = H2.read(b.root)
    assert r.run_id not in _by_run(got.rows)
    assert "several trials" in next(s["reason"] for s in got.skipped if s["run_id"] == r.run_id)


def test_workbuddy_dead_and_failed_rollouts(tmp_path):
    b = F.build(tmp_path / "runs", bench="wb-office", tasks=WB_TASKS, variants={"dead-rollout", "failed-rollout"})
    got = H2.read(b.root)
    rows = _by_run(got.rows)
    dead = [r for r in b.rollouts if r.dead]
    assert dead and all(r.run_id not in rows for r in dead)
    assert all("no label folder" in next(s["reason"] for s in got.skipped if s["run_id"] == r.run_id) for r in dead)
    failed = b.by_label(WB_TASKS[0], "agg", "dom1")
    assert failed.failed and rows[failed.run_id]["status"] == "failed" and "cost" not in rows[failed.run_id]


def test_a_mixed_jb_and_workbuddy_root_reads_both_and_names_the_benches_it_does_not_read(tmp_path):
    root = tmp_path / "runs"
    jb, wb = F.build(root), F.build(root, bench="wb-office", tasks=WB_TASKS)
    (root / "zz" / "adapt" / "zz-oc-agg2" / "t1").mkdir(parents=True)
    (root / "zz" / "adapt" / "zz-oc-agg2" / "t1" / "summary.json").write_text("{}", encoding="utf-8")
    (root / "wb" / "finance" / "adapt").mkdir(parents=True)
    got = H2.read(root)
    assert _by_run(got.rows).keys() == {r.run_id for r in jb.rollouts + wb.rollouts}
    assert {e["bench"] for e in got.experiments} == {"jb", "wb-office"}
    skipped = {s["run_id"]: s["reason"] for s in got.skipped}
    assert skipped.keys() == {"zz/adapt", "wb/finance/adapt"} and got.counts["unsupported"] == 2
    assert all(reason.startswith("unsupported bench folder") for reason in skipped.values())
    jb_cost = [row for row in got.rows if row["ext"][H2NS]["bench"] == "jb" and row.get("cost")]
    assert len(jb_cost) == len(jb.rollouts)      # the JB rows keep their recovered cost
    assert got.counts["cost_unread"] == len(wb.rollouts)


def test_a_root_with_only_benches_it_does_not_read_is_refused_naming_them(tmp_path):
    (tmp_path / "runs" / "zz" / "adapt").mkdir(parents=True)
    (tmp_path / "runs" / "wb" / "finance" / "adapt").mkdir(parents=True)
    with pytest.raises(H2.FormatError, match=r"wb/finance/adapt, zz/adapt"):
        H2.read(tmp_path / "runs")


def test_a_namespace_that_is_not_its_folders_bench_is_named_not_read(tmp_path, wb):
    jb = F.build(tmp_path / "jb-only")
    shutil.copytree(jb.adapt / "jb-oc-agg2", wb.adapt / "jb-oc-agg2")
    shutil.copytree(jb.adapt / "jb-oc-agg2", wb.adapt / "not_a_namespace")
    got = H2.read(wb.root)
    assert _by_run(got.rows).keys() == {r.run_id for r in wb.rollouts}
    skipped = {s["run_id"]: s["reason"] for s in got.skipped}
    assert "is not wb-office" in skipped["wb/office/adapt/jb-oc-agg2"]
    assert "not a harness2 namespace name" in skipped["wb/office/adapt/not_a_namespace"]
    assert got.counts["unread_namespace"] == 2


def test_experiment_id_reads_workbuddy_namespaces(wb):
    exp = {row["experiment"] for row in H2.read(wb.root).rows}
    for runs in (wb.root, wb.home, wb.adapt):
        assert {H2.experiment_id(runs, "wb-office-oc-seq2")} == exp


def _twin(trial: Path) -> Path:
    """A second trial of the same rollout, marked the same: a rerun on the same tree."""
    twin = trial.parent.parent / "2026-10-19__07-00-00" / trial.name
    shutil.copytree(trial, twin)
    return twin


def test_a_rerun_adopted_parent_whose_tree_changed_is_never_a_childs_parent(wb):
    """The adopted c1 ran twice; its tree was edited after. Its children must not take the edited tree."""
    task = WB_TASKS[0]
    c1 = wb.by_label(task, "seq", "c1")
    assert c1.arm == "adopted"
    _twin(c1.run_dir)
    (c1.tree / "guardrails.md").write_text("Edited after the rollout ran.\n", encoding="utf-8")
    got = H2.read(wb.root)
    rows = _by_run(got.rows)
    assert "harness" not in rows[c1.run_id] and rows[c1.run_id]["ext"][H2NS]["tree_changed"] is True
    for label in ("inv2", "dom2", "c2"):
        row = rows[wb.by_label(task, "seq", label).run_id]
        assert row["harness"]["parent"] is None and row["ext"][H2NS]["parent_changed"] is True
    other = rows[wb.by_label(WB_TASKS[1], "seq", "inv2").run_id]
    assert other["harness"]["parent"] == harness.digest(wb.by_label(WB_TASKS[1], "seq", "c1").tree, layout="harness2")


def test_a_parent_whose_rollout_cannot_be_told_apart_is_left_out_by_name(tmp_path):
    b = F.build(tmp_path / "runs", bench="wb-office", tasks=WB_TASKS, variants={"fallback"})
    task = WB_TASKS[0]
    c1 = b.by_label(task, "seq", "c1")
    assert c1.arm == "candidate"           # nothing adopted, so no recorded run dir to go by
    _twin(c1.run_dir)
    got = H2.read(b.root)
    rows = _by_run(got.rows)
    assert c1.run_id not in rows
    inv2 = rows[b.by_label(task, "seq", "inv2").run_id]
    assert inv2["harness"]["parent"] is None and "several trials" in inv2["ext"][H2NS]["parent_unverified"]
    assert got.counts["parent_unverified"] == 3 and any("cannot be checked" in n for n in got.notes)


def test_a_jb_parent_whose_rollout_is_gone_is_left_out_by_name(jb):
    c1 = jb.by_label("finance/task1", "seq", "c1")
    for sub in ("model_output", "model_traj"):
        shutil.rmtree(jb.root / "jb" / "dataset" / "finance/task1" / sub / c1.run_id.replace("/", "__"))
    rows = _by_run(H2.read(jb.root).rows)
    assert c1.run_id not in rows
    inv2 = rows[jb.by_label("finance/task1", "seq", "inv2").run_id]
    assert inv2["harness"]["parent"] is None and "no run folder" in inv2["ext"][H2NS]["parent_unverified"]
