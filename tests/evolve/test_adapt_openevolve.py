"""`loopmath adapt openevolve`: one mutator row per child of an OpenEvolve program database, with the model that
wrote it, its tokens, and whether it beat its parent. Fixtures follow OpenEvolve's code (oefixture.py, README.md)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oefixture as F  # noqa: E402

from loopmath.cli import main  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "user-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "lm-home"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "lm-cache"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))


def _read(path, **kw):
    from loopmath.adapters import openevolve as OE

    return OE.read(path, **kw)


def _by_child(rows: list[dict]) -> dict[str, dict]:
    return {r["tool"]["run_id"]: r for r in rows}


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_one_row_per_child_and_the_root_and_migrants_are_not_rows(tmp_path):
    b = F.build(tmp_path)
    got = _read(b.output_dir)
    assert set(_by_child(got.rows)) == set(b.expect)
    assert {s["run_id"]: s["reason"] for s in got.skipped} == b.skipped
    assert got.format == "openevolve-checkpoint"


def test_each_row_names_the_model_tokens_and_whether_the_child_beat_its_parent(tmp_path):
    b = F.build(tmp_path)
    rows = _by_child(_read(b.output_dir).rows)
    for cid, e in b.expect.items():
        row = rows[cid]
        assert row["schema"] == "loopmath.harness-runs/0.1"
        assert row["model"] == e.model and row["agent"] == "openevolve"
        [stage] = row["stages"]
        assert stage["role"] == "mutator" and stage["piece"] == "mutate"
        assert stage["cost"]["tokens"] == {"input": e.tokens[0], "output": e.tokens[1]}
        assert "usd" not in stage["cost"]          # OpenEvolve records no dollars; loopmath prices tokens
        assert "cost" not in row                   # no top-level cost beside stages
        out = row["outcome"]
        assert out["accepted"] is e.improved and out["by"] == "openevolve"
        assert out["delta"] == pytest.approx(e.delta, abs=1e-9)
        assert row["task_id"] == "circle_packing" and row["task"]["type"] == "research"


def test_a_child_whose_parent_was_evicted_is_scored_from_the_parent_metrics_it_carries(tmp_path):
    b = F.build(tmp_path)
    row = _by_child(_read(b.output_dir).rows)["c4"]
    assert row["outcome"]["parent_score"] == pytest.approx(0.30)
    assert row["outcome"]["accepted"] is True


def test_a_child_with_no_token_usage_needs_model_and_says_how_many(tmp_path):
    b = F.build(tmp_path, variants={"no-usage"})
    with pytest.raises(ValueError, match=r"1 child.*--model"):
        _read(b.output_dir)
    row = _by_child(_read(b.output_dir, model="claude-sonnet-5-5").rows)["c5"]
    assert row["model"] == "claude-sonnet-5-5" and "tokens" not in row["stages"][0].get("cost", {})
    assert _by_child(_read(b.output_dir, model="claude-sonnet-5-5").rows)["c1"]["model"] == "gpt-5-mini"


def test_the_newest_checkpoint_is_read_and_the_notes_name_the_survivor_gap(tmp_path):
    b = F.build(tmp_path)
    got = _read(b.output_dir)
    text = " ".join(got.notes)
    assert "checkpoint_10" in text
    assert "10 iterations" in text and "4 children" in text
    assert "evict" in text and "failed iterations" in text


def test_a_checkpoint_folder_and_the_output_folder_give_the_same_ids(tmp_path):
    b = F.build(tmp_path)
    a = _read(b.output_dir).rows
    c = _read(b.db).rows
    assert [r["id"] for r in a] == [r["id"] for r in c]
    assert all(r["id"].startswith("openevolve:oe-") for r in a)
    assert len({r["experiment"] for r in a}) == 1 and a[0]["experiment"].startswith("oe-")
    assert [r["id"] for r in _read(b.output_dir).rows] == [r["id"] for r in a]


def test_the_trace_is_read_when_present_and_agrees_with_the_checkpoint(tmp_path):
    b = F.build(tmp_path, variants={"trace"})
    exp = _read(b.output_dir).rows[0]["experiment"]
    trace = _read(b.output_dir / "evolution_trace.jsonl", experiment=exp)
    assert trace.format == "openevolve-trace"
    rows = _by_child(trace.rows)
    assert set(rows) == {"c1", "c2", "c3"}       # c4's parent was evicted, so no trace line (process_parallel.py:779-784)
    ids = {r["tool"]["run_id"]: r["id"] for r in _read(b.output_dir).rows}
    for cid, row in rows.items():
        assert row["outcome"]["accepted"] is b.expect[cid].improved
        assert row["model"] == b.expect[cid].model
        assert row["id"] == ids[cid]


def test_a_trace_alone_has_no_root_so_it_needs_experiment(tmp_path):
    b = F.build(tmp_path, variants={"trace"})
    with pytest.raises(ValueError, match="--experiment"):
        _read(b.output_dir / "evolution_trace.jsonl")


def test_a_later_checkpoint_without_the_root_needs_experiment_and_keeps_shared_ids(tmp_path, capsys):
    b = F.build(tmp_path, variants={"root-evicted"})
    with pytest.raises(ValueError, match="--experiment"):
        _read(b.output_dir)                      # no guess from the surviving population
    early = _read(b.older)
    [exp] = {r["experiment"] for r in early.rows}
    assert [e["experiment"] for e in early.experiments] == [exp]
    late = _by_child(_read(b.output_dir, experiment=exp).rows)
    for cid, row in _by_child(early.rows).items():
        assert late[cid]["id"] == row["id"]       # c1 and c2 are in both checkpoints
    code, stdout, err = _run(capsys, "adapt", "openevolve", str(b.older), "--out", str(tmp_path / "o1"), "--json")
    assert code == 0, err
    assert json.loads(stdout)["experiments"] == [{"experiment": exp, "runs": 2}]
    code, _, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--out", str(tmp_path / "o2"))
    assert code == 2 and "--experiment" in err


def test_two_programs_with_no_parent_at_generation_0_need_experiment(tmp_path, capsys):
    # OpenEvolve adds one initial program (controller.py:275-300); a second one leaves the experiment id to a sort.
    b = F.build(tmp_path)
    extra = F.program("a0", parent=None, generation=0, iteration=0, metrics={"combined_score": 0.1})
    (b.db / "programs" / "a0.json").write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(ValueError, match=r"2 programs with no parent at generation 0 .*ambiguous"):
        _read(b.output_dir)
    code, _, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--out", str(tmp_path / "o"))
    assert code == 2 and "--experiment" in err
    assert {r["experiment"] for r in _read(b.output_dir, experiment="run-a").rows} == {"run-a"}


def test_no_code_prompt_or_change_text_reaches_rows_or_run_files(tmp_path, capsys):
    b = F.build(tmp_path, variants={"trace", "prompts"})
    out = tmp_path / "o"
    rows = tmp_path / "rows.jsonl"
    code, _, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--out", str(out), "--rows", str(rows))
    assert code == 0, err
    for path in [rows, *out.glob("*.ocp.json")]:
        assert F.SECRET not in path.read_text(encoding="utf-8"), path
    for doc in out.glob("*.ocp.json"):
        assert json.loads(doc.read_text(encoding="utf-8"))["privacy"]["profile"] == "metadata_only"


def test_the_cli_writes_one_run_per_child_and_summarises_skips(tmp_path, capsys):
    b = F.build(tmp_path)
    out = tmp_path / "o"
    code, stdout, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--out", str(out), "--json")
    assert code == 0, err
    summary = json.loads(stdout)
    assert summary["importer"] == "openevolve" and summary["written"] == 4
    assert [e["runs"] for e in summary["experiments"]] == [4]
    assert len(list(out.glob("*.ocp.json"))) == 4
    assert {s["reason"] for s in summary["skipped"]} == {"root", "migrant"}
    assert summary["cost"]["priced"] + summary["cost"]["unpriced"] == 4     # tokens priced by loopmath, or named


def test_model_is_shared_with_plugin_eval_and_refused_by_the_other_importers(tmp_path, capsys):
    b = F.build(tmp_path / "oe")
    code, _, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--model", "gpt-5-mini",
                        "--out", str(tmp_path / "out"))
    assert code == 0, err
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"schema": "loopmath.harness-runs/0.1", "task_id": "t1"}) + "\n", encoding="utf-8")
    for argv in (["harness-runs", str(rows)], ["harness2", str(tmp_path)]):
        code, _, err = _run(capsys, "adapt", *argv, "--model", "gpt-5-mini")
        assert code == 2 and "--model is for adapt plugin-eval and openevolve only" in err, argv
    code, _, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--cost-from", "c.jsonl")
    assert code == 2 and "--cost-from" in err and "plugin-eval" in err


def test_privacy_full_is_refused_since_the_export_copies_no_text(tmp_path, capsys):
    b = F.build(tmp_path)
    code, _, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--privacy", "full")
    assert code == 2 and "--privacy full applies to adapt harness2 only" in err and "copies no text" in err


def test_accept_at_is_refused_and_task_and_min_delta_apply(tmp_path, capsys):
    b = F.build(tmp_path)
    code, _, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--accept-at", "0.5")
    assert code == 2 and "--accept-at" in err
    rows = _by_child(_read(b.output_dir, task="packing-26", min_delta=0.06).rows)
    assert {r["task_id"] for r in rows.values()} == {"packing-26"}
    assert not any(r["outcome"]["accepted"] for r in rows.values())     # no child gained more than 0.06


def test_metric_picks_another_score_and_a_missing_metric_is_named(tmp_path):
    b = F.build(tmp_path)
    with pytest.raises(ValueError, match="no_such_metric"):
        _read(b.output_dir, metric="no_such_metric")
    db = F.build_features(tmp_path, with_stats=False)
    rows = _by_child(_read(db, metric="fitness").rows)
    assert rows["c1"]["outcome"]["delta"] == pytest.approx(1.0) and rows["c1"]["outcome"]["accepted"] is True
    assert rows["c1"]["outcome"]["metric"] == "fitness"
    rows = _by_child(_read(db, metric="spread").rows)
    assert rows["c1"]["outcome"]["delta"] == pytest.approx(-100.0) and rows["c1"]["outcome"]["accepted"] is False


def test_fitness_leaves_out_feature_dimensions_as_openevolve_does(tmp_path):
    # utils/metrics_utils.py:72-118: no combined_score, so the mean of metrics that are not feature dimensions.
    rows = _by_child(_read(F.build_features(tmp_path)).rows)
    assert rows["p1"]["outcome"]["delta"] == pytest.approx(0.5) and rows["p1"]["outcome"]["accepted"] is True
    assert rows["c1"]["outcome"]["delta"] == pytest.approx(1.0) and rows["c1"]["outcome"]["accepted"] is True
    assert rows["c1"]["outcome"]["child_score"] == pytest.approx(2.0)


def test_with_no_combined_score_and_no_feature_dimensions_known_the_export_asks_for_metric(tmp_path):
    with pytest.raises(ValueError, match="--metric"):
        _read(F.build_features(tmp_path, with_stats=False))


def test_the_reader_prices_only_through_the_table_it_is_given(tmp_path):
    # The adapter takes a loaded table (TablePrice) and never loads one: any caller can supply it.
    from loopmath.adapters.evolve import TablePrice

    b = F.build(tmp_path)
    asked = []
    price = TablePrice(usd=lambda model, tokens: asked.append((model, tokens)) or (0.5 if model == "gpt-5-mini"
                                                                                  else None),
                       path="given", as_of="2026-10-20")
    rows = _by_child(_read(b.output_dir, price=price).rows)
    assert rows["c1"]["stages"][0]["cost"]["usd"] == 0.5 and "usd" not in rows["c2"]["stages"][0]["cost"]
    assert ("gpt-5-mini", {"in": 1000, "cache_read": 0, "cache_write": 0, "out": 200}) in asked
    assert "usd" not in _by_child(_read(b.output_dir).rows)["c1"]["stages"][0]["cost"]


def test_a_price_table_prices_the_tokens_of_the_models_it_lists(tmp_path, capsys):
    from loopmath.harness_runs.commands import evolve_price

    b = F.build(tmp_path)
    table = F.price_table(tmp_path / "prices.toml", {"gpt-5-mini": (0.25, 2.0)})
    rows = _by_child(_read(b.output_dir, price=evolve_price(table)).rows)   # the command layer loads the table
    cost = rows["c1"]["stages"][0]["cost"]
    assert cost["usd"] == pytest.approx((1000 * 0.25 + 200 * 2.0) / 1e6)
    assert cost["usd_source"] == "price-table"
    assert "usd" not in rows["c2"]["stages"][0]["cost"]      # not in the table: left to loopmath's own prices
    code, stdout, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--prices", str(table),
                             "--out", str(tmp_path / "o"), "--json")
    assert code == 0, err
    summary = json.loads(stdout)
    assert summary["counts"]["priced_by_table"] == 2                # c1 and c4 are gpt-5-mini
    assert any(f"--prices {table} (as of " in n for n in summary["notes"])
    code, _, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--prices", str(tmp_path / "none.toml"))
    assert code == 2 and "none.toml" in err


def test_rows_priced_by_a_table_count_as_priced_from_tokens_in_every_import_summary(tmp_path, capsys):
    # A row whose cost.usd_source is price-table counts as priced from tokens (cost.priced), as loopmath-prices
    # does, never as reported: in this export's summary and when its rows are read again by adapt harness-runs.
    b = F.build(tmp_path)
    table = F.price_table(tmp_path / "prices.toml", {"gpt-5-mini": (0.25, 2.0)})
    rows_file = tmp_path / "rows.jsonl"
    code, stdout, err = _run(capsys, "adapt", "openevolve", str(b.output_dir), "--prices", str(table),
                             "--out", str(tmp_path / "o"), "--rows", str(rows_file), "--json")
    assert code == 0, err
    cost = json.loads(stdout)["cost"]
    tabled = [json.loads(line) for line in rows_file.read_text(encoding="utf-8").splitlines()]
    tabled = [r for r in tabled if r["stages"][0]["cost"].get("usd_source") == "price-table"]
    assert len(tabled) == 2 and cost["reported"] == 0 and cost["priced"] >= 2
    code, stdout, err = _run(capsys, "adapt", "harness-runs", str(rows_file), "--out", str(tmp_path / "o2"), "--json")
    assert code == 0, err
    assert json.loads(stdout)["cost"] == cost
