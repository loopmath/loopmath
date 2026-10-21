"""`loopmath adapt shinkaevolve`: one mutator row per patch query in a ShinkaEvolve results folder, with the model
that answered it, its dollars, and whether it wrote a child that beat its parent. Fixtures follow ShinkaEvolve's
code (shinkafixture.py, README.md)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shinkafixture as F  # noqa: E402

from loopmath.cli import main  # noqa: E402

NS = "dev.loopmath.adapter.shinkaevolve"


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
    from loopmath.adapters import shinkaevolve as SH

    return SH.read(path, **kw)


def _by_run(rows: list[dict]) -> dict[str, dict]:
    return {r["tool"]["run_id"]: r for r in rows}


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def _state(folder: Path) -> list[tuple[str, str, int]]:
    return sorted((str(p.relative_to(folder)), hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                  for p in folder.rglob("*") if p.is_file())


def test_one_row_per_answered_query_of_a_settled_generation(tmp_path):
    b = F.build(tmp_path)
    got = _read(b.results_dir)
    assert set(_by_run(got.rows)) == set(b.expect)
    assert got.format == "shinkaevolve-queries"
    assert set(_by_run(_read(b.db).rows)) == set(b.expect)          # the database path works too
    for key, n in b.counts.items():
        assert got.counts[key] == n, key


def test_the_model_is_the_one_that_answered_the_query_not_the_proposal_draw(tmp_path):
    b = F.build(tmp_path)
    rows = _by_run(_read(b.results_dir).rows)
    for rid, e in b.expect.items():
        assert rows[rid]["model"] == e.model and rows[rid]["agent"] == "shinkaevolve", rid
        assert "effort" not in rows[rid], rid        # a query record carries no effort; the draw's is not used
    assert rows["gen-1/n1r1p1"]["model"] == "gpt-5-mini"     # the proposal's draw said o4-mini


def test_the_query_that_wrote_the_child_carries_its_outcome_and_the_rest_did_not_improve(tmp_path):
    b = F.build(tmp_path)
    rows = _by_run(_read(b.results_dir).rows)
    for rid, e in b.expect.items():
        out, note = rows[rid]["outcome"], rows[rid]["ext"][NS]
        assert out["accepted"] is e.improved and out["by"] == "shinkaevolve", rid
        if e.delta is None:
            assert out.get("delta") is None, rid
        else:
            assert out["delta"] == pytest.approx(e.delta, abs=1e-9), rid
        assert note.get("child_id") == e.child and note["wrote_child"] is (e.child is not None), rid
        [stage] = rows[rid]["stages"]
        assert stage["role"] == "mutator" and stage["piece"] == "mutate"
        assert stage["status"] == ("done" if e.applied else "failed"), rid
    # gen 5: the first patch applied but lost the novelty check, so it wrote no child.
    assert rows["gen-5/n1r1p1"]["stages"][0]["status"] == "done"
    assert rows["gen-5/n1r1p1"]["outcome"]["accepted"] is False


def test_an_incorrect_child_did_not_improve_and_has_no_delta(tmp_path):
    b = F.build(tmp_path)
    out = _by_run(_read(b.results_dir).rows)["gen-3/n1r1p1"]["outcome"]
    assert out["accepted"] is False and out.get("delta") is None


def test_each_query_costs_its_own_dollars(tmp_path):
    b = F.build(tmp_path)
    rows = _by_run(_read(b.results_dir).rows)
    for rid, e in b.expect.items():
        cost = rows[rid]["stages"][0]["cost"]
        assert cost["usd"] == pytest.approx(e.usd) and cost["usd_source"] == "shinkaevolve", rid
        assert "tokens" not in cost and "cost" not in rows[rid], rid


def test_accounting_takes_one_proposal_per_generation(tmp_path):
    # gen 7 has two failure events and gen 8 a failure event and a program: each generation is one proposal
    # (its folder is created exclusively), so each counts once, the program first.
    b = F.build(tmp_path)
    got = _read(b.results_dir)
    attributed = sum(e.usd for e in b.expect.values())
    assert got.accounting["usd_total"] == pytest.approx(b.usd_total)
    assert got.accounting["usd_attributed"] == pytest.approx(attributed)
    assert got.accounting["usd_unattributed"] == pytest.approx(b.usd_total - attributed)
    assert got.counts["failed_generations"] == 2          # gens 4 and 7, as Shinka counts them
    assert got.counts["generations"] == 7


def test_unsettled_generations_and_generation_0_are_counted_not_rows(tmp_path):
    b = F.build(tmp_path)
    got = _read(b.results_dir)
    assert not any(r["tool"]["run_id"].startswith(("gen-0/", "gen-6/")) for r in got.rows)
    assert got.counts["unsettled_generations"] == 1 and got.counts["generation_0_queries"] == 1
    assert "unsettled" in " ".join(got.notes)


def test_a_writer_that_disagrees_with_the_child_reply_is_counted(tmp_path):
    b = F.build(tmp_path)
    rec = b.results_dir / "gen_1" / "attempts" / "novelty_1" / "resample_1" / "patch_1" / "metadata.json"
    md = json.loads(rec.read_text(encoding="utf-8"))
    rec.write_text(json.dumps({**md, "llm_model": "o4-mini"}), encoding="utf-8")
    got = _read(b.results_dir)
    assert got.counts["writer_mismatch"] == 1
    assert _by_run(got.rows)["gen-1/n1r1p1"]["model"] == "o4-mini"      # the query's own record


def test_the_folder_is_read_only_and_left_as_it_was(tmp_path):
    b = F.build(tmp_path)
    before = _state(b.results_dir)
    _read(b.results_dir)
    assert _state(b.results_dir) == before
    assert not Path(str(b.db) + "-wal").exists()


def test_a_live_database_is_read_through_a_copy_with_its_wal(tmp_path):
    b = F.build(tmp_path)
    F.query_record(b.results_dir, generation=9, model="gemini-2.5-flash", cost=0.002, success=True, at=F.T0 + 600)
    with contextlib.closing(F.connect(b.db)) as con:
        F.insert_program(con, "s9", parent="s1", generation=9, score=0.80, correct=True, metadata=F.child_metadata(
            drawn_model="o4-mini", drawn_effort=None, api_costs=0.002, start=F.T0 + 600, end=F.T0 + 620,
            result=F.llm_result("gemini-2.5-flash", input_tokens=2000, output_tokens=300, cost=0.002)))
        con.commit()
        assert Path(str(b.db) + "-wal").exists()      # the writer is still open
        before = _state(b.results_dir)
        row = _by_run(_read(b.results_dir).rows)["gen-9/n1r1p1"]
        assert row["outcome"]["accepted"] is True     # s9 lives only in the WAL so far
        assert _state(b.results_dir) == before


def test_without_query_records_the_export_refuses_and_children_only_names_its_unit(tmp_path):
    b = F.build(tmp_path)
    alone = tmp_path / "copied" / "programs.sqlite"
    alone.parent.mkdir()
    shutil.copy2(b.db, alone)
    with pytest.raises(ValueError, match="--children-only"):
        _read(alone)
    got = _read(alone, children_only=True)
    rows = _by_run(got.rows)
    assert set(rows) == set(b.children)
    for cid, e in b.children.items():
        row = rows[cid]
        assert row["model"] == e.model and row.get("effort") == e.effort, cid
        cost = row["stages"][0]["cost"]
        assert cost["usd"] == pytest.approx(e.usd) and cost["tokens"] == {"input": e.tokens[0], "output": e.tokens[1]}
        assert row["outcome"]["accepted"] is e.improved, cid
        if e.delta is not None:
            assert row["outcome"]["delta"] == pytest.approx(e.delta, abs=1e-9), cid
        assert row["ext"][NS]["unit"] == "evaluated child"
    assert rows["s1"]["effort"] == "high"             # the reply's own kwargs; the draw said low
    assert rows["s2"]["model"] == "gemini-2.5-flash"  # the reply's model; the draw said gpt-5-mini
    text = " ".join(got.notes)
    assert "per evaluated child" in text and "failed proposal" in text


def test_no_code_diff_patch_text_or_message_reaches_rows_or_run_files(tmp_path, capsys):
    b = F.build(tmp_path)
    for extra in ([], ["--children-only"]):
        out = tmp_path / f"o{len(extra)}"
        rows = tmp_path / f"rows{len(extra)}.jsonl"
        code, _, err = _run(capsys, "adapt", "shinkaevolve", str(b.results_dir), "--out", str(out), "--rows",
                            str(rows), *extra)
        assert code == 0, err
        for path in [rows, *out.glob("*.ocp.json")]:
            assert F.SECRET not in path.read_text(encoding="utf-8"), path
        for doc in out.glob("*.ocp.json"):
            assert json.loads(doc.read_text(encoding="utf-8"))["privacy"]["profile"] == "metadata_only"


def test_copies_of_a_root_added_later_never_change_a_row_id(tmp_path):
    # Shinka copies the initial program to every island and spawns islands from a root copy; both keep a null
    # parent (database/islands.py:603-660, 974-1000) and take uuid4 ids, which may sort before the root's.
    b = F.build(tmp_path)
    before = {mode: sorted(r["id"] for r in _read(b.results_dir, children_only=mode).rows) for mode in (False, True)}
    with contextlib.closing(F.connect(b.db)) as con, con:
        F.insert_program(con, "00000000-0000-4000-8000-000000000001", parent=None, generation=0, score=0.50,
                         correct=True, island=3, metadata={"_is_island_copy": True, "_original_program_id": "s0"})
        F.insert_program(con, "00000000-0000-4000-8000-000000000002", parent=None, generation=0, score=0.50,
                         correct=True, island=4, metadata=F.spawned("s0", island=4, strategy="initial"))
    for mode in (False, True):
        got = _read(b.results_dir, children_only=mode)
        assert sorted(r["id"] for r in got.rows) == before[mode], mode
        assert got.counts["copies"] == b.counts["copies"] + 2


def test_an_export_after_copies_are_added_imports_as_the_same_runs(tmp_path, capsys):
    # The same queries exported before and after a copy of the root appears are one set of runs in the store.
    b = F.build(tmp_path)
    _import = lambda out: [_run(capsys, *argv)[0] for argv in (   # noqa: E731
        ("adapt", "shinkaevolve", str(b.results_dir), "--out", str(out)),
        ("run", "import", str(out), "--finish", "--no-fit", "--json"))]
    assert _import(tmp_path / "before") == [0, 0]
    with contextlib.closing(F.connect(b.db)) as con, con:
        F.insert_program(con, "00000000-0000-4000-8000-000000000001", parent=None, generation=0, score=0.50,
                         correct=True, island=3, metadata=F.spawned("s0", island=3, strategy="initial"))
    assert _import(tmp_path / "after") == [0, 0]
    names = lambda d: sorted(p.name for p in (tmp_path / d).glob("*.ocp.json"))   # noqa: E731
    assert names("before") == names("after")
    assert len(list((tmp_path / "lm-home" / "runs").glob("*.ocp.json"))) == len(b.expect)


def test_no_root_or_two_roots_that_are_not_copies_need_experiment(tmp_path, capsys):
    from loopmath.adapters.evolve import NeedsFlag

    b = F.build(tmp_path)
    with contextlib.closing(F.connect(b.db)) as con, con:
        F.insert_program(con, "r2", parent=None, generation=0, score=0.40, correct=True, metadata={})
    with pytest.raises(NeedsFlag, match=r"2 programs with no parent that are not island or spawned copies"):
        _read(b.results_dir)
    code, _, err = _run(capsys, "adapt", "shinkaevolve", str(b.results_dir), "--out", str(tmp_path / "o"))
    assert code == 2 and "ambiguous" in err and "--experiment" in err
    assert {r["experiment"] for r in _read(b.results_dir, experiment="run-a").rows} == {"run-a"}
    with contextlib.closing(F.connect(b.db)) as con, con:
        con.execute("DELETE FROM programs WHERE id IN ('s0', 'r2')")   # left with no parentless program but copies
    with pytest.raises(NeedsFlag, match=r"no root program .* not an island or spawned copy"):
        _read(b.results_dir)


def test_the_cli_writes_one_run_per_query_with_stable_ids_and_the_accounting(tmp_path, capsys):
    b = F.build(tmp_path)
    code, stdout, err = _run(capsys, "adapt", "shinkaevolve", str(b.results_dir), "--out", str(tmp_path / "a"),
                             "--json")
    assert code == 0, err
    summary = json.loads(stdout)
    assert summary["importer"] == "shinkaevolve" and summary["written"] == len(b.expect)
    assert summary["cost"]["reported"] == len(b.expect)
    assert summary["accounting"]["usd_total"] == pytest.approx(b.usd_total)
    exp = "shinka-" + hashlib.sha256(b"s0").hexdigest()[:16]
    assert summary["experiments"] == [{"experiment": exp, "runs": len(b.expect)}]
    code, _, err = _run(capsys, "adapt", "shinkaevolve", str(b.db), "--out", str(tmp_path / "b"))
    assert code == 0, err
    names = lambda d: sorted(p.name for p in (tmp_path / d).glob("*.ocp.json"))   # noqa: E731
    assert names("a") == names("b")
    rows = _read(b.results_dir).rows
    assert all(r["id"].startswith(f"shinkaevolve:{exp}:gen-") for r in rows)
    assert rows[0]["task_id"] == "circle_packing"      # the folder holding results_<time>, unless --task


def test_accept_at_and_model_are_refused(tmp_path, capsys):
    b = F.build(tmp_path)
    for flag, value in (("--accept-at", "0.5"), ("--model", "gpt-5-mini")):
        code, _, err = _run(capsys, "adapt", "shinkaevolve", str(b.results_dir), flag, value)
        assert code == 2 and flag in err
