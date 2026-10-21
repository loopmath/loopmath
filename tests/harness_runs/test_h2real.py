"""`adapt harness2` on a realistic runs folder: every file and field harness2 writes, imported with no loss.

`h2real.py` writes the folder from reading harness2's code (clone at 1cfe254, never run or imported), and
`H2REAL.md` lists every file and key harness2 writes there, with the harness2 line that writes it and
where the importer puts it. These tests hold the three together: the keys in the files the builder wrote
are exactly the inventory's (read from the files, so a key nobody writes cannot pass), every rollout
harness2 ran is one row or one named skip, and every per-rollout fact lands where the inventory says.

harness2's own records (a stage, a selection, a crashed mode's record) hold text a model or harness2 wrote. By
default (privacy metadata_only) a row keeps only their text-free part under `h2.NS` and the import counts what it
withheld, so no model text reaches the rows or the run files. Under privacy full they land whole under
`h2.RECORDS`, outside loopmath's namespaces, and a run that keeps them is `full` (spec/harness-runs.md section 6).

harness2 serializes no RunView (h2:harness2/core/recursion.py:79-105 omits `rollouts`; summary `base[]`
holds label, run id and run dir only, h2:harness2/core/schedule.py:628): a rollout's `ok` and `error`
never reach disk, so a rollout with no run folder stays an `unavailable` skip that says so, never a row
with an invented error.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import h2real  # noqa: E402
finally:
    sys.path.remove(str(HERE))

from loopmath.adapters import harness2 as h2  # noqa: E402
from loopmath.cli import main  # noqa: E402

NS = h2.NS
REC = h2.RECORDS
CELLS = [("jb", "oc"), ("lab", "cx")]
STAGE_KEYS = ("label", "dim", "fell_back", "kept_base", "attempts", "usd", "cached", "notes", "error")  # recursion.py:88-99
SELECTION_KEYS = ("chosen_id", "chosen_label", "chosen_run_id", "chosen_is_base", "why", "id_map", "dropped",
                  "n_candidates", "ok", "fallback", "error", "usd", "experience", "trivial", "evidence",
                  "chosen_tree_is_base", "harness_fingerprint")  # selection.py:46-68, schedule.py:539-548
STAGE_META = ("dim", "attempts", "cached", "fell_back", "kept_base")
SELECTION_META = ("chosen_id", "chosen_label", "chosen_is_base", "chosen_tree_is_base", "n_candidates", "ok",
                  "fallback", "trivial", "harness_fingerprint", "evidence")


def _inventory() -> list[dict]:
    """H2REAL.md's key table: | file | key | type | written at | lands in |."""
    rows = []
    for line in (HERE / "H2REAL.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].startswith("`") and cells[1].startswith("`"):
            rows.append({"file": cells[0].strip("`"), "key": cells[1].strip("`"), "type": cells[2],
                         "h2": cells[3], "lands": cells[4]})
    return rows


def _leaves(value, prefix: str = "") -> set[str]:
    """Dotted key paths to every leaf: `a.b`, `[]` for a list's dicts; an empty dict or list is a leaf."""
    if isinstance(value, dict) and value:
        out: set[str] = set()
        for key, sub in value.items():
            out |= _leaves(sub, f"{prefix}.{key}" if prefix else str(key))
        return out
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        out = set()
        for sub in value:
            out |= _leaves(sub, f"{prefix}[]" if prefix else "[]")
        return out
    return {prefix} if prefix else set()


def _keys_in(path: Path, pattern: str) -> set[str]:
    """The keys a written file holds, read from the file itself."""
    if pattern.endswith("/"):
        return {"(tree)"}
    name = path.name
    if name.endswith(".json") or name == h2.MARK:
        return _leaves(json.loads(path.read_text(encoding="utf-8")))
    if name.endswith(".jsonl"):
        out: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out |= _leaves(json.loads(line))
        return out
    if name.endswith(".tsv"):
        with path.open(encoding="utf-8", newline="") as fh:
            return set(next(csv.reader(fh, delimiter="\t")))
    return {"(file)"}


def _matches(pattern: str, key: str) -> bool:
    """`*` is one map key (a mode, an anonymous id); everything else must be equal."""
    ps, ks = re.split(r"\.(?![^\[]*\])", pattern), re.split(r"\.(?![^\[]*\])", key)
    return len(ps) == len(ks) and all(p == "*" or p == k or (p.startswith("*") and k.endswith(p[1:]))
                                      for p, k in zip(ps, ks))


@pytest.fixture(scope="module", params=CELLS, ids=lambda c: f"{c[0]}-{c[1]}")
def built(request, tmp_path_factory):
    bench, sub = request.param
    root = tmp_path_factory.mktemp(f"h2real-{bench}-{sub}") / "runs"
    b = h2real.build(root, bench=bench, sub=sub)
    return b, h2.read(root)


@pytest.fixture(scope="module")
def kept(built):
    """The same folder read with privacy full."""
    b, _res = built
    return b, h2.read(b.root, privacy="full")


def _model_texts(b) -> list[str]:
    """Text a model or harness2 wrote in the folder's records: the editors' notes and errors, the selector's reason,
    lesson and drop causes, a crashed mode's error and traceback."""
    out = [s[k] for s in b.stages.values() for k in ("notes", "error")]
    for sel in b.selections.values():
        out += [sel.get("why"), sel.get("error"), *sel.get("dropped", []), *(sel.get("experience") or {}).values()]
    out += [b.crash.get("error"), b.crash.get("traceback")]
    return sorted({t for t in out if isinstance(t, str) and len(t) > 8})


def _holding(text: str, texts: list[str]) -> list[str]:
    return [t for t in texts if t in text or json.dumps(t, ensure_ascii=False)[1:-1] in text]


def test_every_inventory_line_cites_harness2():
    inv = _inventory()
    assert len(inv) > 50
    for r in inv:
        assert re.search(r"h2:[\w/.]+\.(py|sh):\d+", r["h2"]), r
        assert r["lands"], r


def test_the_files_written_hold_exactly_the_inventorys_keys(built):
    """Read every file the builder wrote; each key in it is in the inventory, and each inventory key for
    this cell is in some written file. A key harness2 does not write (a stage `ok`) fails here."""
    b, _res = built
    inv = [r for r in _inventory() if b.applies(r["file"])]
    seen: dict[str, set[str]] = {}
    for rel, pattern in b.files.items():
        seen.setdefault(pattern, set()).update(_keys_in(b.root / rel, pattern))
    extra = sorted((pat, key) for pat, keys in seen.items() for key in keys
                   if not any(r["file"] == pat and _matches(r["key"], key) for r in inv))
    assert not extra, f"written but not in H2REAL.md: {extra}"
    unwritten = sorted((r["file"], r["key"]) for r in inv
                       if not any(_matches(r["key"], key) for key in seen.get(r["file"], ())))
    assert not unwritten, f"in H2REAL.md but never written: {unwritten}"


def test_every_rollout_harness2_ran_is_one_row_or_one_named_skip(built):
    b, res = built
    rows = [r["tool"]["run_id"] for r in res.rows]
    skips = [s["run_id"] for s in res.skipped if s["run_id"] in b.rollout_ids]
    assert len(rows) == len(set(rows))
    assert not set(rows) & set(skips)
    assert sorted(rows + skips) == sorted(b.rollout_ids)


def test_a_rollout_with_no_folder_stays_unavailable_and_says_why(built, kept):
    b, res = built
    by_skip = {s["run_id"]: s for s in res.skipped}
    for rid in (b.dead_id, b.dead_base_id):
        skip = by_skip[rid]
        assert skip["reason"].startswith("unavailable"), skip
        assert "records no rollout error" in skip["reason"], skip
    assert "recorded it dead" in by_skip[b.dead_base_id]["reason"]
    stage = by_skip[b.dead_id]["stage"]  # the stage's session was paid even though its rollout died
    assert stage == {k: b.stages[b.dead_id][k] for k in STAGE_META}
    stage = {s["run_id"]: s for s in kept[1].skipped}[b.dead_id]["stage"]
    assert {k: stage[k] for k in STAGE_KEYS} == b.stages[b.dead_id]


def test_a_rollout_that_left_no_marker_is_a_failed_row(built):
    b, res = built
    row = next(r for r in res.rows if r["tool"]["run_id"] == b.unmarked_id)
    assert row["status"] == "failed" and row.get("outcome") is None


def test_stage_records_reach_the_rows_as_harness2_writes_them(kept):
    b, res = kept
    by_id = {r["tool"]["run_id"]: r for r in res.rows}
    checked = 0
    for rid, stage in b.stages.items():
        if rid not in by_id:
            continue
        note = by_id[rid]["ext"][REC]["stage"]
        assert {k: note[k] for k in STAGE_KEYS} == stage, rid
        assert "ok" not in note, rid  # harness2 writes no stage ok
        checked += 1
    assert checked >= 2 * len(b.modes_ok)


def test_by_default_a_row_keeps_the_stage_and_selection_without_their_text(built, kept):
    b, res = built
    by_id = {r["tool"]["run_id"]: r for r in res.rows}
    full = {r["tool"]["run_id"]: r for r in kept[1].rows}
    assert not any(REC in r["ext"] for r in res.rows)
    for rid, stage in b.stages.items():
        if rid in by_id:
            assert by_id[rid]["ext"][NS]["stage"] == {k: stage[k] for k in STAGE_META}, rid
    for mode, sel in b.selections.items():
        note = by_id[sel["chosen_run_id"]]["ext"][NS]["selection"]
        assert note == {**{k: sel[k] for k in SELECTION_META if k in sel}, "n_dropped": len(sel["dropped"]),
                        "has_experience": bool(sel["experience"])}, mode
    for rid, row in by_id.items():      # the text-free part is the same in both modes
        assert row["ext"][NS] == full[rid]["ext"][NS], rid
    assert res.counts["records_withheld"] == sum(len(r["ext"].get(REC, {})) for r in kept[1].rows) > 0
    assert kept[1].counts["records_withheld"] == kept[1].counts["text_withheld"] == 0


def test_the_selection_reaches_the_adopted_row(kept):
    b, res = kept
    by_id = {r["tool"]["run_id"]: r for r in res.rows}
    for mode, sel in b.selections.items():
        note = by_id[sel["chosen_run_id"]]["ext"][REC]["selection"]
        for key in SELECTION_KEYS:
            if key in sel:
                assert note[key] == sel[key], (mode, key)
        assert "chosen_run_dir" in sel and "chosen_run_dir" not in note, mode   # an absolute path, not kept


def test_the_judges_detail_is_kept(built):
    b, res = built
    by_id = {r["tool"]["run_id"]: r for r in res.rows}
    assert b.judged
    for rid, raw in b.judged.items():
        out = by_id[rid]["outcome"]
        assert out["score"] == pytest.approx(raw["n_passed"] / raw["n_criteria"]), rid
        detail = out.get("detail", {})
        for key, value in raw.items():   # every member of the judge's dict, once, where the spec puts it
            got = out["judge"] if key == "judge_model" else out[key] if key in out else detail[key]
            assert got == value, (rid, key)
        assert not set(detail) & set(out), rid


def test_a_crashed_modes_record_and_rollouts_are_kept(built, kept):
    b, res = kept
    by_id = {r["tool"]["run_id"]: r for r in res.rows}
    assert b.crashed_ids
    for rid in b.crashed_ids:
        assert by_id[rid]["ext"][REC]["mode_crash"] == {k: v for k, v in b.crash.items() if k != "traceback"}, rid
        assert by_id[rid]["ext"][NS]["mode_crashed"] is True, rid
    by_id = {r["tool"]["run_id"]: r for r in built[1].rows}
    for rid in b.crashed_ids:     # by default the kind only (h2real.CRASH is an OSError)
        assert by_id[rid]["ext"][NS]["crash_kind"] == "OSError" and REC not in by_id[rid]["ext"], rid


def test_by_default_the_experiment_keeps_no_error_text(built):
    b, res = built
    for exp in res.experiments:
        assert exp["config"] == b.config
        assert exp["sweep"] == {**b.sweep, "domains": [
            {**{k: v for k, v in d.items() if k != "error"},
             "tasks": [{k: v for k, v in t.items() if k != "error"} for t in d["tasks"]]} for d in b.sweep["domains"]]}
        for rec in exp["tasks"].values():
            assert not set(h2.TEXT_KEYS) & set(rec)
            assert not any(set(h2.TEXT_KEYS) & set(m) for m in rec["modes"].values() if isinstance(m, dict))
    assert res.counts["text_withheld"] > 0


def test_the_sweep_config_and_record_are_on_the_experiment_once(kept):
    b, res = kept
    assert len(res.experiments) == len(b.namespaces)
    for exp in res.experiments:
        assert exp["config"] == b.config
        assert exp["sweep"] == b.sweep
        summaries = {p.parent.name: json.loads(p.read_text()) for p in (b.root / b.bench / "adapt" / exp["namespace"]).glob("*/summary.json")}
        assert len(exp["tasks"]) == len(summaries) == 2
        for task, rec in exp["tasks"].items():   # the task's record, its modes without the stages the rows carry
            s = summaries[task.replace("/", "__")]
            assert {k: rec[k] for k in ("domain", "seed", "status", "errors", "k")} == \
                {k: s[k] for k in ("domain", "seed", "status", "errors", "k")}
            for mode, m in s["modes"].items():
                assert rec["modes"][mode] == {k: v for k, v in m.items() if k != "stages"}


def test_text_harness2_wrote_makes_a_run_full_and_a_base_stays_metadata_only(kept):
    from loopmath.harness_runs.ocp import to_ocp

    b, res = kept
    base = next(r for r in res.rows if r["arm"] == "base")
    stage = next(r for r in res.rows if r["tool"]["run_id"] in b.stages)
    assert REC not in base["ext"]
    assert to_ocp(base)["privacy"]["profile"] == "metadata_only"
    doc = to_ocp(stage)
    assert doc["privacy"]["profile"] == "full"
    assert f"ext.{REC}.stage.notes" in doc["run"]["ext"]["dev.loopmath.adapter.harness-runs"]["retained_text"]


def test_a_default_import_leaves_no_model_text_in_the_rows_or_the_runs(built, tmp_path, monkeypatch, capsys):
    from loopmath.harness_runs.ocp import to_ocp

    b, res = built
    texts = _model_texts(b)
    assert len(texts) >= 5
    assert not _holding(json.dumps(res.rows, ensure_ascii=False), texts)
    assert all(to_ocp(r)["privacy"]["profile"] == "metadata_only" for r in res.rows)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for privacy in ("metadata_only", "full"):
        out, rows = tmp_path / privacy / "out", tmp_path / privacy / "rows.jsonl"
        capsys.readouterr()
        code = main(["adapt", "harness2", str(b.root), "--out", str(out), "--rows", str(rows), "--json",
                     *(["--privacy", "full"] if privacy == "full" else [])])
        summary = json.loads(capsys.readouterr().out)
        assert code == 0 and summary["privacy"] == privacy
        written = rows.read_text(encoding="utf-8") + "".join(p.read_text(encoding="utf-8") for p in out.iterdir())
        profiles = {json.loads(p.read_text(encoding="utf-8"))["privacy"]["profile"] for p in out.iterdir()}
        if privacy == "metadata_only":
            assert not _holding(written + json.dumps(summary, ensure_ascii=False), texts)
            assert profiles == {"metadata_only"}
            assert summary["counts"]["records_withheld"] == res.counts["records_withheld"] > 0
        else:
            assert _holding(written, texts) and profiles == {"metadata_only", "full"}
            assert summary["counts"]["records_withheld"] == 0
    capsys.readouterr()
    assert main(["adapt", "harness2", str(b.root), "--out", str(tmp_path / "text")]) == 0
    assert "--privacy full keeps them" in capsys.readouterr().out
    rows = tmp_path / "metadata_only" / "rows.jsonl"
    assert main(["adapt", "harness-runs", str(rows), "--privacy", "full"]) == 2
    assert "applies to adapt harness2 only" in capsys.readouterr().err


def test_a_jobbench_weighted_score_scores(built):
    """0.3.1 dropped a JobBench weighted score as unscored: its n_criteria is the float max score
    (benchmarks/jobbench.py:611-613), and the importer took only an int."""
    assert h2._score(7.5, 10.0) == 0.75
    assert h2._score(6.0, 12.0) == 0.5
    for bad in (0.0, float("nan"), float("inf"), True, None, "12"):
        assert h2._score(6.0, bad) is None, bad
    assert h2._score(float("nan"), 12.0) is None
    b, res = built
    by_id = {r["tool"]["run_id"]: r for r in res.rows}
    for rid, raw in b.judged.items():
        if raw.get("metric") == "weighted":
            assert isinstance(raw["n_criteria"], float)
            assert by_id[rid]["outcome"]["score"] == raw["n_passed"] / raw["n_criteria"], rid


def test_nothing_is_dropped_silently(built):
    b, res = built
    lands = {(r["file"], r["key"]): r["lands"] for r in _inventory() if b.applies(r["file"])}
    for (file, key), where in lands.items():
        if where.startswith("not imported"):
            assert "because" in where, (file, key)
    for skip in res.skipped:
        if skip["run_id"] in b.rollout_ids:
            assert skip["reason"], skip
