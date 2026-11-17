"""The small synthetic study under `tests/eval/data/study-small`: it reads, and it matches its rubric and judges."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import quality as Q

STUDY = Path(__file__).parent / "data" / "study-small"


def test_the_study_reads_and_is_consistent():
    rubric = C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC)
    judges = C.read_document(STUDY / "judges.json", C.SCHEMA_JUDGES)
    value = C.read_document(STUDY / "value.json", C.SCHEMA_VALUE)
    study = C.read_study(STUDY)
    assert (len(study.runs), len(study.scores), len(study.pairs), len(study.items)) == (16, 305, 18, 0)
    assert sorted(study.files()) == ["pairs", "runs", "scores"]
    runs = {run["run"]: run for run in study.runs}
    judge_ids = {row["id"] for row in judges["judges"]}
    for n, row in enumerate(study.scores, 1):
        assert row["run"] in runs and row["judge"] in judge_ids and row["rubric"] == C.rubric_ref(rubric)
        C.check_reading(row, C.component(rubric, row["component"]), where=f"line {n}: ")
    for pair in study.pairs:
        assert runs[pair["a"]]["input"] == runs[pair["b"]]["input"] and pair["judge"] in judge_ids
        assert pair["component"] == C.overall_component(rubric)
    assert Counter(run["arm"] for run in study.runs) == {"baseline": 8, "strong": 8}
    assert len({run["input"] for run in study.runs}) == 8 and not any(run["simulated"] for run in study.runs)


def test_the_design_has_repeats_an_anchor_a_correction_and_both_pair_orders():
    study = C.read_study(STUDY)
    read, not_read = C.live(study.scores)
    assert len(read) == 304 and not_read == [{"id": "s0021", "reason": "replaced", "by": "s0305"}]
    calls = Counter((r["run"], r["component"], r["judge"]) for r in read)
    by_judge = Counter(judge for (_, _, judge), n in calls.items() if n >= 2)
    assert by_judge == {"judge-strong": 64, "judge-cheap": 64, "reviewer": 8}  # targets x components with repeats
    assert len({run for (run, _, judge) in calls if judge == "reviewer"}) == 10  # anchored targets, above the 6 needed
    assert len({r["call"] for r in read}) == len(read)  # one component per call
    kinds = Counter(str(r["value"]) for r in read if r["component"] == "clarity")
    assert set(kinds) == {"poor", "ok", "good"}
    assert {r["value"] for r in read if r["component"] == "numbers_right"} == {0, 1}
    orders = Counter((p["judge"], p["shown_first"]) for p in study.pairs)
    assert orders == {("reviewer", "a"): 6, ("judge-strong", "a"): 6, ("judge-strong", "b"): 6}
    assert set(Counter(p["verdict"] for p in study.pairs)) == {"a", "b", "tie"}


def test_the_study_indexes_and_its_value_file_compiles():
    rubric = C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC)
    judges = C.read_document(STUDY / "judges.json", C.SCHEMA_JUDGES)
    study = C.read_study(STUDY)
    index = F.build_index(rubric, judges, study.runs)
    assert index["arms"] == ["baseline", "strong"] and index["families"] == ["alpha", "beta"]
    assert [d["id"] for d in index["dimensions"]] == ["final/numbers_right", "final/clarity", "final/depth",
                                                      "final/overall"]
    assert index["judges"] == ["reviewer", "judge-strong", "judge-cheap"] and len(index["runs"]) == 16
    spec = Q.compile_value(C.read_document(STUDY / "value.json", C.SCHEMA_VALUE), C.dimensions(rubric))
    assert spec.weights.tolist() == [0.0, 1.0, 2.0, 0.0] and spec.gates == [(0, 1.0)]
    digests = C.input_digests({**study.files(), "rubric": STUDY / "rubric.json", "judges": STUDY / "judges.json"})
    assert sorted(digests) == ["judges", "pairs", "rubric", "runs", "scores"]
