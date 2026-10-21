"""`loopmath gate winner`: read a search's candidate scores (CSV, JSON or harness-runs rows), print the verdict."""

from __future__ import annotations

import json

import pytest

from loopmath.cli import main
from tests.store.store_helpers import cli

CSV = """id,score,role
seed,0.700,parent
c0,0.702,
c1,0.689,
c2,0.711,
c3,0.734,
c4,0.698,
"""


def _file(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def _row(arm, score, task):
    return {"schema": "loopmath.harness-runs/0.1", "task_id": task, "arm": arm,
            "outcome": {"name": "judge", "score": score, "judge": "fake-judge", "scale": "fraction"}}


def _rows_file(tmp_path, rows, name="runs.jsonl"):
    return _file(tmp_path, name, "".join(json.dumps(r) + "\n" for r in rows))


def test_json_from_a_csv(tmp_path, capsys):
    f = _file(tmp_path, "scores.csv", CSV)
    code, out, _err = cli(capsys, tmp_path / "lm", "gate", "winner", "--scores", str(f), "--spread", "0.02",
                          "--draws", "20", "--json")
    assert code == 0
    assert list(out)[0] == "schema" and out["schema"] == "loopmath.gate-winner/1"
    assert out["winner"] == "c3" and out["parent"] == "seed"
    assert out["observed_lead"] == pytest.approx(0.034)
    assert out["n_candidates"] == {"scored": 5, "censored": 0, "total": 5}
    assert out["spread"]["source"] == "given"
    assert out["gate"]["schema"] == "loopmath.gate/1"
    for key in ("gain", "p_better", "shrinkage", "prior", "projection", "buy", "reasons"):
        assert key in out


def test_json_scores_with_censored_rows_and_a_source(tmp_path, capsys):
    doc = {"schema": "loopmath.gate-winner-scores/0.1", "better": "higher", "scale": "fraction",
           "spread": [0.015, 0.035],
           "parent": {"id": "seed", "score": 0.7247},
           "candidates": [{"id": "g1", "score": 0.7753, "order": 1}, {"id": "g3", "score": 0.7865, "order": 3}],
           "censored": [{"count": 1, "at_most": 0.7753, "order": [2, 2]},
                        {"count": 10, "at_most": 0.7865, "order": [4, 13]}],
           "source": {"cite": "arXiv 0000.00000v1", "where": "Table 1"}}
    f = _file(tmp_path, "scores.json", json.dumps(doc))
    code, out, _err = cli(capsys, tmp_path / "lm", "gate", "winner", "--scores", str(f), "--draws", "0", "--json")
    assert code == 0
    assert out["winner"] == "g3"
    assert out["n_candidates"] == {"scored": 2, "censored": 11, "total": 13}
    assert out["spread"]["source"] == "range" and len(out["by_spread"]) == 5
    assert out["source"]["cite"] == "arXiv 0000.00000v1"


def test_runs_rows_read_the_spread_from_replicates(tmp_path, capsys):
    rows = []
    for arm, level in (("base", 0.60), ("c1", 0.61), ("c2", 0.66), ("c3", 0.59)):
        for t in range(4):
            for j, wobble in enumerate((-0.02, 0.02)):
                rows.append(_row(arm, round(level + wobble + 0.01 * t, 4), f"t{t}"))
    f = _rows_file(tmp_path, rows)
    code, out, _err = cli(capsys, tmp_path / "lm", "gate", "winner", "--runs", str(f), "--parent", "arm:base",
                          "--draws", "0", "--json")
    assert code == 0
    assert out["winner"] == "c2" and out["parent"] == "base"
    assert out["n_candidates"]["scored"] == 3
    # every arm and task: two runs 0.04 apart, so sigma_t^2 = 0.0008 on each of T = 4 tasks
    assert out["spread"]["source"] == "runs" and out["spread"]["unit"] == "evaluation"
    assert out["spread"]["value"] == pytest.approx((4 * 0.0008) ** 0.5 / 4, abs=1e-9)  # one fresh pass
    assert out["inputs"]["winner"]["se"] == pytest.approx((4 * 0.0008 / 2) ** 0.5 / 4, abs=1e-9)  # the mean of 2


def test_runs_with_no_replicate_need_a_spread(tmp_path, capsys):
    rows = [_row(arm, 0.6 + 0.01 * i, f"t{t}") for i, arm in enumerate(("base", "c1", "c2")) for t in range(3)]
    f = _rows_file(tmp_path, rows)
    code, _out, err = cli(capsys, tmp_path / "lm", "gate", "winner", "--runs", str(f), "--parent", "arm:base")
    assert code == 2
    assert "--spread" in err


def test_text_output_leads_with_the_verdict(tmp_path, capsys):
    f = _file(tmp_path, "scores.csv", CSV)
    code, out, _err = cli(capsys, tmp_path / "lm", "gate", "winner", "--scores", str(f), "--spread", "0.02",
                          "--draws", "0")
    assert code == 0
    first = out.splitlines()[0]
    assert first.startswith("gate winner: c3 leads seed by +0.034")
    assert "P(beats parent)" in out and "gate: replicate" in out


@pytest.mark.parametrize(
    "text, needle",
    [
        ("id,score,role\nseed,0.7,parent\nseed2,0.71,parent\nc0,0.72,\n", "parent"),
        ("id,score\nc0,0.72\nc1,0.71\n", "parent"),
        ("id,score,role\nseed,0.7,parent\nc0,abc,\n", "score"),
        ("id,score,role\nseed,0.7,parent\nc0,0.72,\nc0,0.73,\n", "c0"),
        ("id,score,reps,role\nseed,0.7,,parent\nc0,0.72,0,\n", "reps"),
        ("id,score,at_most,count,reps,role\nseed,0.7,,,,parent\nc0,0.72,,,,\n,,0.72,5,0,\n", "reps"),
    ],
)
def test_bad_score_files_are_refused_naming_the_problem(tmp_path, capsys, text, needle):
    f = _file(tmp_path, "scores.csv", text)
    code, _out, err = cli(capsys, tmp_path / "lm", "gate", "winner", "--scores", str(f), "--spread", "0.02")
    assert code == 1  # a refused file exits 1, a missing input 2, as for the run importers
    assert needle in err


def test_a_confirm_file_goes_to_the_gate(tmp_path, capsys):
    f = _file(tmp_path, "scores.csv", CSV)
    confirm = _rows_file(tmp_path, [_row("base", 0.70, "t1"), _row("base", 0.71, "t2"),
                                    _row("candidate", 0.73, "t1"), _row("candidate", 0.74, "t2")], "confirm.jsonl")
    code, out, _err = cli(capsys, tmp_path / "lm", "gate", "winner", "--scores", str(f), "--spread", "0.02",
                          "--confirm", str(confirm), "--draws", "0", "--json")
    assert code == 0
    assert out["gate"]["inputs"]["base"]["scored"] == 2 and out["gate"]["inputs"]["cand"]["scored"] == 2


def test_gate_winner_reads_no_store(tmp_path, capsys):
    f = _file(tmp_path, "scores.csv", CSV)
    home = tmp_path / "lm"
    code, _out, _err = cli(capsys, home, "gate", "winner", "--scores", str(f), "--spread", "0.02", "--draws", "0")
    assert code == 0
    assert not home.exists()


def test_the_gate_keeps_its_own_form(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["gate", "--home", str(tmp_path / "lm")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--base" in err and "--cand" in err


def test_help_lists_winner(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["gate", "--help"])
    assert exc.value.code == 0
    assert "winner" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["gate", "winner", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--scores", "--runs", "--spread", "--confirm", "--winner"):
        assert flag in out
