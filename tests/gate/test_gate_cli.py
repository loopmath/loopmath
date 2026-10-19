"""`loopmath gate`: read runs (a JSON lines file or the store), print the decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from loopmath.cli import main
from tests.store.store_helpers import cli, finished_doc

try:
    from loopmath.harness import EXT_KEY
except ImportError:  # the harness record module is not merged on this branch yet
    EXT_KEY = "dev.loopmath.harness"

BASE = "sha256:" + "a" * 64
CAND = "sha256:" + "b" * 64


def _iv(mean, p80, p95):
    return {"mean": mean, "median": mean, "lo": p80[0], "hi": p80[1], "bands": {"80": p80, "95": p95}}


def _fake_comparison(base, cand, *, rule, budget_usd=None, **kw):
    n = min(len(base), len(cand))
    return SimpleNamespace(
        n_pairs=n,
        gain=_iv(4.0, [1.0, 7.0], [-0.5, 8.5]),
        p_better=0.9,
        cost_base=_iv(0.5, [0.5, 0.5], [0.5, 0.5]),
        cost_cand=_iv(0.6, [0.6, 0.6], [0.6, 0.6]),
        cpar_base=None,
        cpar_cand=None,
        verdict="more",
        next_pair_value_usd=5.0,
        reasons=["fake"],
        floor=0.0,
    )


@pytest.fixture
def fake_compare(monkeypatch):
    import loopmath.gate.core as core

    calls = []

    def fake(base, cand, *, rule, budget_usd=None, **kw):
        calls.append((len(base), len(cand), rule))
        return _fake_comparison(base, cand, rule=rule, budget_usd=budget_usd, **kw)

    monkeypatch.setattr(core, "default_compare", lambda: fake)
    return calls


def _line(task, digest, score, usd=0.5, arm=None):
    row = {
        "schema": "loopmath.harness-runs/0.1",
        "task_id": task,
        "harness": {"schema": "loopmath.harness/0.1", "digest": digest, "source": "hand", "factors": []},
        "outcome": {"name": "judge", "score": score, "judge": "fake-judge", "scale": "linear"},
        "cost": {"usd": usd, "tokens": {"input": 900, "cache_read": 0, "cache_write": 0, "output": 100}},
    }
    if arm is not None:
        row["arm"] = arm
    return row


def _runs_file(tmp_path, rows):
    p = tmp_path / "runs.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_json_from_a_runs_file(tmp_path, capsys, fake_compare):
    rows = [_line("t1", BASE, 50), _line("t1", BASE, 52), _line("t1", CAND, 55), _line("t1", CAND, 57)]
    f = _runs_file(tmp_path, rows)
    code, out, _err = cli(
        capsys, tmp_path / "lm", "gate", "--runs", str(f), "--base", f"digest:{BASE}", "--cand",
        f"digest:{CAND}", "--json",
    )
    assert code == 0
    assert list(out)[0] == "schema" and out["schema"] == "loopmath.gate/1"
    assert out["action"] == "replicate" and out["n_more"] == 2
    assert out["inputs"]["base"]["scored"] == 2 and out["inputs"]["cand"]["scored"] == 2
    assert fake_compare == [(2, 2, "score:judge")]


def test_task_filter(tmp_path, capsys, fake_compare):
    rows = [_line("t1", BASE, 50), _line("t2", BASE, 40), _line("t1", CAND, 55), _line("t2", CAND, 45)]
    f = _runs_file(tmp_path, rows)
    code, out, _ = cli(
        capsys, tmp_path / "lm", "gate", "--runs", str(f), "--task", "t2", "--base", f"digest:{BASE}",
        "--cand", f"digest:{CAND}", "--json",
    )
    assert code == 0
    assert fake_compare == [(1, 1, "score:judge")]


def _store_doc(run, task, digest, score, usd):
    doc = finished_doc(run, usd=usd)
    doc["run"]["task"]["id"] = task
    doc["run"]["ext"] = {EXT_KEY: {"schema": "loopmath.harness/0.1", "digest": digest, "source": "hand"}}
    doc["run"]["signals"] = [
        {
            "id": f"sig_{run}",
            "kind": "score",
            "name": "bench_score",
            "value": float(score),
            "better": "higher",
            "observed_at": "2026-09-20T10:05:00-07:00",
            "tier": "verified",
        }
    ]
    return doc


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_arms_by_digest_from_the_store_and_read_only(tmp_path, capsys, fake_compare):
    from loopmath.store import Store

    home = tmp_path / "lm"
    st = Store(home)
    for i, (digest, score) in enumerate([(BASE, 50), (BASE, 51), (BASE, 49), (CAND, 56), (CAND, 54)]):
        st.import_run(_store_doc(f"run_g{i}", "tsk_one", digest, score, 0.5), check=False)
    before = _tree_hash(home)
    code, out, _ = cli(
        capsys, home, "gate", "--base", f"digest:{BASE}", "--cand", f"digest:{CAND}",
        "--rule", "score:bench_score", "--json",
    )
    assert code == 0, out
    assert fake_compare == [(3, 2, "score:bench_score")]
    assert _tree_hash(home) == before


def test_arms_by_arm_name_from_a_runs_file(tmp_path, capsys, fake_compare):
    rows = [
        _line("t1", BASE, 50, arm="base"),
        _line("t1", BASE, 51, arm="base"),
        _line("t1", CAND, 55, arm="candidate"),
        _line("t1", BASE, 49),
    ]
    f = _runs_file(tmp_path, rows)
    code, out, _ = cli(
        capsys, tmp_path / "lm", "gate", "--runs", str(f), "--base", "arm:base", "--cand", "arm:candidate",
        "--json",
    )
    assert code == 0
    assert fake_compare == [(2, 1, "score:judge")]


def test_arms_by_arm_label_from_the_store(tmp_path, capsys, fake_compare):
    from loopmath.store import Store

    home = tmp_path / "lm"
    st = Store(home)
    for i, (arm, score) in enumerate([("base", 50), ("base", 51), ("candidate", 56)]):
        doc = _store_doc(f"run_a{i}", "tsk_one", BASE, score, 0.5)
        # The harness-runs importer writes a row's arm to run.labels.arm.
        doc["run"].setdefault("labels", {})["arm"] = arm
        st.import_run(doc, check=False)
    code, out, _ = cli(
        capsys, home, "gate", "--base", "arm:base", "--cand", "arm:candidate", "--rule", "score:bench_score",
        "--json",
    )
    assert code == 0, out
    assert fake_compare == [(2, 1, "score:bench_score")]


def test_arm_from_the_pair_object_when_there_is_no_label():
    from loopmath.gate import inputs

    doc = _store_doc("run_p1", "tsk_one", BASE, 50, 0.5)
    doc["run"].setdefault("ext", {})["dev.loopmath.pair"] = {"pair": "t1#1", "arm": "candidate", "experiment": None}
    assert inputs.arm_of(doc) == "candidate"
    doc["run"].setdefault("labels", {})["arm"] = "base"
    assert inputs.arm_of(doc) == "base"


def test_a_partial_cost_is_unknown_not_a_number():
    from loopmath.gate import inputs

    row = _line("t1", BASE, 50, usd=0.5)
    assert inputs.usd_of(row) == 0.5
    row["cost"]["complete"] = False
    assert inputs.usd_of(row) is None


def test_text_output(tmp_path, capsys, fake_compare):
    rows = [_line("t1", BASE, 50), _line("t1", BASE, 52), _line("t1", CAND, 55), _line("t1", CAND, 57)]
    f = _runs_file(tmp_path, rows)
    code, out, _ = cli(
        capsys, tmp_path / "lm", "gate", "--runs", str(f), "--base", f"digest:{BASE}", "--cand",
        f"digest:{CAND}",
    )
    assert code == 0
    lines = out.strip().splitlines()
    assert lines[0].startswith("gate: replicate 2 more pairs")
    assert "P(better) 0.90" in lines[1]
    # The unit is always printed: these rows say no scale, so linear.
    assert "floor 0 on the gain (linear)" in lines[1]
    assert "$0.50" in lines[2] and "$0.60" in lines[2]
    assert "\u2014" not in out


def test_fraction_rows_print_their_unit(tmp_path, capsys, fake_compare):
    rows = [_line("t1", BASE, 0.50), _line("t1", BASE, 0.52), _line("t1", CAND, 0.55), _line("t1", CAND, 0.57)]
    for r in rows:
        r["outcome"]["scale"] = "fraction"
    f = _runs_file(tmp_path, rows)
    code, out, _ = cli(capsys, tmp_path / "lm", "gate", "--runs", str(f), "--base", f"digest:{BASE}", "--cand",
                       f"digest:{CAND}")
    assert code == 0
    assert "(fraction)" in out.splitlines()[1]
    code, out, err = cli(capsys, tmp_path / "lm", "gate", "--runs", str(f), "--base", f"digest:{BASE}", "--cand",
                         f"digest:{CAND}", "--floor", "2")
    assert code == 1
    assert "did you mean 0.02?" in (out + err)
    code, out, err = cli(capsys, tmp_path / "lm", "gate", "--runs", str(f), "--base", f"digest:{BASE}", "--cand",
                         f"digest:{CAND}", "--floor", "1")
    assert code == 1
    assert "is not under 1" in (out + err) and "did you mean 0.01?" in (out + err)


def test_a_floor_under_zero_is_refused_before_compare_is_asked(tmp_path, capsys, fake_compare):
    rows = [_line("t1", BASE, 50), _line("t1", BASE, 52), _line("t1", CAND, 55), _line("t1", CAND, 57)]
    f = _runs_file(tmp_path, rows)
    code, out, err = cli(capsys, tmp_path / "lm", "gate", "--runs", str(f), "--base", f"digest:{BASE}", "--cand",
                         f"digest:{CAND}", "--floor", "-0.05")
    assert code == 1
    assert "floor must be a number, 0 or more" in (out + err)
    assert fake_compare == []


def test_unknown_selector_names_the_three_forms(tmp_path, capsys, fake_compare):
    f = _runs_file(tmp_path, [_line("t1", BASE, 50)])
    code, _out, err = cli(capsys, tmp_path / "lm", "gate", "--runs", str(f), "--base", "nope", "--cand", "x")
    assert code == 1
    for form in ("digest:", "cfg_", "arm:"):
        assert form in err


def test_usage_error_exits_2(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["gate", "--base", f"digest:{BASE}", "--home", str(tmp_path / "lm")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--cand" in err and "invalid choice" not in err


def test_gate_is_listed(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "gate" in capsys.readouterr().out
