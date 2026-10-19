"""`loopmath adapt harness2 PATH` and `loopmath adapt harness-runs FILE`, then `loopmath run import`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2fixture as F  # noqa: E402

from loopmath import harness_runs as HR  # noqa: E402
from loopmath.cli import main  # noqa: E402
from loopmath.ocp.conformance import validate_doc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "spec" / "examples" / "harness-runs"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "user-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_adapt_harness2_writes_one_valid_run_per_rollout(tmp_path, capsys):
    b = F.build(tmp_path / "runs")
    out = tmp_path / "ocp"
    code, stdout, err = _run(capsys, "adapt", "harness2", str(b.root), "--out", str(out))
    assert code == 0, err
    files = sorted(out.glob("*.ocp.json"))
    assert len(files) == len(b.rollouts)
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert not [f for f in validate_doc(doc) if f.level == "error"], path
        assert path.name == f"{doc['run']['id']}.ocp.json"
    assert "28" in stdout


def test_json_summary_counts(tmp_path, capsys):
    b = F.build(tmp_path / "runs", variants={"dead-rollout", "judge-failed"})
    code, stdout, err = _run(capsys, "adapt", "harness2", str(b.root), "--out", str(tmp_path / "o"), "--json")
    assert code == 0, err
    got = json.loads(stdout)
    assert got["schema"] == "loopmath.adapt.runs/1"
    assert got["format"] == "harness2/method-2"
    assert got["written"] == len([r for r in b.rollouts if not r.dead])
    assert got["scored"] == len([r for r in b.rollouts if r.score is not None])
    assert got["skipped"] and all(s["reason"] for s in got["skipped"])
    assert set(got["cost"]) >= {"recovered", "priced", "unpriced"}


def test_without_out_documents_go_to_stdout_as_json_lines(tmp_path, capsys):
    b = F.build(tmp_path / "runs", k=1, tasks=("finance/task1",))
    code, stdout, err = _run(capsys, "adapt", "harness2", str(b.root))
    assert code == 0, err
    docs = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    assert len(docs) == len(b.rollouts) and all(d["ocp"] == "0.3" for d in docs)


def test_rows_option_writes_the_generic_rows(tmp_path, capsys):
    b = F.build(tmp_path / "runs")
    rows = tmp_path / "rows.jsonl"
    code, _, err = _run(capsys, "adapt", "harness2", str(b.root), "--out", str(tmp_path / "o"), "--rows", str(rows))
    assert code == 0, err
    assert len(HR.read(rows)) == len(b.rollouts)


def test_a_refused_format_exits_1_and_writes_nothing(tmp_path, capsys):
    b = F.build(tmp_path / "runs", method_version=3)
    out = tmp_path / "o"
    code, _, err = _run(capsys, "adapt", "harness2", str(b.root), "--out", str(out))
    assert code == 1
    assert "method_version 3" in err
    assert not out.exists() or not any(out.iterdir())


def test_a_missing_path_exits_2(tmp_path, capsys):
    code, _, err = _run(capsys, "adapt", "harness2", str(tmp_path / "nope"), "--out", str(tmp_path / "o"))
    assert code == 2 and "nope" in err


def test_importing_twice_writes_the_same_files(tmp_path, capsys):
    b = F.build(tmp_path / "runs")
    _run(capsys, "adapt", "harness2", str(b.root), "--out", str(tmp_path / "a"))
    _run(capsys, "adapt", "harness2", str(b.root), "--out", str(tmp_path / "b"))
    a = {p.name: p.read_bytes() for p in (tmp_path / "a").glob("*.ocp.json")}
    assert a == {p.name: p.read_bytes() for p in (tmp_path / "b").glob("*.ocp.json")}


def test_adapt_harness_runs_imports_a_rows_file(tmp_path, capsys):
    src = EXAMPLES / "hand-ab.jsonl"
    code, _, err = _run(capsys, "adapt", "harness-runs", str(src), "--out", str(tmp_path / "o"))
    assert code == 0, err
    assert len(list((tmp_path / "o").glob("*.ocp.json"))) == len(HR.read(src))


def test_adapt_harness_runs_refuses_a_bad_file_by_line(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"schema": HR.SCHEMA, "task_id": "t"}) + "\n" + json.dumps({"schema": HR.SCHEMA}) + "\n")
    code, _, err = _run(capsys, "adapt", "harness-runs", str(bad), "--out", str(tmp_path / "o"))
    assert code == 1 and "bad.jsonl:2" in err


def test_a_fraction_score_over_1_is_refused_and_nothing_is_written(tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"schema": HR.SCHEMA, "task_id": "t1", "arm": arm,
                                        "outcome": {"score": score, "judge": "j", "scale": "fraction"}}) + "\n"
                            for arm, score in (("base", 0.5), ("candidate", 1.5))), encoding="utf-8")
    code, _, err = _run(capsys, "adapt", "harness-runs", str(rows), "--out", str(tmp_path / "o"))
    assert code == 1 and "rows.jsonl:2" in err and "outcome.score" in err and "[0, 1]" in err
    assert not (tmp_path / "o").exists()


def test_a_run_that_fails_strict_ocp_is_refused_before_anything_is_written(tmp_path, capsys, monkeypatch):
    """Every built run passes the store's strict validation before the first file is written."""
    from loopmath.harness_runs import commands

    real = commands.to_ocp

    def bad(row, **kw):
        doc = real(row, **kw)
        if row.get("arm") == "candidate":
            doc["run"]["signals"][0]["value"] = 2.0          # a fraction outside [0, 1]: E195
        return doc

    monkeypatch.setattr(commands, "to_ocp", bad)
    rows = tmp_path / "rows.jsonl"
    for arm in ("base", "candidate"):
        HR.append(rows, HR.row("t1", arm=arm, outcome={"score": 0.5, "judge": "j", "scale": "fraction"}))
    code, out, err = _run(capsys, "adapt", "harness-runs", str(rows), "--out", str(tmp_path / "o"))
    assert code == 1 and "not valid OCP" in err and "E195" in err and "nothing written" in err
    assert not (tmp_path / "o").exists() and out == ""
    code, out, err = _run(capsys, "adapt", "harness-runs", str(rows))
    assert code == 1 and out == ""


def test_rows_without_times_import_with_no_execution_times(tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    HR.append(rows, HR.row("t1", arm="base", outcome={"score": 0.5, "judge": "j", "scale": "fraction"}))
    code, _, err = _run(capsys, "adapt", "harness-runs", str(rows), "--out", str(tmp_path / "o"))
    assert code == 0, err
    [path] = (tmp_path / "o").glob("*.ocp.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert "started_at" not in doc["run"] and "started_at" not in doc["attempts"][0]
    assert "ended_at" not in doc["attempts"][0]
    assert doc["run"]["signals"][0]["observed_at"].endswith("Z")       # the rows file's time, observation only


def test_run_import_stores_every_run_with_its_record(tmp_path, capsys):
    from loopmath import harness
    from loopmath.store import Store

    b = F.build(tmp_path / "runs")
    out, home = tmp_path / "o", tmp_path / "store"
    assert _run(capsys, "adapt", "harness2", str(b.root), "--out", str(out))[0] == 0
    code, stdout, err = _run(capsys, "run", "import", str(out), "--finish", "--no-fit", "--json", "--home", str(home))
    assert code == 0, err
    got = json.loads(stdout)
    assert got["new"] == len(b.rollouts)
    store = Store(home)
    for rid in got["runs"]:
        doc = store.run_doc(rid)
        assert harness.run_record(doc)["source"] == "harness2"
        assert doc["run"]["labels"]["arm"] in ("base", "candidate", "adopted")
    code, stdout, _ = _run(capsys, "run", "import", str(out), "--finish", "--no-fit", "--json", "--home", str(home))
    assert code == 0 and json.loads(stdout)["new"] == 0


def test_help_names_the_run_importers(capsys):
    with pytest.raises(SystemExit):
        main(["adapt", "--help"])
    text = capsys.readouterr().out
    assert "harness2" in text and "harness-runs" in text


def test_nothing_appears_in_the_real_home(tmp_path, capsys):
    real = Path.home()  # HOME is the temp one here; the guard is that the reader never writes beside its input
    b = F.build(tmp_path / "runs")
    before = sorted(p.name for p in b.root.iterdir())
    _run(capsys, "adapt", "harness2", str(b.root), "--out", str(tmp_path / "o"))
    assert sorted(p.name for p in b.root.iterdir()) == before
    assert not (real / ".loopmath").exists()


def test_a_mixed_jb_and_workbuddy_root_writes_both_and_says_what_it_skipped(tmp_path, capsys):
    root = tmp_path / "runs"
    jb = F.build(root)
    wb = F.build(root, bench="wb-office", tasks=("quarterly-budget-variance-analysis-for-finance", "travel-policy-memo"))
    (root / "zz" / "adapt").mkdir(parents=True)
    out = tmp_path / "ocp"
    code, stdout, err = _run(capsys, "adapt", "harness2", str(root), "--out", str(out))
    assert code == 0, err
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.ocp.json"))]
    assert len(docs) == len(jb.rollouts) + len(wb.rollouts)
    assert all(not [f for f in validate_doc(d) if f.level == "error"] for d in docs)
    wb_docs = [d for d in docs if d["run"]["task"]["repo"] == "harness2-wb-office"]
    assert len(wb_docs) == len(wb.rollouts) and all("cost" not in d["attempts"][0] for d in wb_docs)
    assert "skipped zz/adapt: unsupported bench folder" in stdout
    assert "WorkBuddy rollouts carry no cost" in stdout and "28 none" in stdout
