"""The whole chain on plugin eval results, in process and under temp stores: `adapt plugin-eval`, `run import
--finish`, `check-change from-runs` and `gate --runs`. The twin of the shipped plugin rows
(spec/examples/harness-runs/plugin-evals.jsonl) must give the same comparison as those rows do."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pefixture as F  # noqa: E402

from loopmath.cli import main  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "spec" / "examples" / "harness-runs" / "plugin-evals.jsonl"
SAME = ("n_pairs", "gain", "p_better", "cost_base", "cost_cand", "cpar_base", "cpar_cand", "verdict", "objective",
        "next_pair_value_usd", "improvement")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    (tmp_path / "user-home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))


def _json(capsys, *argv: str) -> tuple[int, dict, str]:
    capsys.readouterr()
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, (json.loads(out) if out.strip() else {}), err


def _chain(capsys, monkeypatch, tmp_path: Path, name: str, adapt: list[str], base: str, cand: str) -> dict:
    store, ocp, rows = tmp_path / f"{name}-store", tmp_path / f"{name}-ocp", tmp_path / f"{name}-rows.jsonl"
    monkeypatch.setenv("LOOPMATH_HOME", str(store))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / f"{name}-cache"))
    code, summary, err = _json(capsys, "adapt", *adapt, "--out", str(ocp), "--json",
                               *(["--rows", str(rows)] if adapt[0] == "plugin-eval" else []))
    assert code == 0, err
    code, _, err = _json(capsys, "run", "import", str(ocp), "--finish", "--no-fit", "--json")
    assert code == 0, err
    code, cc, err = _json(capsys, "check-change", "from-runs", "--base-label", f"arm:{base}", "--cand-label",
                          f"arm:{cand}", "--json")
    assert code == 0, err
    runs = rows if adapt[0] == "plugin-eval" else SHIPPED
    code, gate, err = _json(capsys, "gate", "--base", f"arm:{base}", "--cand", f"arm:{cand}", "--runs", str(runs),
                            "--rule", "accept", "--json")
    assert code == 0, err
    return {"summary": summary, "check": cc, "gate": gate}


def test_the_twin_of_the_shipped_plugin_rows_gives_their_comparison(capsys, monkeypatch, tmp_path):
    shipped = _chain(capsys, monkeypatch, tmp_path, "shipped", ["harness-runs", str(SHIPPED)],
                     "without", "with-lint-plugin")
    twin = F.write(tmp_path / "twin", F.twin_suite())
    costs = F.write_costs(tmp_path / "twin-costs.jsonl", F.TWIN_COSTS)
    got = _chain(capsys, monkeypatch, tmp_path, "twin",
                 ["plugin-eval", str(twin.path), "--cost-from", str(costs), "--model", "claude-sonnet-5-5",
                  "--type", "bug_fix", "--repo", "acme/widgets"], "without", "with")
    for k in SAME:
        assert got["check"]["comparison"][k] == shipped["check"]["comparison"][k], k
    for k in ("action", "n_more", "stop"):
        assert got["gate"][k] == shipped["gate"][k], k
    for k in ("p_better", "gain", "n_pairs"):
        assert got["gate"]["comparison"][k] == shipped["gate"]["comparison"][k], k
    # the shipped rows' own numbers under 0.3.1 (`loopmath gate ... --runs plugin-evals.jsonl --json`)
    c = got["gate"]["comparison"]
    assert c["n_pairs"] == 3 and c["p_better"] == pytest.approx(0.80825, abs=1e-5)
    assert c["gain"]["mean"] == pytest.approx(0.201881, abs=1e-6)
    assert got["check"]["comparison"]["verdict"] == "more" and got["gate"]["action"] == shipped["gate"]["action"]


def test_without_a_cost_file_the_twin_answers_on_outcomes_alone(capsys, monkeypatch, tmp_path):
    twin = F.write(tmp_path / "twin", F.twin_suite())
    got = _chain(capsys, monkeypatch, tmp_path, "twin", ["plugin-eval", str(twin.path)], "without", "with")
    comp = got["check"]["comparison"]
    assert comp["cost_base"] is None and comp["cost_cand"] is None and comp["cpar_base"] is None
    assert comp["objective"] == "gain"
    assert comp["gain"]["mean"] == pytest.approx(0.201881, abs=0.05)   # the same outcomes, no cost model
    assert got["summary"]["cost"]["none"] == 6


def test_the_docs_built_suite_gives_a_decision_with_ranges_and_the_next_commands(capsys, monkeypatch, tmp_path):
    built = F.write(tmp_path / "lint-helper", F.docs_suite(), manifest={"name": "lint-helper", "version": "0.4.0"})
    got = _chain(capsys, monkeypatch, tmp_path, "docs", ["plugin-eval", str(built.path)], "without", "with")
    comp = got["check"]["comparison"]
    assert comp["n_pairs"] == 11   # 12 pairs, one with an unscored run
    assert comp["verdict"] in ("adopt", "reject", "more")
    assert set(comp["gain"]["bands"]) == {"80", "95"}
    assert got["check"]["cap"]["max_pairs"] == 24 and got["check"]["cap"]["from"] == "auto"   # above 12 pairs
    pe = got["summary"]["plugin_eval"]
    assert pe["pairs"] == 12
    check = next(line for line in pe["next"] if line.startswith("loopmath check-change from-runs "))
    assert "--max-pairs" not in check
    gate = next(line for line in pe["next"] if line.startswith("loopmath gate "))
    assert "--runs" in gate and "--r-max 24" in gate and "--rule accept" in gate
    code, out, err = _json(capsys, *shlex.split(gate)[1:], "--json")   # the gate line as printed runs
    assert code == 0, err
    assert out["policy"]["r_max"] == 24 and out["comparison"]["n_pairs"] == 11 and out["stop"] != "cap", out


def test_grades_with_replicates_are_compared_case_by_case_and_the_gate_line_says_it_pairs_runs(
        capsys, monkeypatch, tmp_path):
    built = F.write(tmp_path / "lint-helper", F.docs_suite(), manifest={"name": "lint-helper", "version": "0.4.0"})
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "score-store"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "score-cache"))
    ocp, rows = tmp_path / "score-ocp", tmp_path / "score-rows.jsonl"
    code, summary, err = _json(capsys, "adapt", "plugin-eval", str(built.path), "--outcome", "score", "--out",
                               str(ocp), "--rows", str(rows), "--json")
    assert code == 0, err
    nxt = summary["plugin_eval"]["next"]
    check = next(line for line in nxt if line.startswith("loopmath check-change from-runs "))
    gate = next(i for i, line in enumerate(nxt) if line.startswith("loopmath gate "))
    assert nxt[gate + 1] == "# gate pairs runs, not cases: its answer is about these cases; check-change from-runs " \
                            "compares the 4 cases, one pair each", nxt
    assert _json(capsys, "run", "import", str(ocp), "--finish", "--no-fit", "--json")[0] == 0
    code, cc, err = _json(capsys, *shlex.split(check)[1:], "--json")   # the from-runs line as printed
    assert code == 0, err
    assert (cc["unit"]["unit"], cc["unit"]["from"], cc["unit"]["most_runs"]) == ("case", "auto", 3), cc["unit"]
    assert cc["comparison"]["n_pairs"] == 4 and (cc["cap"]["max_pairs"], cc["cap"]["pairs"]) == (6, 4), cc["cap"]
    code, cr, err = _json(capsys, *shlex.split(check)[1:], "--unit", "run", "--json")
    assert code == 0 and cr["comparison"]["n_pairs"] == 11 and cr["cap"]["max_pairs"] == 24, err


def test_a_withheld_case_never_reaches_from_runs_or_the_gate(capsys, monkeypatch, tmp_path):
    doc = F.docs_suite()
    del doc["cases"][1]["aggregates"]["delta"]   # rename-config-key: both arms, no delta: not comparable
    built = F.write(tmp_path / "held", doc, manifest={"name": "lint-helper", "version": "0.4.0"})
    got = _chain(capsys, monkeypatch, tmp_path, "held", ["plugin-eval", str(built.path)], "without", "with")
    assert got["check"]["comparison"]["n_pairs"] == 8 and got["check"]["cap"]["pairs"] == 9   # 9 pairs, 1 unscored
    assert got["gate"]["comparison"]["n_pairs"] == 8
    assert [w["case"] for w in got["summary"]["plugin_eval"]["withheld"]] == ["rename-config-key"]
