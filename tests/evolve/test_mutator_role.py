"""Exported runs carry the mutator role through to the fit: `posterior --level role` lists it, and
`recommend --role mutator` prices each model per improvement on simulated runs with known rates.

These need `stages` on harness-runs rows and skip, saying so, without them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oefixture as OE  # noqa: E402
import simdb  # noqa: E402

from loopmath.cli import main  # noqa: E402
from loopmath.harness_runs import rows as hr_rows  # noqa: E402

pytestmark = pytest.mark.skipif("stages" not in hr_rows.KNOWN, reason="needs stages on harness-runs rows")

# Per query: chance the patch applies x chance the child beats its parent, and dollars per query. Pro improves
# most (0.30), flash is cheapest per improvement (0.002 / 0.10 = $0.02, vs $0.10 and $0.20).
MODELS = [simdb.Model("gemini-2.5-flash", q=0.20, apply=0.5, usd=0.002),
          simdb.Model("gemini-2.5-pro", q=0.30, apply=1.0, usd=0.03),
          simdb.Model("gpt-5-mini", q=0.10, apply=0.5, usd=0.01)]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "user-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "lm-home"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "lm-cache"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def store(tmp_path, capsys):
    results, truth = simdb.shinka_run(tmp_path / "sim", MODELS, n=360, seed=7)
    out = tmp_path / "ocp"
    code, _, err = _run(capsys, "adapt", "shinkaevolve", str(results), "--out", str(out))
    assert code == 0, err
    code, _, err = _run(capsys, "run", "import", str(out), "--finish", "--no-fit", "--json")
    assert code == 0, err
    code, _, err = _run(capsys, "fit", "--json")
    assert code == 0, err
    return out, truth


def test_an_exported_run_is_one_mutate_piece_with_role_mutator(store):
    out, _ = store
    doc = json.loads(next(out.glob("*.ocp.json")).read_text(encoding="utf-8"))
    wf = doc["run"]["configuration"]["workflow"]
    assert [(p["id"], p["role"]) for p in wf["pieces"]] == [("mutate", "mutator")]
    assert wf["id"] == "stages-mutate"


def test_posterior_by_role_lists_the_mutator(store, capsys):
    code, out, err = _run(capsys, "posterior", "--level", "role", "--json")
    assert code == 0, err
    keys = {e["key"] for e in json.loads(out)["levels"]["role"]}
    assert "mutator" in keys


def test_recommend_by_role_prices_each_model_per_improvement(store, capsys):
    _, truth = store
    code, out, err = _run(capsys, "recommend", "--role", "mutator", "--json")
    assert code == 0, err
    obj = json.loads(out)
    assert obj["schema"] == "loopmath.recommend.role/1" and obj["role"] == "mutator"
    assert obj["task"]["id"] == "circle_packing" and obj["unit"] == "per query"
    notes = " ".join(obj["notes"])
    assert "per answered patch query in a settled generation" in notes and "no reply" in notes
    assert obj["accounting"]["usd_in_no_row"] >= 0 and obj["accounting"]["generations"] > 0
    assert obj["label_q"] == 0.95 and "outcome.q.reported" in notes
    models = {m["model"]: m for m in obj["models"]}
    assert set(models) == {m.name for m in MODELS}
    for name, m in models.items():
        p, cpi = m["p_improve"], m["usd_per_improvement"]
        assert 0 < p["lo"] <= p["mean"] <= p["hi"] < 1 and p["level"] == 0.8
        assert cpi["lo"] <= cpi["median"] <= cpi["hi"] and cpi["mean"] > 0
        assert m["usd_per_call"]["mean"] > 0 and m["runs"] == truth.calls[name]
    assert sum(m["p_best"] for m in models.values()) == pytest.approx(1.0, abs=1e-6)
    assert obj["pick"]["model"] == "gemini-2.5-flash"         # $0.02 per improvement vs $0.10 and $0.20
    assert "usd_per_improvement" in obj["pick"]["rule"]
    best_chance = max(models.values(), key=lambda m: m["p_improve"]["mean"])
    assert best_chance["model"] == "gemini-2.5-pro"


def test_exact_labels_raise_each_chance_and_drop_the_label_note(store, capsys):
    # The exporters compute the improvement from recorded scores; read at q 0.95, a 5% flip rate is assumed and a
    # chance near 5% reads low. Set to 1 and refit, the fit reads them as exact.
    code, out, err = _run(capsys, "recommend", "--role", "mutator", "--json")
    assert code == 0, err
    before = {m["model"]: m["p_improve"]["mean"] for m in json.loads(out)["models"]}
    for argv in (["config", "set", "outcome.q.reported", "1"], ["fit", "--json"]):
        code, _, err = _run(capsys, *argv)
        assert code == 0, err
    code, out, err = _run(capsys, "recommend", "--role", "mutator", "--json")
    assert code == 0, err
    obj = json.loads(out)
    assert obj["label_q"] == 1.0 and "outcome.q.reported" not in " ".join(obj["notes"])
    after = {m["model"]: m["p_improve"]["mean"] for m in obj["models"]}
    assert all(after[name] > before[name] for name in before), (before, after)


def test_recommend_by_role_names_the_roles_it_has_when_asked_for_another(store, capsys):
    code, out, err = _run(capsys, "recommend", "--role", "evaluator", "--json")
    assert code == 1
    assert "mutator" in json.loads(out)["error"]


def _import(capsys, *exports: list[str]) -> None:
    for argv in exports:
        code, _, err = _run(capsys, "adapt", *argv)
        assert code == 0, err
        out = argv[argv.index("--out") + 1]
        code, _, err = _run(capsys, "run", "import", out, "--finish", "--no-fit", "--json")
        assert code == 0, err


def test_a_role_in_more_than_one_unit_needs_unit_and_reports_the_one_chosen(tmp_path, capsys):
    # OpenEvolve runs are per stored child, ShinkaEvolve query rows per query: one chance cannot stand for both.
    oe, _ = simdb.openevolve_run(tmp_path / "oe", MODELS, n=60, seed=3)
    sh, _ = simdb.shinka_run(tmp_path / "sh", MODELS, n=90, seed=5)
    table = OE.price_table(tmp_path / "prices.toml", {m.name: (0.5, 2.0) for m in MODELS})
    _import(capsys, ["openevolve", str(oe), "--prices", str(table), "--out", str(tmp_path / "a")],
            ["shinkaevolve", str(sh), "--out", str(tmp_path / "b")])
    code, out, err = _run(capsys, "fit", "--json")
    assert code == 0, err
    code, out, _ = _run(capsys, "recommend", "--role", "mutator", "--json")
    assert code == 1
    error = json.loads(out)["error"]
    assert "per query" in error and "per stored child" in error and "--unit" in error
    code, out, err = _run(capsys, "recommend", "--role", "mutator", "--unit", "query", "--json")
    assert code == 0, err
    obj = json.loads(out)
    assert obj["unit"] == "per query" and {m["harness"] for m in obj["models"]} == {"shinkaevolve"}
    assert "per stored child" in " ".join(obj["notes"])
    code, out, err = _run(capsys, "recommend", "--role", "mutator", "--unit", "stored-child", "--json")
    assert code == 0, err
    obj = json.loads(out)
    assert obj["unit"] == "per stored child" and {m["harness"] for m in obj["models"]} == {"openevolve"}
    assert "survivors" in " ".join(obj["notes"])


def test_query_and_children_only_rows_of_one_configuration_are_refused_as_pooled(tmp_path, capsys):
    # Both exports of one folder give the same configuration (same harness, model and effort), so the fit has
    # one chance for both units: no --unit can separate them.
    sh, _ = simdb.shinka_run(tmp_path / "sh", MODELS, n=60, seed=5)
    _import(capsys, ["shinkaevolve", str(sh), "--out", str(tmp_path / "q")],
            ["shinkaevolve", str(sh), "--children-only", "--out", str(tmp_path / "c")])
    for extra in ([], ["--unit", "query"], ["--unit", "evaluated child"]):
        code, out, _ = _run(capsys, "recommend", "--role", "mutator", *extra, "--json")
        assert code == 1, extra
        error = json.loads(out)["error"]
        assert "per query" in error and "per evaluated child" in error, extra
    code, out, _ = _run(capsys, "recommend", "--role", "mutator", "--unit", "query", "--json")
    assert "pooled" in json.loads(out)["error"]


def _rows(path: Path, runs: list[tuple[str, list[tuple[str, str]], bool]]) -> Path:
    # Generic runs rows (harness_runs/rows.py STAGE_KNOWN): each a model and its stages as (role, piece).
    rows = [{"schema": "loopmath.harness-runs/0.1", "id": f"r{i}", "task_id": "piece-ids",
             "task": {"type": "research", "repo": "piece-ids"}, "experiment": "piece-ids", "agent": "openevolve",
             "model": model, "status": "done", "outcome": {"accepted": ok},
             "stages": [{"role": role, "piece": piece, "status": "done",
                         "cost": {"usd": 0.01, "usd_source": "fixture"}} for role, piece in stages]}
            for i, (model, stages, ok) in enumerate(runs)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _import_rows(capsys, rows: Path, out: Path) -> None:
    for argv in (["adapt", "harness-runs", str(rows), "--out", str(out)],
                 ["run", "import", str(out), "--finish", "--no-fit", "--json"]):
        code, _, err = _run(capsys, *argv)
        assert code == 0, err


def test_one_role_under_different_piece_ids_reads_each_configuration_s_own_piece(tmp_path, capsys):
    # Two exports may name the mutator's piece differently; each configuration keeps its own.
    runs = ([("gpt-5-mini", [("mutator", "mutate-a")], i % 2 == 0) for i in range(5)]
            + [("gemini-2.5-flash", [("mutator", "mutate-b")], i % 3 == 0) for i in range(4)]
            + [("gpt-5-mini", [("mutator", "mutate-b")], i == 0) for i in range(2)])
    _import_rows(capsys, _rows(tmp_path / "rows.jsonl", runs), tmp_path / "ocp")
    code, _, err = _run(capsys, "fit", "--json")
    assert code == 0, err
    code, out, err = _run(capsys, "recommend", "--role", "mutator", "--json")
    assert code == 0, err
    obj = json.loads(out)
    got = {(m["model"], m["piece"]): m["runs"] for m in obj["models"]}
    assert got == {("gpt-5-mini", "mutate-a"): 5, ("gemini-2.5-flash", "mutate-b"): 4,
                   ("gpt-5-mini", "mutate-b"): 2}
    [chosen] = [m for m in obj["models"] if m["config"] == obj["pick"]["config"]]
    assert (obj["pick"]["model"], obj["pick"]["piece"]) == (chosen["model"], chosen["piece"])
    code, out, err = _run(capsys, "recommend", "--role", "mutator")
    assert code == 0, err
    assert "gpt-5-mini (mutate-a)" in out and "gpt-5-mini (mutate-b)" in out
    code, out, err = _run(capsys, "recommend", "--role", "mutator", "--models", "gemini-2.5-flash,o4-mini",
                          "--json")
    assert code == 0, err
    got = {(m["model"], m["piece"]): m["runs"] for m in json.loads(out)["models"]}
    assert got == {("gemini-2.5-flash", "mutate-b"): 4, ("o4-mini", "mutate-a"): 0}   # on the most-run one


def test_a_workflow_of_more_than_one_piece_is_refused_not_priced_as_one_step(tmp_path, capsys):
    # The fit's chance and dollars are a whole run's; with a second piece in the run they are not the mutator's.
    runs = ([("gpt-5-mini", [("mutator", "mutate")], True)] * 3
            + [("gpt-5-mini", [("mutator", "mutate"), ("reviewer", "review")], True)] * 2)
    _import_rows(capsys, _rows(tmp_path / "rows.jsonl", runs), tmp_path / "ocp")
    code, out, _ = _run(capsys, "recommend", "--role", "mutator", "--json")
    assert code == 1
    error = json.loads(out)["error"]
    assert "2 of the mutator runs" in error and "more than one piece" in error and "whole run" in error


def test_role_flags_stay_off_the_default_recommend(capsys):
    code, _, err = _run(capsys, "recommend", "--unit", "query", "--type", "bug_fix", "--repo", "acme/api")
    assert code == 2 and "--role" in err
    code, _, err = _run(capsys, "recommend", "--role", "mutator", "--goal", "p80")
    assert code == 2 and "--goal" in err
