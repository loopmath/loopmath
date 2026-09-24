"""`loopmath posterior` end to end on a temporary store with a fake belief state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.output import EXIT_NO_FIT, EXIT_NOT_FOUND, EXIT_USER
from loopmath.views import posterior as P
from tests.views._posterior_helpers import FIT_ID, FIX, FakeState


def _plans() -> dict:
    return json.loads((FIX / "view-plans-binary.json").read_text())


def _config(workflow_id: str) -> dict:
    return next(c["config"] for c in _plans()["candidates"] if c["config"]["workflow"]["id"] == workflow_id)


def _ids() -> dict[str, str]:
    """The usual configuration (implement_review), one plan_implement_review and one solo from the plans fixture."""
    return {"usual": _plans()["usual"]["config"]["id"], "pir": _config("plan_implement_review")["id"],
            "solo": _config("solo")["id"]}


@pytest.fixture
def store(tmp_path, monkeypatch) -> tuple[Path, FakeState]:
    home = tmp_path / "home"
    (home / "recs").mkdir(parents=True)
    (home / "runs").mkdir()
    (home / "fits" / FIT_ID).mkdir(parents=True)
    (home / "recs" / "rec_01.json").write_text(json.dumps(_plans()))
    ids = _ids()
    rows = ([{"run": f"run_{i}", "task_type": "feature", "repo": "loopmath/loopmath", "config": ids["pir"]} for i in range(3)]
            + [{"run": f"run_o{i}", "task_type": "feature", "repo": "loopmath/loopmath", "config": ids["solo"]} for i in range(2)]
            + [{"run": "run_b", "task_type": "bug_fix", "repo": "acme/app", "config": ids["solo"]}]
            + [{"run": "run_unknown", "task_type": "feature", "repo": "loopmath/loopmath", "config": "cfg_000000000000"}])
    (home / "runs" / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    (home / "config.toml").write_text(f'[usual.feature]\n"*" = "{ids["usual"]}"\n')
    (home / "fits" / FIT_ID / "meta.json").write_text(json.dumps({
        "n_runs": {"prior": 1150, "user": 7},
        "rows": {"cost": {"sweep": 1549, "user": 61}, "success": {"sweep": 660, "user": 7}},
        "dropped": [{"run": "run_x", "reason": "asserted cost"}, {"run": "run_y", "reason": "asserted cost"},
                    {"run": "run_z", "reason": "open run"}],
        "fit_time_s": 12.5,
        "scales": {"cost": {"family": 0.62}, "success": {"family": 0.9}},
        "sensitivity": None}))
    state = FakeState()
    monkeypatch.setattr(P, "_load_state", lambda home_: state)
    return home, state


def _json(capsys, argv: list[str]) -> dict:
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    obj = json.loads(out)
    assert next(iter(obj)) == "schema" and obj["schema"] == P.SCHEMA
    return obj


def test_no_fit_exits_5_with_a_hint(tmp_path, capsys):
    code = cli.main(["posterior", "--home", str(tmp_path)])
    assert code == EXIT_NO_FIT
    assert "loopmath fit" in capsys.readouterr().err


def test_json_levels_task_workflows_and_data(store, capsys):
    home, state = store
    obj = _json(capsys, ["posterior", "--home", str(home), "--json"])
    assert list(obj["levels"])[:7] == list(P.SECTIONS)
    assert obj["fit"] == {"id": FIT_ID, "at": state.created_at, "n_runs": {"prior": 1150, "user": 7}}
    assert obj["task"] == {"type": "feature", "repo": "loopmath/loopmath", "subtype": None, "features": {}, "from": "store"}
    ids = _ids()
    configs = [w["config"] for w in obj["workflows"]]
    assert configs == [ids["usual"], ids["pir"], ids["solo"]]
    assert [w["origin"] for w in obj["workflows"]] == ["usual", "recorded", "recorded"]
    assert obj["workflow"] == obj["workflows"][0]
    assert {p[:2] for p in state.predicted} == {("feature", "loopmath/loopmath")}
    assert obj["data"]["dropped"] == [{"reason": "asserted cost", "n": 2}, {"reason": "open run", "n": 1}]
    assert obj["data"]["fit_time_s"] == 12.5 and obj["data"]["scales"]["cost"]["family"] == 0.62
    w = obj["workflow"]
    assert set(w["shares"]) == {n["id"] for n in w["graph"]["nodes"] if n["kind"] == "piece"}
    assert w["prediction"]["config"] == w["config"]


def test_level_and_head_filters(store, capsys):
    home, _ = store
    obj = _json(capsys, ["posterior", "--home", str(home), "--json", "--level", "model", "--head", "cost"])
    assert list(obj["levels"]) == ["model"] and obj["head"] == "cost" and list(obj["heads"]) == ["cost"]
    assert obj["levels"]["model"] and all(n["head"] == "cost" for n in obj["levels"]["model"])
    effects = obj["workflow"]["effects"]["pieces"]
    nodes = [n for refs in effects.values() for n in P.resolve_refs(refs, obj["levels"])]
    assert nodes and all(n["head"] == "cost" for n in nodes)


def test_unknown_head_is_a_user_error(store, capsys):
    home, _ = store
    assert cli.main(["posterior", "--home", str(home), "--head", "score:nope"]) == EXIT_USER
    assert "cost, success, gate" in capsys.readouterr().err


def test_unknown_task_type_is_a_user_error(store, capsys):
    home, _ = store
    assert cli.main(["posterior", "--home", str(home), "--type", "chores"]) == EXIT_USER


def test_type_and_repo_from_the_command_line(store, capsys):
    home, state = store
    obj = _json(capsys, ["posterior", "--home", str(home), "--json", "--type", "bug_fix", "--repo", "acme/app"])
    assert obj["task"]["from"] == "arguments" and obj["task"]["type"] == "bug_fix"
    assert [w["origin"] for w in obj["workflows"]] == ["recorded"]
    obj = _json(capsys, ["posterior", "--home", str(home), "--json", "--type", "docs", "--repo", "new/repo"])
    assert obj["workflows"] == [] and obj["workflow"] is None


def test_store_written_by_loopmath_run_and_config(tmp_path, capsys, monkeypatch):
    """Runs started and finished through lane 7's `loopmath run`, the usual set by `loopmath config set usual.feature CFG`
    (a plain id, not a table). A finished run has two index rows but counts once."""
    from loopmath.store import Store

    catalog = Path(P.__file__).parents[1] / "workflows" / "catalog"
    home = tmp_path / "home"
    ir = ["--set", "implement=claude-code:claude-opus-5-5:high", "--set", "review=codex:gpt-6-astra:xhigh"]
    starts = [("implement_review", ir, "usual"), ("implement_review", ir, "usual"),
              ("plan_implement_review", ["--set", "plan=claude-code:claude-opus-5-5:xhigh", *ir], "alternative"),
              ("plan_implement_review", ["--set", "plan=claude-code:claude-opus-5-5:xhigh", *ir], "alternative")]
    for wf, sets, source in starts:
        assert cli.main(["run", "start", "--home", str(home), "--type", "feature", "--repo", "acme/cli",
                         "--workflow", str(catalog / f"{wf}.toml"), *sets, "--source", source]) == 0
    rows = list(Store(home).index_rows().values())
    ir_id, pir_id = rows[0]["config"], rows[2]["config"]
    for row in rows[2:]:
        assert cli.main(["run", "finish", "--home", str(home), "--run", row["run"], "--no-fit"]) == 0
    assert cli.main(["config", "set", "--home", str(home), "usual.feature", ir_id]) == 0
    capsys.readouterr()
    assert len((home / "runs" / "index.jsonl").read_text().splitlines()) == 6
    assert len(P._index_rows(home)) == 4
    assert P._usual_config_id(home, P.Task(id="t", type="feature", repo="acme/cli")) == ir_id

    (home / "fits" / FIT_ID).mkdir(parents=True)
    state = FakeState()
    monkeypatch.setattr(P, "_load_state", lambda home_: state)
    obj = _json(capsys, ["posterior", "--home", str(home), "--json"])
    assert (obj["task"]["type"], obj["task"]["repo"], obj["task"]["from"]) == ("feature", "acme/cli", "store")
    assert [(w["config"], w["origin"]) for w in obj["workflows"]] == [(ir_id, "usual"), (pir_id, "recorded")]
    assert [p["id"] for p in obj["workflows"][1]["graph"]["nodes"] if p["kind"] == "piece"] == ["plan", "implement", "review"]

    assert cli.main(["config", "set", "--home", str(home), "usual.feature", pir_id]) == 0
    capsys.readouterr()
    obj = _json(capsys, ["posterior", "--home", str(home), "--json"])
    assert [(w["config"], w["origin"]) for w in obj["workflows"]] == [(pir_id, "usual"), (ir_id, "recorded")]


def test_workflow_by_configuration_id(store, capsys):
    home, state = store
    pir = _ids()["pir"]
    obj = _json(capsys, ["posterior", "--home", str(home), "--json", "--workflow", pir])
    assert [w["config"] for w in obj["workflows"]] == [pir] and obj["workflow"]["origin"] == "argument"
    assert [n["id"] for n in obj["workflow"]["graph"]["nodes"] if n["kind"] == "piece"] == ["plan", "implement", "review"]
    (loop,) = obj["workflow"]["loops"]
    assert loop["pieces"] == ["implement", "review"]


def test_workflow_not_found(store, capsys):
    home, _ = store
    assert cli.main(["posterior", "--home", str(home), "--workflow", "cfg_ffffffffffff"]) == EXIT_NOT_FOUND
    assert cli.main(["posterior", "--home", str(home), "--workflow", str(home / "missing.toml")]) == EXIT_NOT_FOUND


def test_workflow_from_an_ocp_run_document(store, capsys):
    home, _ = store
    goal = _config("plan_implement_review")
    ocp = {"ocp": "0.3", "run": {"configuration": {
        "id": "cfg_0cp0cp0cp0cp", "settings": {k: {**v, "model": {"id": v["model"], "raw": v["model"]}}
                                               for k, v in goal["settings"].items()},
        "workflow": {**goal["workflow"], "artifacts": [{"id": a, "kind": "file"} for a in goal["workflow"]["artifacts"]],
                     "edges": [{"from": a, "to": b} for a, b in goal["workflow"]["edges"]]}}}}
    (home / "runs" / "run_ocp.ocp.json").write_text(json.dumps(ocp))
    obj = _json(capsys, ["posterior", "--home", str(home), "--json", "--workflow", "cfg_0cp0cp0cp0cp"])
    assert obj["workflow"]["config"] == "cfg_0cp0cp0cp0cp"
    assert {n["setting"]["model"] for n in obj["workflow"]["graph"]["nodes"] if n["kind"] == "piece"} <= {
        s["model"] for s in goal["settings"].values()}


def test_ocp_control_gives_gates_loops_and_budget(store, capsys):
    """OCP v0.3 control (D29, D30): gate piece ids, a repair map, budget, rules in ext, rescue object."""
    home, _ = store
    goal = _config("plan_implement_review")
    control = {"gates": ["review"], "repair": {"review": "implement"}, "budget": 3,
               "rescue": {"kind": "configuration", "ref": "usual"},
               "ext": {"dev.loopmath.gate_rules": {"review": "command:lint"}}}
    doc = {"ocp": "0.3", "run": {"configuration": {
        "id": "cfg_0cp0cp0cp0c1", "settings": goal["settings"], "workflow": {**goal["workflow"], "control": control}}}}
    (home / "runs" / "run_ocp1.ocp.json").write_text(json.dumps(doc))
    w = _json(capsys, ["posterior", "--home", str(home), "--json", "--workflow", "cfg_0cp0cp0cp0c1"])["workflow"]
    assert w["graph"]["gates"] == [{"id": "g_review", "after": "review", "rule": "command:lint", "on_fail": "implement"}]
    assert w["graph"]["budget_rounds"] == 3
    (gate,) = w["gates"]
    assert gate["after"] == "review" and gate["on_fail"] == "implement"
    (loop,) = w["loops"]
    assert loop["pieces"] == ["implement", "review"]
    control.update(budget=0, gates=["review"], ext={})
    doc["run"]["configuration"]["id"] = "cfg_0cp0cp0cp0c2"
    (home / "runs" / "run_ocp2.ocp.json").write_text(json.dumps(doc))
    w = _json(capsys, ["posterior", "--home", str(home), "--json", "--workflow", "cfg_0cp0cp0cp0c2"])["workflow"]
    assert w["graph"]["budget_rounds"] == 1 and w["graph"]["gates"][0]["rule"] == "review_approve"


def test_ocp_workflow_given_as_a_catalog_ref(store, capsys):
    """An OCP run document may name its workflow `{ref, version}`; lane 4 resolves it from the catalog."""
    home, _ = store
    settings = {"implement": {"harness": "claude-code", "model": {"id": "claude-opus-5-5"}, "effort": "high"},
                "review": {"harness": "codex", "model": "gpt-6-astra", "effort": "xhigh"}}
    doc = {"ocp": "0.3", "run": {"configuration": {
        "id": "cfg_4ef4ef4ef4ef", "settings": settings, "workflow": {"ref": "implement_review", "version": 1}}}}
    (home / "runs" / "run_ref.ocp.json").write_text(json.dumps(doc))
    w = _json(capsys, ["posterior", "--home", str(home), "--json", "--workflow", "cfg_4ef4ef4ef4ef"])["workflow"]
    assert [n["id"] for n in w["graph"]["nodes"] if n["kind"] == "piece"] == ["implement", "review"]
    assert [(lp["from"], lp["to"]) for lp in w["loops"]] == [("implement", "review")]
    assert w["label"].startswith("implement_review: claude-opus-5-5/high")


def test_workflow_from_a_toml_file(store, capsys):
    """A file written by `workflows.format.dump_workflow` with its settings gives the same configuration; a file lane 4
    rejects, or one without settings, is a user error that names the problem."""
    from loopmath.types import Configuration
    from loopmath.workflows.format import dump_workflow

    home, _ = store
    pir = Configuration.from_dict(_config("plan_implement_review"))
    path = home / "mine.toml"
    argv = ["posterior", "--home", str(home), "--json", "--workflow", str(path)]
    path.write_text(dump_workflow(pir.workflow, pir.settings))
    w = _json(capsys, argv)["workflow"]
    assert (w["origin"], w["config"], w["label"]) == ("file", pir.id, pir.label())
    path.write_text(dump_workflow(pir.workflow, pir.settings).replace("width = 1", "width = 0", 1))
    assert cli.main(argv) == EXIT_USER
    assert "pieces[0].width: expected an integer >= 1, got 0" in capsys.readouterr().err
    path.write_text(dump_workflow(pir.workflow))
    assert cli.main(argv) == EXIT_USER
    assert "pieces without a setting: plan, implement, review" in capsys.readouterr().err


def test_workflow_file_in_lane4_format(store, capsys):
    """A catalog file plus `[settings.<piece>]` tables, read by lane 4's own loader."""
    import loopmath.workflows.format as fmt

    home, _ = store
    src = (Path(fmt.__file__).parent / "catalog" / "swarm.toml").read_text(encoding="utf-8")
    tables = "".join(f'\n[settings.{pid}]\nharness = "claude-code"\nmodel = "claude-opus-5-5"\neffort = "high"\n'
                     for pid in ("plan", "work", "review"))
    path = home / "swarm-mine.toml"
    path.write_text(src + tables)
    w = _json(capsys, ["posterior", "--home", str(home), "--json", "--workflow", str(path)])["workflow"]
    assert w["origin"] == "file" and w["config"].startswith("cfg_")
    assert [(p["id"], p.get("width", 1)) for p in w["graph"]["nodes"] if p["kind"] == "piece"] == [
        ("plan", 1), ("work", 3), ("review", 1)]
    assert [(lp["from"], lp["to"]) for lp in w["loops"]] == [("work", "review")]
    path.write_text(src)
    assert cli.main(["posterior", "--home", str(home), "--workflow", str(path)]) == EXIT_USER
    assert "pieces without a setting: plan, work, review" in capsys.readouterr().err


def test_html_path_and_json_embed_the_same_object(store, capsys, tmp_path):
    home, _ = store
    target = tmp_path / "out" / "posterior.html"
    assert cli.main(["posterior", "--home", str(home), "--json", "--html", str(target)]) == 0
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert captured.err.strip() == str(target)
    embedded = P.extract_data(target.read_text())
    assert embedded == printed


def test_html_default_path_under_views(store, capsys):
    home, _ = store
    assert cli.main(["posterior", "--home", str(home), "--html"]) == 0
    printed = capsys.readouterr().out.strip()
    path = Path(printed)
    assert path.parent == home / "views" and path.name.startswith("posterior-") and path.suffix == ".html"
    assert P.extract_data(path.read_text())["schema"] == P.SCHEMA
    assert not list((home / "views").glob("*.tmp"))


def test_terminal_summary(store, capsys):
    home, _ = store
    assert cli.main(["posterior", "--home", str(home)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert 3 < len(lines) <= 25
    assert lines[0].startswith("Current estimates (the posterior) from fit fit_20260923160000")
    assert lines[1].startswith("Graph: implement_review:")
    # D60: implement costs $1 per round over 1.4 rounds, review $2; shares from the run totals 1.4 and 2.8
    implement = next(line for line in lines if line.startswith("  implement:"))
    assert implement.startswith("  implement: $1.00 ($0.60 to $1.60) per round, $1.40 ($0.84 to $2.24) per run,")
    assert implement.endswith("33% of cost")
    assert next(line for line in lines if line.startswith("  review:")).endswith("67% of cost")


def test_cost_per_round_reaches_the_json(store, capsys):
    home, _ = store
    w = _json(capsys, ["posterior", "--home", str(home), "--json"])["workflow"]
    assert w["per_piece"]["review"]["cost_per_round"]["usd"]["mean"] == 2.0
    assert w["per_piece"]["review"]["cost"]["usd"]["mean"] == pytest.approx(2.8)
    assert w["shares"] == {"implement": pytest.approx(1 / 3, abs=1e-3), "review": pytest.approx(2 / 3, abs=1e-3)}


def test_reads_nothing_it_does_not_need_and_writes_only_the_page(store, capsys):
    home, _ = store
    before = sorted(p.relative_to(home) for p in home.rglob("*"))
    assert cli.main(["posterior", "--home", str(home), "--json"]) == 0
    assert sorted(p.relative_to(home) for p in home.rglob("*")) == before
