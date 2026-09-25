"""`loopmath workflows list|show|validate|diff`, the show page and package exports (spec 02 section 2; lane 04)."""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import pytest

from loopmath.cli import main
from loopmath.store.home import Store
from loopmath.types import DEFAULT_RULE, Setting, Task
from loopmath.workflows.format import catalog, dump_workflow
from loopmath.workflows.graphview import workflow_graph
from loopmath.workflows.ids import make_config
from loopmath.workflows.ocp import configuration_to_ocp

ROOT = Path(__file__).resolve().parents[2]
NATIVE_B = ROOT / "tests" / "fixtures" / "graph" / "roundtrip" / "native-swarm-b.json"

IMPL = Setting("claude-code", "claude-opus-5-5", "high")
REVIEW = Setting("codex", "gpt-6-astra", "xhigh")

MY_REVIEW = """\
id = "my_review"
title = "Implement, two reviewers"
artifacts = [{id = "issue", kind = "issue"}, {id = "diff", kind = "diff"}, {id = "v1", kind = "verdict"}, {id = "v2", kind = "verdict"}]
edges = ["issue -> implement", "implement -> diff", "diff -> r1", "diff -> r2", "r1 -> v1", "r2 -> v2"]

[[pieces]]
id = "implement"
role = "implementer"

[[pieces]]
id = "r1"
role = "reviewer"

[[pieces]]
id = "r2"
role = "reviewer"

[control]
budget_rounds = 2
rescue = "person"

[[control.gates]]
after = "r1"
on_fail = "implement"

[settings.implement]
harness = "claude-code"
model = "claude-opus-5-5"
effort = "high"

[settings.r1]
harness = "codex"
model = "gpt-6-astra"
effort = "xhigh"

[settings.r2]
harness = "claude-code"
model = "claude-fable-5-1"
effort = "high"
"""

BAD = """\
id = "bad"
edges = ["a -> b"]

[[pieces]]
id = "a"
role = "implementer"
"""


def run(capsys, *argv):
    code = main(["workflows", *argv])
    out, err = capsys.readouterr()
    return code, out, err


def run_json(capsys, *argv):
    code, out, err = run(capsys, *argv, "--json")
    obj = json.loads(out)  # exactly one object on stdout
    assert next(iter(obj)) == "schema"
    return code, obj, err


@pytest.fixture
def home(tmp_path):
    root = tmp_path / "store"
    (root / "workflows").mkdir(parents=True)
    (root / "workflows" / "mine.toml").write_text(MY_REVIEW, encoding="utf-8")
    (root / "workflows" / "bad.toml").write_text(BAD, encoding="utf-8")
    return root


def ir_config():
    return make_config(catalog()["implement_review"], {"implement": IMPL, "review": REVIEW})


# ------------------------------------------------------------------ list

def test_list_text_and_json(capsys, home):
    code, out, _ = run(capsys, "list", "--home", str(home))
    assert code == 0
    lines = out.splitlines()
    assert [ln.split()[0] for ln in lines[:6]] == list(catalog())
    assert any(ln.startswith("my_review") and "user" in ln for ln in lines)
    assert any(ln.startswith("bad") and "INVALID" in ln for ln in lines)
    assert len(lines) <= 25

    code, obj, _ = run_json(capsys, "list", "--home", str(home))
    assert code == 0 and obj["schema"] == "loopmath.workflows.list/1"
    rows = {r["id"]: r for r in obj["workflows"]}
    assert [r["id"] for r in obj["workflows"][:6]] == list(catalog())
    assert rows["swarm"]["origin"] == "catalog" and rows["swarm"]["shape"] == "swarm" and rows["swarm"]["valid"]
    assert rows["my_review"]["config"].startswith("cfg_") and rows["my_review"]["valid"]
    assert not rows["bad"]["valid"] and rows["bad"]["errors"]


def test_list_empty_store(capsys, tmp_path):
    code, out, _ = run(capsys, "list", "--home", str(tmp_path))
    assert code == 0 and "no user workflows" in out


def test_list_and_show_name_each_piece_before_its_role(capsys, home):
    # the piece name is what `run start --set PIECE=...` takes; the role alone does not say it
    _, out, _ = run(capsys, "list", "--home", str(home))
    rows = {ln.split()[0]: ln for ln in out.splitlines()}
    assert rows["best_of_n"].endswith("catalog  implement (implementer) x3 -> select (referee)")
    assert rows["my_review"].endswith("user     implement (implementer) -> r1 (reviewer) -> r2 (reviewer)")
    _, out, _ = run(capsys, "show", "mine", "--home", str(home))
    assert "  implement (implementer)  claude-opus-5-5/high (claude-code)" in out.splitlines()
    assert "  r1 (reviewer)            gpt-6-astra/xhigh (codex)" in out.splitlines()


# ------------------------------------------------------------------ show

def test_show_catalog_text(capsys, home):
    code, out, _ = run(capsys, "show", "plan_implement_review", "--home", str(home))
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("plan_implement_review: ") and "source: catalog" in lines[1]
    assert any(ln.strip().startswith("implement") and "implementer" in ln for ln in lines)
    assert any(ln.startswith("artifacts: ") and "plan_doc (plan)" in ln for ln in lines)
    assert any(ln.startswith("edges: ") for ln in lines)
    assert "gate g_review: after review, review_approve, on fail implement" in lines
    assert lines[-1] == "round limit: 3 (up to 2 repairs); rescue: redo_usual"
    assert len(lines) <= 25


def test_show_json(capsys, home):
    code, obj, _ = run_json(capsys, "show", "my_review", "--home", str(home))
    assert code == 0 and obj["schema"] == "loopmath.workflows.show/1"
    assert obj["origin"] == "user" and obj["path"].endswith("mine.toml")
    assert obj["workflow"]["id"] == "my_review" and obj["settings"]["r1"]["model"] == "gpt-6-astra"
    assert obj["config"].startswith("cfg_") and obj["ocp"]["configuration_id"] == obj["config"]
    assert obj["ocp"]["workflow"]["control"]["gates"] == ["r1"]
    assert obj["html"] is None


def test_show_by_stem_and_path(capsys, home):
    assert run(capsys, "show", "mine", "--home", str(home))[0] == 0
    code, obj, _ = run_json(capsys, "show", str(home / "workflows" / "mine.toml"), "--home", str(home))
    assert code == 0 and obj["origin"] == "file" and obj["workflow"]["id"] == "my_review"


def test_show_exit_codes(capsys, home):
    code, out, err = run(capsys, "show", "nope", "--home", str(home))
    assert code == 2 and out == "" and "nope" in err
    code, _, err = run(capsys, "show", "cfg_000000000000", "--home", str(home))
    assert code == 2 and "cfg_000000000000" in err
    code, _, err = run(capsys, "show", "bad", "--home", str(home))
    assert code == 1 and "invalid" in err and "'b' is not a piece or an artifact" in err
    junk = home / "junk.json"
    junk.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    assert run(capsys, "show", str(junk), "--home", str(home))[0] == 1


def test_show_says_what_it_expected_from_a_folder_or_jsonl(capsys, home, tmp_path):
    (tmp_path / "empty").mkdir()
    code, _, err = run(capsys, "show", str(tmp_path / "empty"), "--home", str(home))
    assert code == 1 and "a folder without config.json; expected a workflow TOML" in err
    assert "No such file" not in err
    lines = tmp_path / "attempts.jsonl"
    lines.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    code, _, err = run(capsys, "show", str(lines), "--home", str(home))
    assert code == 1 and "not one JSON document (Extra data: line 2 column 1" in err and "run folder" in err


def test_show_config_from_rec_and_from_run(capsys, home):
    # A rec in the recommend contract (the fixture lane 06's output follows), read through lane 07's store.
    from tests.store.store_helpers import EXPLORE_CFG, REC_ID, USUAL_CFG, install_rec

    install_rec(home)
    for cid in (USUAL_CFG, EXPLORE_CFG):
        code, obj, _ = run_json(capsys, "show", cid, "--home", str(home))
        assert code == 0 and obj["origin"] == "store" and obj["config"] == cid and obj["note"] == f"from {REC_ID}"

    # A run opened by lane 07's store, found through its index.
    other = make_config(catalog()["solo"], {"implement": REVIEW})
    run_id = Store(home).new_run(Task(id="tsk_1", type="docs", repo="r"), other, source="usual", rec=None,
                                 slate=None, rule=DEFAULT_RULE, base_commit=None)
    code, obj, _ = run_json(capsys, "show", other.id, "--home", str(home))
    assert code == 0 and obj["config"] == other.id and obj["settings"]["implement"]["model"] == "gpt-6-astra"
    assert obj["path"].endswith(f"{run_id}.ocp.json") and obj["note"] == f"from {run_id}"


def test_show_infers_a_graph_file(capsys, home):
    code, obj, _ = run_json(capsys, "show", str(NATIVE_B), "--home", str(home))
    assert code == 0 and obj["origin"] == "inferred" and obj["shape"] == "swarm"
    assert obj["note"].startswith("inferred from graph, confidence ")


def _page_data(page: str) -> dict:
    found = re.search(r'<script[^>]*id="data"[^>]*>(.*?)</script>', page, re.S)
    assert found, "the page embeds no data"
    return json.loads(found.group(1))


def test_show_html_writes_lane12_workflow_page(capsys, home, tmp_path):
    target = tmp_path / "w.html"
    code, obj, _ = run_json(capsys, "show", "my_review", "--home", str(home), "--html", str(target))
    assert code == 0 and obj["renderer"] == "views.common.graph_page"
    page = target.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "loopmath workflow: my_review" in page
    data = _page_data(page)
    assert data["schema"] == "loopmath.view.workflow/1" and data["graph"]["config"] == obj["config"]
    assert [n["id"] for n in data["graph"]["nodes"] if n["kind"] == "piece"] == ["implement", "r1", "r2"]

    code, out, _ = run(capsys, "show", "swarm", "--home", str(home), "--html", str(tmp_path / "s.html"))
    assert code == 0 and out.strip() == str(tmp_path / "s.html")

    code, obj, err = run_json(capsys, "show", "implement_review", "--home", str(home), "--html")
    default = Path(obj["html"])
    assert code == 0 and default.parent == home / "views" and default.name.startswith("workflows-show-")
    assert default.exists() and str(default) in err  # the path goes to stderr so stdout stays one JSON object


def test_workflow_graph_matches_fixture_format():
    cfg = ir_config()
    g = workflow_graph(cfg.workflow, cfg.settings, cfg)
    assert set(g) == {"config", "label", "nodes", "edges", "gates", "workflow", "title", "budget_rounds"}
    assert g["config"] == cfg.id and g["workflow"] == "implement_review" and g["budget_rounds"] == 3
    pieces = [n for n in g["nodes"] if n["kind"] == "piece"]
    assert [n["id"] for n in pieces] == ["implement", "review"]
    assert pieces[0]["setting"]["model"] == "claude-opus-5-5"
    assert {"issue", "diff"} <= {n["id"] for n in g["nodes"] if n["kind"] == "artifact"}
    assert {"from": "implement", "to": "diff"} in g["edges"]
    assert g["gates"] == [{"id": "g_review", "after": "review", "rule": "review_approve", "on_fail": "implement"}]
    wide = workflow_graph(catalog()["swarm"])
    assert wide["config"] is None and next(n for n in wide["nodes"] if n["id"] == "work")["width"] == 3


# ------------------------------------------------------------------ validate

def test_validate_ok_and_catalog_files(capsys, home):
    code, out, _ = run(capsys, "validate", str(home / "workflows" / "mine.toml"), "--home", str(home))
    assert code == 0 and ": ok, workflow my_review, config cfg_" in out
    for name in catalog():
        path = ROOT / "src" / "loopmath" / "workflows" / "catalog" / f"{name}.toml"
        code, obj, _ = run_json(capsys, "validate", str(path), "--home", str(home))
        assert code == 0 and obj["valid"] and obj["shape"] == name and obj["config"] is None, name
        assert obj["warnings"] == [], name


def test_validate_errors_and_missing(capsys, home, tmp_path):
    code, obj, _ = run_json(capsys, "validate", str(home / "workflows" / "bad.toml"), "--home", str(home))
    assert code == 1 and obj["schema"] == "loopmath.workflows.validate/1" and not obj["valid"]
    assert obj["errors"] == ["edge a -> b: 'b' is not a piece or an artifact"]
    code, out, _ = run(capsys, "validate", str(home / "workflows" / "bad.toml"), "--home", str(home))
    assert code == 1 and "invalid" in out and "error: edge a -> b" in out
    broken = tmp_path / "broken.toml"
    broken.write_text("id = [", encoding="utf-8")
    code, obj, _ = run_json(capsys, "validate", str(broken), "--home", str(home))
    assert code == 1 and obj["errors"] and obj["workflow"] is None
    code, _, err = run(capsys, "validate", str(tmp_path / "nothere.toml"), "--home", str(home))
    assert code == 2 and "no such file" in err


def test_validate_settings_and_warnings(capsys, home, tmp_path):
    partial = tmp_path / "partial.toml"
    partial.write_text(MY_REVIEW.split("[settings.r1]")[0], encoding="utf-8")
    code, obj, _ = run_json(capsys, "validate", str(partial), "--home", str(home))
    assert code == 0 and obj["config"] is None
    assert any("no settings for r1, r2" in w for w in obj["warnings"])

    no_model = tmp_path / "no_model.toml"
    no_model.write_text(MY_REVIEW.replace('model = "gpt-6-astra"\n', ""), encoding="utf-8")
    code, obj, _ = run_json(capsys, "validate", str(no_model), "--home", str(home))
    assert code == 1 and any("r1" in e and "model" in e for e in obj["errors"])

    changed = catalog()["implement_review"]
    clash = tmp_path / "clash.toml"
    clash.write_text(dump_workflow(changed).replace("budget_rounds = 3", "budget_rounds = 4"), encoding="utf-8")
    code, obj, _ = run_json(capsys, "validate", str(clash), "--home", str(home))
    assert code == 0 and any("is a catalog shape" in w for w in obj["warnings"])


def test_validate_warns_on_settings_that_cannot_run(capsys, home, tmp_path):
    # a codex piece at effort "ultra", and codex asked to run a Claude model: still valid, but warned
    odd = tmp_path / "odd.toml"
    odd.write_text(MY_REVIEW.replace('effort = "xhigh"', 'effort = "ultra"', 1)
                   .replace('harness = "claude-code"', 'harness = "codex"', 1), encoding="utf-8")
    code, obj, _ = run_json(capsys, "validate", str(odd), "--home", str(home))
    assert code == 0 and obj["valid"] and obj["config"]
    assert "setting 'implement': codex does not run claude-opus-5-5; claude-code does" in obj["warnings"]
    assert "setting 'r1': effort 'ultra' is not one codex offers (low, medium, high, xhigh, max)" in obj["warnings"]
    code, obj, _ = run_json(capsys, "validate", str(home / "workflows" / "mine.toml"), "--home", str(home))
    assert code == 0 and not any(w.startswith("setting ") for w in obj["warnings"])
    # config overrides: efforts.codex offers ultra here, and this home runs claude-opus-5-5 in codex
    (home / "config.toml").write_text('[efforts]\ncodex = ["high", "ultra"]\n\n'
                                      '[harnesses]\n"claude-opus-5-5" = "codex"\n', encoding="utf-8")
    code, obj, _ = run_json(capsys, "validate", str(odd), "--home", str(home))
    assert code == 0 and not any(w.startswith("setting ") for w in obj["warnings"])


# ------------------------------------------------------------------ diff

def test_diff_catalog_shapes(capsys, home):
    code, out, _ = run(capsys, "diff", "solo", "plan_implement_review", "--home", str(home))
    lines = out.splitlines()
    assert code == 0 and lines[0] == "shape: solo to plan_implement_review"
    assert "planner added" in lines and "reviewer added" in lines
    assert "artifact added: plan_doc (plan)" in lines and "edge added: plan_doc -> implement" in lines
    code, out, _ = run(capsys, "diff", "swarm", "swarm", "--home", str(home))
    assert code == 0 and out.strip() == "no differences between swarm and swarm"


def test_diff_configurations_json(capsys, home, tmp_path):
    usual = tmp_path / "usual.json"
    usual.write_text(json.dumps(configuration_to_ocp(ir_config())), encoding="utf-8")
    code, obj, _ = run_json(capsys, "diff", str(usual), "my_review", "--home", str(home))
    assert code == 0 and obj["schema"] == "loopmath.workflows.diff/1" and not obj["same"]
    assert obj["a"]["config"] == ir_config().id and obj["b"]["workflow"] == "my_review"
    assert "r2 (reviewer) added: claude-fable-5-1/high" in obj["lines"]
    assert "rescue: redo_usual to person" in obj["lines"]
    code, obj, _ = run_json(capsys, "diff", str(usual), str(usual), "--home", str(home))
    assert code == 0 and obj["same"] and obj["lines"] == []


def test_diff_not_found(capsys, home):
    code, _, err = run(capsys, "diff", "solo", "nope", "--home", str(home))
    assert code == 2 and "nope" in err


def test_the_config_id_of_a_user_workflow_resolves(capsys, home):
    # `show mine` prints a config id; that id names the same file
    code, obj, _ = run_json(capsys, "show", "mine", "--home", str(home))
    cid = obj["config"]
    assert code == 0 and cid.startswith("cfg_")
    code, again, _ = run_json(capsys, "show", cid, "--home", str(home))
    assert code == 0 and again["origin"] == "user" and again["path"].endswith("mine.toml") and again["config"] == cid
    code, obj, _ = run_json(capsys, "diff", "mine", cid, "--home", str(home))
    assert code == 0 and obj["same"]
    code, _, err = run(capsys, "show", "cfg_000000000000", "--home", str(home))
    assert code == 2 and "(workflows, recs, runs)" in err


# ------------------------------------------------------------------ exports

def test_package_exports():
    import loopmath.workflows as wf
    from loopmath.workflows import infer as infer_module  # lane 01's migrate imports the module this way
    from loopmath.workflows import infer_workflow, normalize_role

    assert isinstance(infer_module, types.ModuleType) and infer_module.infer_workflow is infer_workflow
    assert wf.catalog is catalog and callable(wf.config_id)
    assert normalize_role("Review") == "reviewer" and normalize_role("") == "" and normalize_role(None) is None
    with pytest.raises(AttributeError):
        wf.not_a_name  # noqa: B018


def test_normalize_role_agrees_with_lane05():
    from loopmath.belief.forest import ROLE_WORDS, canonical_role
    from loopmath.workflows import normalize_role
    from loopmath.workflows.models import ROLE_ALIASES

    assert set(ROLE_ALIASES.values()) == set(ROLE_WORDS)  # every word folds onto lane 05's vocabulary
    for word in [*ROLE_ALIASES, "Reviewer ", "Something"]:
        assert canonical_role(word) == normalize_role(word), word  # lane 05 folds through normalize_role
    assert normalize_role("Something") == "something"  # unknown words pass through


def test_show_round_limit_counts_the_first_round(capsys, home):
    # budget_rounds counts the first round: 1 is no repair, 2 is one repair
    _, solo, _ = run(capsys, "show", "solo", "--home", str(home))
    assert solo.splitlines()[-1] == "round limit: 1 (no repair); rescue: redo_usual"
    path = home / "two.toml"
    text = dump_workflow(catalog()["implement_review"]).replace('id = "implement_review"', 'id = "two_rounds"')
    path.write_text(text.replace("budget_rounds = 3", "budget_rounds = 2"), encoding="utf-8")
    _, out, _ = run(capsys, "show", str(path), "--home", str(home))
    assert out.splitlines()[-1] == "round limit: 2 (up to 1 repair); rescue: redo_usual"
    assert "repair rounds" not in solo + out
