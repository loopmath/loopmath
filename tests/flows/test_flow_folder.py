"""The flow folder: `load`, `FlowFolder.configuration`, a stale plan and `save_plan`."""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from loopmath.flows import FlowFolder, load
from loopmath.flows.folder import save_plan
from loopmath.flows.records import Flow, Plan
from loopmath.ocp import conformance
from loopmath.ocp.canonical import UnresolvedWorkflow, config_id

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief" / "flow.ocp.json"
PLAN_SAMPLE = ROOT / "tests" / "fixtures" / "flows" / "plan.sample.json"
GOLDEN = ROOT / "spec" / "examples" / "golden" / "v0.4"
DIGEST = "flw_774ab7a14c50"


def _folder(tmp_path: Path, plan: dict | None = None, with_plan: bool = True) -> Path:
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    shutil.copyfile(EXAMPLE, folder / "flow.ocp.json")
    if with_plan:
        data = plan if plan is not None else json.loads(PLAN_SAMPLE.read_text(encoding="utf-8"))
        (folder / "plan.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return folder


def _plan(change) -> dict:
    data = json.loads(PLAN_SAMPLE.read_text(encoding="utf-8"))
    change(data)
    return data


def test_flow_folder_fields_are_pinned():
    assert [f.name for f in dataclasses.fields(FlowFolder)] == ["path", "dir", "doc", "flow", "digest", "plan"]


# 10. load


def test_load_reads_a_folder_and_its_plan(tmp_path):
    folder = _folder(tmp_path)
    ff = load(folder)
    assert ff.path == folder / "flow.ocp.json" and ff.dir == folder
    assert ff.doc == json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(ff.flow, Flow) and ff.flow.id == "lit-brief"
    assert ff.digest == DIGEST and isinstance(ff.plan, Plan) and ff.plan_stale is False


def test_load_reads_a_flow_document_of_any_name(tmp_path):
    path = tmp_path / "lit.json"
    shutil.copyfile(EXAMPLE, path)
    ff = load(str(path))
    assert ff.path == path and ff.dir == tmp_path and ff.plan is None and ff.plan_stale is False


def test_load_refuses_a_run_document():
    path = ROOT / "spec" / "examples" / "v0.4" / "two-checks.ocp.json"
    with pytest.raises(ValueError, match="not a flow document") as exc:
        load(path)
    assert str(path) in str(exc.value) and "flow.ocp.json" in str(exc.value)


def test_load_refuses_a_missing_path_and_a_folder_without_a_flow(tmp_path):
    with pytest.raises(ValueError, match="no such file or folder") as exc:
        load(tmp_path / "missing")
    assert "flow.ocp.json" in str(exc.value)
    with pytest.raises(ValueError, match="there is no flow.ocp.json here"):
        load(tmp_path)


def test_load_refuses_a_flow_that_does_not_validate_naming_each_code():
    with pytest.raises(ValueError, match="E212") as exc:
        load(GOLDEN / "E212-fail.ocp.json")
    assert "$['flow']['digest']" in str(exc.value)


def test_load_names_a_bad_plan_file(tmp_path):
    folder = _folder(tmp_path, plan=_plan(lambda d: d.update(schema="loopmath.flow-plan/2")))
    with pytest.raises(ValueError, match="plan.json") as exc:
        load(folder)
    assert "loopmath.flow-plan/2" in str(exc.value)


def test_load_names_a_plan_whose_flow_has_no_id(tmp_path):
    folder = _folder(tmp_path, plan=_plan(lambda d: d["flow"].pop("id")))
    with pytest.raises(ValueError, match="plan.json") as exc:
        load(folder)
    assert "the member 'id' is missing" in str(exc.value)


# 11. configuration()


def test_base_configuration_is_a_copy_of_the_flow_configuration(tmp_path):
    ff = load(_folder(tmp_path))
    for arm in (None, "base"):
        name, c = ff.configuration(arm)
        assert name == "base" and c["id"] == "cfg_9bc4f6e876f9" == config_id(c["workflow"], c["settings"])
        c["settings"]["extract"]["effort"] = "low"
        assert ff.flow.configuration["settings"]["extract"]["effort"] == "high"


def test_plan_arm_configuration(tmp_path):
    ff = load(_folder(tmp_path))
    name, c = ff.configuration("extract-sonnet")
    assert name == "extract-sonnet" and c["id"] == config_id(c["workflow"], c["settings"]) == "cfg_b8de56a265ea"
    assert c["settings"]["extract"]["model"] == {"id": "claude-sonnet-5-5"}
    c["settings"]["extract"]["effort"] = "low"
    assert ff.configuration("extract-sonnet")[1]["settings"]["extract"]["effort"] == "high"


def test_unknown_arm_is_refused_naming_the_arms(tmp_path):
    ff = load(_folder(tmp_path))
    with pytest.raises(ValueError, match="'nope'") as exc:
        ff.configuration("nope")
    assert "base, extract-sonnet" in str(exc.value)


def test_named_arm_without_a_plan_is_refused(tmp_path):
    ff = load(_folder(tmp_path, with_plan=False))
    assert ff.plan is None and ff.configuration()[0] == "base"
    with pytest.raises(ValueError, match="'extract-sonnet'") as exc:
        ff.configuration("extract-sonnet")
    assert "plan.json" in str(exc.value) and "the only arm is base" in str(exc.value)


def test_arm_whose_id_does_not_recompute_is_refused(tmp_path):
    ff = load(_folder(tmp_path, plan=_plan(lambda d: d["arms"][1]["configuration"].update(id="cfg_000000000000"))))
    with pytest.raises(ValueError, match="cfg_000000000000") as exc:
        ff.configuration("extract-sonnet")
    assert "cfg_b8de56a265ea" in str(exc.value)


def _solo_reference() -> dict:
    workflow = conformance.catalog_resolver("solo", 1)
    c = {"workflow": {"ref": "solo", "version": 1}, "source": "usual",
         "settings": {p["id"]: {"harness": "api", "model": {"id": "claude-sonnet-5-5"}}
                      for p in workflow["pieces"] if p.get("role")}}
    c["id"] = config_id(c["workflow"], c["settings"], resolve=conformance.catalog_resolver)
    return c


def test_arm_with_a_catalog_reference_resolves_its_id(tmp_path):
    c = _solo_reference()
    with pytest.raises(UnresolvedWorkflow):
        config_id(c["workflow"], c["settings"])
    ff = load(_folder(tmp_path, plan=_plan(lambda d: d["arms"][1].update(configuration=copy.deepcopy(c)))))
    assert ff.configuration("extract-sonnet") == ("extract-sonnet", c)


def test_arm_with_an_unresolved_reference_is_refused(tmp_path):
    c = _solo_reference()
    c["workflow"]["ref"] = "no-such-workflow"
    ff = load(_folder(tmp_path, plan=_plan(lambda d: d["arms"][1].update(configuration=c))))
    with pytest.raises(ValueError, match="'extract-sonnet'") as exc:
        ff.configuration("extract-sonnet")
    assert "'no-such-workflow' is not resolved" in str(exc.value)


def test_stale_plan_loads_and_refuses_its_arms(tmp_path):
    ff = load(_folder(tmp_path, plan=_plan(lambda d: d["flow"].update(digest="flw_000000000000"))))
    assert ff.plan_stale is True
    assert ff.configuration("base")[1]["id"] == "cfg_9bc4f6e876f9"
    with pytest.raises(ValueError, match="stale") as exc:
        ff.configuration("extract-sonnet")
    assert "flw_000000000000" in str(exc.value) and DIGEST in str(exc.value)
    assert "loopmath flow plan" in str(exc.value)


# 12. save_plan


def test_save_plan_writes_the_pinned_format_atomically(tmp_path):
    folder = _folder(tmp_path, with_plan=False)
    plan = Plan.from_dict(json.loads(PLAN_SAMPLE.read_text(encoding="utf-8")))
    path = save_plan(folder, plan)
    assert path == folder / "plan.json"
    text = path.read_text(encoding="utf-8")
    assert text == json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) + "\n"
    assert text.encode("utf-8") == PLAN_SAMPLE.read_bytes()
    assert sorted(p.name for p in folder.iterdir()) == ["flow.ocp.json", "plan.json"]
    assert load(folder).plan == plan


def test_save_plan_keeps_text_as_written(tmp_path):
    folder = _folder(tmp_path, with_plan=False)
    data = json.loads(PLAN_SAMPLE.read_text(encoding="utf-8"))
    data["excluded"][0]["reason"] = "kept as written: été"
    save_plan(folder, Plan.from_dict(copy.deepcopy(data)))
    assert "été" in (folder / "plan.json").read_text(encoding="utf-8")
    assert load(folder).plan.to_dict() == data
