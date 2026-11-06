"""`loopmath flow show`, `export` and `import`: show prints every finding before the flow, export and import move
the example between its two files byte for byte, and each says which file a folder reads."""

from __future__ import annotations

import argparse
import copy
import errno
import json
import os
import re
import shutil
import unicodedata
from pathlib import Path

import pytest

from loopmath import cli, cli_registry
from loopmath.flows import commands, load, simulate, walk
from loopmath.flows.folder import save_plan
from loopmath.flows.next import next_runs
from loopmath.flows.plan import make_plan
from loopmath.ocp import conformance
from loopmath.ocp.canonical import flow_digest
from loopmath.store import Store, fitjob

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "examples" / "flows" / "lit-brief"
EXAMPLE = FOLDER / "flow.ocp.json"
TOML = FOLDER / "flow.toml"
HEADINGS = ("findings:", "file read:", "flow:", "digest:", "steps:", "checks:", "budget:", "rescue:", "judges:",
            "acceptance rule:", "objective:", "hypothesis:", "allowed edits:", "inputs:", "plan:")
NOT_READ = ("flow.toml differs from flow.ocp.json and is not read while flow.ocp.json is here; run loopmath flow "
            "import flow.toml to use it")


def _flow(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["flow", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _copy(tmp_path: Path, *names: str) -> Path:
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    for name in names:
        shutil.copyfile(FOLDER / name, folder / name)
    return folder


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


# ------------------------------------------------------------------ show

def test_show_json_on_the_example(capsys):
    code, out, err = _flow(capsys, "show", str(EXAMPLE), "--json")
    shown = json.loads(out)
    assert code == 0 and err == ""
    assert tuple(shown) == commands.SHOW_KEYS == (
        "schema", "path", "findings", "id", "version", "title", "digest", "steps", "checks", "budget", "rescue",
        "judges", "acceptance_rule", "objective", "hypothesis", "allow", "inputs", "plan", "unread")
    assert shown["schema"] == "loopmath.flow-show/1" and shown["path"] == str(EXAMPLE) and shown["findings"] == []
    assert shown["id"] == "lit-brief" and shown["digest"] == "flw_774ab7a14c50"
    assert [s["id"] for s in shown["steps"]] == ["scout", "extract", "synthesize", "critic"]
    assert shown["steps"][2] == {"id": "synthesize", "role": "worker", "width": 1, "harness": "api",
                                 "model": "gpt-6-astra", "effort": "high", "context_policy": "fresh", "options": {},
                                 "reads": ["claims", "question"], "writes": ["brief"]}
    assert shown["checks"] == [
        {"id": "g_extract", "after": "extract", "rule": "command:claims-check", "on_fail": "extract"},
        {"id": "g_critic", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"}]
    assert (shown["budget"], shown["rescue"]) == (2, "redo_usual")
    assert shown["hypothesis"]["edits"] == [{"kind": "model", "piece": "extract", "value": {"id": "claude-sonnet-5-5"}},
                                            {"kind": "drop", "piece": "critic"}]
    assert shown["inputs"] == {"ref": "questions.jsonl", "type": "research", "n": 20, "holdout_n": 8}
    assert shown["plan"] is None and shown["unread"] is None


def test_show_json_holds_no_absolute_path_but_the_file_read(capsys):
    _, out, _ = _flow(capsys, "show", str(FOLDER), "--json")
    shown = json.loads(out)
    assert shown["path"] == str(EXAMPLE)
    assert [s for s in _strings({k: v for k, v in shown.items() if k != "path"}) if s.startswith("/")] == []


def test_show_reads_a_toml_view_the_same(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.toml")
    _, from_json, _ = _flow(capsys, "show", str(EXAMPLE), "--json")
    code, from_toml, _ = _flow(capsys, "show", str(folder), "--json")
    assert code == 0
    a, b = json.loads(from_json), json.loads(from_toml)
    assert b["path"] == str(folder / "flow.toml")
    assert {k: v for k, v in a.items() if k != "path"} == {k: v for k, v in b.items() if k != "path"}


def test_show_with_a_plan_says_whether_it_is_stale(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    plan = make_plan(load(folder))
    save_plan(folder, plan)
    _, out, _ = _flow(capsys, "show", str(folder), "--json")
    assert json.loads(out)["plan"] == {"id": plan.id, "digest": "flw_774ab7a14c50", "stale": False}


def test_show_with_an_error_prints_only_the_findings(tmp_path, capsys):
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc["flow"]["digest"] = "flw_000000000000"
    path = tmp_path / "flow.ocp.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    code, out, _ = _flow(capsys, "show", str(path), "--json")
    shown = json.loads(out)
    assert code == 1 and list(shown) == ["schema", "path", "findings"]
    assert "E212" in [f["code"] for f in shown["findings"] if f["level"] == "error"]
    code, out, _ = _flow(capsys, "show", str(path))
    assert code == 1 and out.splitlines()[0] == "findings:"
    assert any(line.startswith("  error E212 at ") for line in out.splitlines())
    assert "steps:" not in out.splitlines()


def test_show_text_prints_the_findings_then_each_heading(capsys):
    code, out, err = _flow(capsys, "show", str(FOLDER))
    lines = out.splitlines()
    assert code == 0 and err == "" and lines[:2] == ["findings: none", f"file read: {EXAMPLE}"]
    starts = [next(i for i, line in enumerate(lines) if line.startswith(h)) for h in HEADINGS]
    assert starts == sorted(starts)
    assert "  critic (reviewer, width 1): api claude-haiku-5-5, effort default; reads brief, claims; writes " \
           "verdict" in lines
    assert "  g_extract after extract: command:claims-check; on fail rerun extract" in lines
    assert "inputs: questions.jsonl (research), 20 inputs, 8 of them holdout" in lines
    assert NOT_READ not in lines and lines[-1] == "plan: none; loopmath flow plan writes plan.json"


def test_an_edited_toml_beside_the_document_is_named_as_not_read(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json", "flow.toml")
    toml = folder / "flow.toml"
    text = toml.read_text(encoding="utf-8")
    assert text.count("rate = 0.2") == 1
    toml.write_text(text.replace("rate = 0.2", "rate = 0.3"), encoding="utf-8")
    code, out, _ = _flow(capsys, "show", str(folder))
    assert code == 0 and out.splitlines()[1:3] == [f"file read: {folder / 'flow.ocp.json'}", NOT_READ]
    _, out, _ = _flow(capsys, "show", str(folder), "--json")
    assert json.loads(out)["unread"] == str(toml)


def test_the_example_toml_beside_the_document_is_the_same_flow(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json", "flow.toml")
    _, out, _ = _flow(capsys, "show", str(folder))
    assert NOT_READ not in out.splitlines()
    _, out, _ = _flow(capsys, "show", str(folder), "--json")
    assert json.loads(out)["unread"] is None


def test_a_toml_that_will_not_read_counts_as_differing(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    (folder / "flow.toml").write_text("not = [toml\n", encoding="utf-8")
    _, out, _ = _flow(capsys, "show", str(folder), "--json")
    assert json.loads(out)["unread"] == str(folder / "flow.toml")


@pytest.mark.parametrize("content, named", [(None, "no flow document there"),
                                             ("{", "cannot be read as a flow document"),
                                             ('{"kind": "run"}', "not a flow document (kind 'flow')")])
def test_show_refusals_are_one_error_object_under_json(content, named, tmp_path, capsys):
    given = tmp_path
    if content is not None:
        given = tmp_path / "x.json"
        given.write_text(content, encoding="utf-8")
    code, out, err = _flow(capsys, "show", str(given))
    assert code == 1 and out == "" and err.startswith("error: ") and named in err
    code, out, _ = _flow(capsys, "show", str(given), "--json")
    error = json.loads(out)
    assert code == 1 and error["schema"] == "loopmath.error/1" and named in error["error"]


NESTED = [[[[]]], [[{}]], [[]], {"a": [{"b": [1, None]}]}, [{"id": "m"}, [2]], [{}], {"id": 3}]  # lists in lists
PLACES = {  # where the member goes, and how the text shows it (the rule's line names only its own parts)
    "a judge": (lambda f, v: f["judges"][0].update(unknown_detail=v), "unknown_detail "),
    "the acceptance rule": (lambda f, v: f["acceptance_rule"].update(unknown_detail=v), None),
    "the rule's score": (lambda f, v: f["acceptance_rule"]["score"].update(unknown_detail=v), "unknown_detail "),
    "the objective": (lambda f, v: f["objective"].update(unknown_detail=v), "unknown_detail "),
    "allow": (lambda f, v: f["allow"].update(unknown_detail=v), "  unknown_detail: "),
    "a piece's allowed edits": (lambda f, v: f["allow"]["pieces"]["extract"].update(unknown_detail=v),
                                "unknown_detail "),
    "a merge": (lambda f, v: f["allow"]["merges"][0].update(unknown_detail=v), "unknown_detail "),
}


def _written(tmp_path: Path, edit) -> Path:
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    edit(doc["flow"])
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    assert [f for f in conformance.validate_doc(doc) if f.level == "error"] == []
    path = tmp_path / "flow.ocp.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.mark.parametrize("place", PLACES)
def test_show_prints_any_valid_member_a_document_carries(place, tmp_path, capsys):
    edit, shown = PLACES[place]
    for value in NESTED:
        path = _written(tmp_path, lambda f: edit(f, value))
        code, out, err = _flow(capsys, "show", str(path))
        assert code == 0 and err == ""
        text = commands._members(value) if shown == "  unknown_detail: " and isinstance(value, dict) else \
            commands._plain(value)  # allow's own members show an object's members
        assert shown is None or shown + text in out
        code, out, _ = _flow(capsys, "show", str(path), "--json")
        assert code == 0 and json.loads(out)["findings"] == []
        assert json.dumps(value) in json.dumps(json.loads(out)["acceptance_rule"] if shown is None else value)


def test_a_model_reference_list_still_prints_by_id(capsys):
    assert commands._plain([{"id": "claude-sonnet-5-5"}, {"raw": "x"}, 1, None]) == "claude-sonnet-5-5, x, 1, null"
    assert commands._plain([{"id": "m"}, [2]]) == '[{"id": "m"}, [2]]' and commands._plain({"id": 3}) == '{"id": 3}'


UNRESOLVED = ("error: workflow unresolved-workflow version 1 is not in the catalog, so flow show cannot list its "
              "steps\n")


def _unresolved(tmp_path: Path) -> Path:
    path = _written(tmp_path, lambda f: f["configuration"].update(workflow={"ref": "unresolved-workflow",
                                                                             "version": 1}))
    load(path)
    return path


def test_show_refuses_a_workflow_not_in_the_catalog_by_name(tmp_path, capsys):
    path = _unresolved(tmp_path)
    code, out, err = _flow(capsys, "show", str(path))
    assert code == 1 and out == "findings: none\n" and err == UNRESOLVED


def test_show_json_refuses_a_workflow_not_in_the_catalog_as_a_refused_document(tmp_path, capsys):
    path = _unresolved(tmp_path)
    code, out, err = _flow(capsys, "show", str(path), "--json")
    assert code == 1 and err == UNRESOLVED
    assert json.loads(out) == {"schema": "loopmath.flow-show/1", "path": str(path), "findings": []}


def test_only_a_missing_workflow_is_refused(tmp_path, capsys, monkeypatch):
    def broken(obj, resolver=None):
        raise KeyError("pieces")

    monkeypatch.setattr("loopmath.workflows.ocp.workflow_from_ocp", broken)
    with pytest.raises(KeyError):
        cli.main(["flow", "show", str(EXAMPLE)])


# ------------------------------------------------------------------ export

def test_export_writes_the_example_view(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    code, out, err = _flow(capsys, "export", str(folder))
    assert code == 0 and err == "" and (folder / "flow.toml").read_bytes() == TOML.read_bytes()
    assert out.splitlines() == [f"wrote {folder / 'flow.toml'}", "flow lit-brief, digest flw_774ab7a14c50",
                                "flow.ocp.json is the file read here; after editing flow.toml, run loopmath flow "
                                "import flow.toml"]


def test_export_json(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    code, out, err = _flow(capsys, "export", str(folder / "flow.ocp.json"), "--json")
    assert code == 0 and json.loads(out) == {"schema": "loopmath.flow-export/1", "path": str(folder / "flow.toml"),
                                             "digest": "flw_774ab7a14c50"}
    assert err == commands.READ_HERE + "\n"


def test_export_elsewhere_has_no_header(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    out_file = tmp_path / "view.toml"
    code, out, _ = _flow(capsys, "export", str(folder), "--out", str(out_file))
    header = commands.TOML_HEADER
    assert code == 0 and TOML.read_text(encoding="utf-8").startswith(header)
    assert out_file.read_text(encoding="utf-8") == TOML.read_text(encoding="utf-8")[len(header):]
    assert commands.READ_HERE not in out.splitlines() and not (folder / "flow.toml").exists()


@pytest.mark.parametrize("source, out", [
    ("flow.toml", None), ("flow.toml", "flow.toml"), ("flow.toml", "./sub/../flow.toml"),
    ("flow.ocp.json", "flow.ocp.json"), ("flow.ocp.json", "link.json"),
])
def test_export_onto_the_document_read_is_refused(source, out, tmp_path, capsys):
    folder = _copy(tmp_path, source)
    (folder / "sub").mkdir()
    (folder / "link.json").symlink_to(folder / source)
    before = (folder / source).read_bytes()
    argv = ["export", str(folder / source)] + ([] if out is None else ["--out", str(folder / out)])
    code, stdout, err = _flow(capsys, *argv)
    assert code == 1 and stdout == "" and "is the flow document read, so export would write over it" in err
    code, stdout, _ = _flow(capsys, *argv, "--json")
    assert code == 1 and json.loads(stdout)["schema"] == "loopmath.error/1"
    assert (folder / source).read_bytes() == before


STANDARD = "is not a finite number; flow.ocp.json holds standard JSON only"


@pytest.mark.parametrize("source, out, name", [
    ("flow.ocp.json", "plan.json", "plan.json"), ("flow.ocp.json", "PLAN.JSON", "plan.json"),
    ("flow.ocp.json", "plan-link.json", "plan.json"), ("flow.ocp.json", "plan-hard.json", "plan.json"),
    ("flow.toml", "plan.json", "plan.json"), ("flow.toml", "flow.ocp.json", "flow.ocp.json"),
    ("flow.toml", "Flow.Ocp.Json", "flow.ocp.json"), ("flow.toml", "doc-link.json", "flow.ocp.json"),
], ids=["plan", "plan folded", "plan symlink", "plan hard link", "plan beside a view", "document not read",
        "document not read folded", "document symlink, dangling"])
def test_export_onto_the_folders_own_files_is_refused(source, out, name, tmp_path, capsys):
    folder = _copy(tmp_path, source)
    save_plan(folder, make_plan(load(folder)))
    (folder / "plan-link.json").symlink_to(folder / "plan.json")
    os.link(folder / "plan.json", folder / "plan-hard.json")
    (folder / "doc-link.json").symlink_to(folder / "flow.ocp.json")
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    argv = ["export", str(folder), "--out", str(folder / out)]
    code, stdout, err = _flow(capsys, *argv)
    assert code == 1 and stdout == ""
    assert err == f"error: {folder / out} is the flow folder's {name}, so export would write over it; give --out FILE\n"
    code, stdout, _ = _flow(capsys, *argv, "--json")
    assert code == 1 and json.loads(stdout)["schema"] == "loopmath.error/1"
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()} == before
    assert load(folder).digest == "flw_774ab7a14c50" and load(folder).plan_stale is False
    code, _, _ = _flow(capsys, "show", str(folder))
    assert code == 0


def test_export_onto_a_plan_json_in_another_folder_writes(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    (folder / "notes").mkdir()
    code, _, _ = _flow(capsys, "export", str(folder), "--out", str(folder / "notes" / "plan.json"))
    assert code == 0 and (folder / "notes" / "plan.json").read_text(encoding="utf-8") == (
        TOML.read_text(encoding="utf-8")[len(commands.TOML_HEADER):])


@pytest.mark.parametrize("out", ["outdir", "outlink"])
def test_export_onto_a_folder_is_refused(out, tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    target = tmp_path / "outdir"
    target.mkdir()
    (target / "kept.txt").write_text("kept\n", encoding="utf-8")
    (tmp_path / "outlink").symlink_to(target)
    argv = ["export", str(folder), "--out", str(tmp_path / out)]
    code, stdout, err = _flow(capsys, *argv)
    assert code == 1 and stdout == "" and err == f"error: {tmp_path / out} is a folder; give --out FILE\n"
    code, stdout, _ = _flow(capsys, *argv, "--json")
    assert code == 1 and json.loads(stdout)["schema"] == "loopmath.error/1"
    assert sorted(p.name for p in target.iterdir()) == ["kept.txt"] and (tmp_path / "outlink").is_symlink()
    assert sorted(p.name for p in folder.iterdir()) == ["flow.ocp.json"]


NFC, NFD = unicodedata.normalize("NFC", "caf\u00e9"), unicodedata.normalize("NFD", "caf\u00e9")


def _cafe(tmp_path: Path, *names: str, plan: bool = False) -> Path:
    """A flow folder named in NFC, holding the example's NAMES, and a real plan when PLAN."""
    folder = tmp_path / NFC
    folder.mkdir()
    for name in names:
        shutil.copyfile(FOLDER / name, folder / name)
    if plan:
        save_plan(folder, make_plan(load(folder)))
    return folder


def _bytes(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def _resolve_before_3_13(monkeypatch) -> None:
    """Path.resolve as Python 3.11 and 3.12 have it: a stat after resolving, and RuntimeError on a link loop."""
    resolve = Path.resolve

    def resolve_and_stat(self, strict=False):
        path = resolve(self, strict=strict)
        try:
            os.stat(path)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RuntimeError(f"Symlink loop from {str(path)!r}") from None
        return path

    monkeypatch.setattr(Path, "resolve", resolve_and_stat)


RESOLVERS = ["this python", "resolve before 3.13", "loop check missed"]


def _resolver(mode: str, monkeypatch) -> None:
    """MODE's way to find a link loop: this Python's; `Path.resolve` as before 3.13; or that alone, the system's loop
    check missing the loop (as for one made after it)."""
    if mode != "this python":
        _resolve_before_3_13(monkeypatch)
    if mode == "loop check missed":
        monkeypatch.setattr(commands, "_link_loop", lambda path: False)


@pytest.mark.parametrize("spelling", [NFD, NFC], ids=["NFD", "NFC"])
@pytest.mark.parametrize("names, plan, out, why", [
    (["flow.toml"], False, "flow.ocp.json", "is the flow folder's flow.ocp.json"),
    (["flow.toml"], False, "plan.json", "is the flow folder's plan.json"),
    (["flow.ocp.json"], False, "plan.json", "is the flow folder's plan.json"),
    (["flow.ocp.json"], True, "plan.json", "is the flow folder's plan.json"),
    (["flow.ocp.json"], True, "flow.ocp.json", "is the flow document read"),
], ids=["document before", "plan beside a view, before", "plan before", "plan after", "document after"])
def test_export_onto_the_folders_own_files_in_either_spelling_is_refused(spelling, names, plan, out, why, tmp_path,
                                                                        capsys):
    folder = _cafe(tmp_path, *names, plan=plan)
    other = tmp_path / spelling
    other.mkdir(exist_ok=True)  # where the forms name two folders, the other one
    before = _bytes(folder)
    argv = ["export", str(folder), "--out", str(other / out)]
    code, stdout, err = _flow(capsys, *argv)
    if not os.path.samefile(other, folder):
        assert code == 0 and (other / out).is_file() and _bytes(folder) == before
        return
    assert code == 1 and stdout == ""
    assert err == f"error: {other / out} {why}, so export would write over it; give --out FILE\n"
    code, stdout, _ = _flow(capsys, *argv, "--json")
    assert code == 1 and json.loads(stdout)["schema"] == "loopmath.error/1"
    assert _bytes(folder) == before


@pytest.mark.parametrize("spelling", [NFD, NFC], ids=["NFD", "NFC"])
def test_export_beside_the_document_in_either_spelling_writes_the_header(spelling, tmp_path, capsys):
    folder = _cafe(tmp_path, "flow.ocp.json")
    other = tmp_path / spelling
    other.mkdir(exist_ok=True)
    code, _, _ = _flow(capsys, "export", str(folder), "--out", str(other / "flow.toml"))
    view = TOML.read_text(encoding="utf-8")
    beside = os.path.samefile(other, folder)
    assert code == 0 and (other / "flow.toml").read_text(encoding="utf-8") == (
        view if beside else view[len(commands.TOML_HEADER):])


DATA = Path("/System/Volumes/Data")  # on macOS, the folders under / by a second path


@pytest.mark.parametrize("names, out, why", [
    (["flow.toml"], "flow.ocp.json", "is the flow folder's flow.ocp.json"),
    (["flow.ocp.json"], "plan.json", "is the flow folder's plan.json"),
], ids=["document before", "plan before"])
def test_export_onto_the_folders_own_files_by_a_second_path_to_the_folder_is_refused(names, out, why, tmp_path,
                                                                                     capsys):
    folder = _copy(tmp_path, *names)
    other = Path(f"{DATA}{folder}")
    if not (other.is_dir() and os.path.samefile(other, folder)):
        pytest.skip("no second path to the same folder here")
    before = _bytes(folder)
    argv = ["export", str(folder), "--out", str(other / out)]
    code, stdout, err = _flow(capsys, *argv)
    assert code == 1 and stdout == ""
    assert err == f"error: {other / out} {why}, so export would write over it; give --out FILE\n"
    code, stdout, _ = _flow(capsys, *argv, "--json")
    assert code == 1 and json.loads(stdout)["schema"] == "loopmath.error/1"
    assert _bytes(folder) == before


def test_export_onto_a_missing_own_file_spelled_with_a_long_s_is_refused(tmp_path, capsys):
    """A missing own file is compared by its canonical caseless name, as APFS compares names: casefold takes the long
    s (U+017F) to s, so plan.json spelled with it is plan.json, where lower() keeps the two apart."""
    folder = _copy(tmp_path, "flow.toml")
    out = folder / ("plan.j" + chr(0x17F) + "on")
    before = _bytes(folder)
    code, stdout, err = _flow(capsys, "export", str(folder), "--out", str(out))
    assert code == 1 and stdout == ""
    assert err == f"error: {out} is the flow folder's plan.json, so export would write over it; give --out FILE\n"
    assert _bytes(folder) == before


@pytest.mark.parametrize("resolver", RESOLVERS)
@pytest.mark.parametrize("out", ["loop.toml", "loop/view.toml"], ids=["a link that loops", "below one"])
def test_export_through_a_link_that_loops_is_refused(out, resolver, tmp_path, capsys, monkeypatch):
    folder = _copy(tmp_path, "flow.ocp.json")
    (tmp_path / "loop.toml").symlink_to(tmp_path / "loop.toml")
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    links = {p.name: os.readlink(p) for p in tmp_path.iterdir() if p.is_symlink()}
    _resolver(resolver, monkeypatch)
    why = f"{tmp_path / out} goes through a link that loops, or a chain of links too long to follow; give --out FILE"
    argv = ["export", str(folder), "--out", str(tmp_path / out)]
    code, stdout, err = _flow(capsys, *argv)
    assert code == 1 and stdout == "" and err == f"error: {why}\n"
    code, stdout, _ = _flow(capsys, *argv, "--json")
    assert code == 1 and json.loads(stdout)["error"] == why
    assert {p.name: os.readlink(p) for p in tmp_path.iterdir() if p.is_symlink()} == links
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([folder.name, *links])
    assert sorted(p.name for p in folder.iterdir()) == ["flow.ocp.json"]


@pytest.mark.parametrize("resolver", RESOLVERS)
def test_a_plan_json_that_loops_does_not_stop_an_export_elsewhere(resolver, tmp_path, capsys, monkeypatch):
    folder = _copy(tmp_path, "flow.ocp.json")
    (folder / "plan.json").symlink_to(folder / "plan.json")
    _resolver(resolver, monkeypatch)
    code, _, err = _flow(capsys, "export", str(folder), "--out", str(tmp_path / "view.toml"))
    assert code == 0 and err == "" and (tmp_path / "view.toml").is_file()
    assert os.readlink(folder / "plan.json") == str(folder / "plan.json")


@pytest.mark.parametrize("out", ["gone.toml", "new/sub/view.toml"], ids=["a dangling link", "in folders not made"])
def test_export_where_nothing_is_yet_is_not_a_link_loop(out, tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    (tmp_path / "gone.toml").symlink_to(tmp_path / "nowhere" / "view.toml")
    code, _, err = _flow(capsys, "export", str(folder), "--out", str(tmp_path / out))
    assert code == 0 and err == "" and (tmp_path / out).read_text(encoding="utf-8") == (
        TOML.read_text(encoding="utf-8")[len(commands.TOML_HEADER):])


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_export_of_a_document_holding_a_non_finite_number_writes_nothing(value, tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    doc_path = folder / "flow.ocp.json"
    text = doc_path.read_text(encoding="utf-8")
    held = '"ext": {\n      "dev.loopmath.hypothesis_edits"'  # the flow's ext, written as text
    assert text.count(held) == 1
    added = held.replace("{\n", f'{{\n      "x": [1, {value}],\n')
    doc_path.write_text(text.replace(held, added), encoding="utf-8")
    view = tmp_path / "view.toml"
    view.write_bytes(b"# mine\n")
    for argv in (["export", str(folder)], ["export", str(folder), "--out", str(view)]):
        code, stdout, err = _flow(capsys, *argv)
        assert code == 1 and stdout == "" and err.startswith("error: ")
        assert err == f"error: {doc_path}: flow.ext.x.1: {value} is not a JSON number\n"
        code, stdout, _ = _flow(capsys, *argv, "--json")
        assert code == 1 and json.loads(stdout)["schema"] == "loopmath.error/1"
    assert sorted(p.name for p in folder.iterdir()) == ["flow.ocp.json"] and view.read_bytes() == b"# mine\n"


def test_export_of_a_flow_that_does_not_validate_gives_the_checkers_words(tmp_path, capsys):
    """As show prints the finding and import refuses it: the file, the code, the path and the checker's words."""
    folder = _copy(tmp_path, "flow.ocp.json")
    doc = json.loads((folder / "flow.ocp.json").read_text(encoding="utf-8"))
    del doc["flow"]["objective"]["margin"]
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    (folder / "flow.ocp.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    line = "E215 at $['flow']['objective']: objective 'cheapest_within_margin' has no margin"
    argv = ["export", str(folder), "--out", str(tmp_path / "view.toml")]
    code, out, err = _flow(capsys, *argv)
    assert code == 1 and out == "" and err == f"error: {folder / 'flow.ocp.json'}: {line}\n"
    code, out, _ = _flow(capsys, *argv, "--json")
    assert code == 1 and json.loads(out)["error"] == f"{folder / 'flow.ocp.json'}: {line}"
    assert f"  error {line}" in _flow(capsys, "show", str(folder))[1].splitlines()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["lit-brief"]


def test_export_through_a_chain_of_links_too_long_to_follow_is_refused(tmp_path, capsys):
    """The system tells a chain longer than it follows from a loop no more than this message does."""
    folder = _copy(tmp_path, "flow.ocp.json")
    chain = tmp_path / "view.toml"
    for n in range(50):
        link = tmp_path / f"link{n}"
        link.symlink_to(chain)
        chain = link
    why = f"{chain} goes through a link that loops, or a chain of links too long to follow; give --out FILE"
    code, out, err = _flow(capsys, "export", str(folder), "--out", str(chain))
    assert code == 1 and out == "" and err == f"error: {why}\n"
    assert not (tmp_path / "view.toml").exists() and sorted(p.name for p in folder.iterdir()) == ["flow.ocp.json"]


def test_export_into_a_folder_it_cannot_write_is_refused_by_name(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("this user can write in a read-only folder")
        out_file = locked / "view.toml"
        why = f"{out_file} cannot be written (Permission denied); give --out FILE in a folder you can write"
        code, out, err = _flow(capsys, "export", str(folder), "--out", str(out_file))
        assert code == 1 and out == "" and err == f"error: {why}\nnothing was written\n"
        code, out, _ = _flow(capsys, "export", str(folder), "--out", str(out_file), "--json")
        assert code == 1 and json.loads(out)["error"] == why
        assert list(locked.iterdir()) == []
    finally:
        locked.chmod(0o755)


@pytest.mark.parametrize("below", ["view.toml", "sub/view.toml"])
def test_export_below_a_file_is_refused_by_name(below, tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    held = tmp_path / "notes.txt"
    held.write_text("kept\n", encoding="utf-8")
    out_file = held / below
    code, out, err = _flow(capsys, "export", str(folder), "--out", str(out_file))
    assert code == 1 and out == ""
    assert err == f"error: {out_file} is below {held}, which is not a folder; give --out FILE in a folder\n"
    code, out, _ = _flow(capsys, "export", str(folder), "--out", str(out_file), "--json")
    assert code == 1 and json.loads(out)["schema"] == "loopmath.error/1"
    assert held.read_text(encoding="utf-8") == "kept\n" and sorted(p.name for p in tmp_path.iterdir()) == [
        "lit-brief", "notes.txt"]


def test_export_of_a_document_holding_a_number_past_the_digit_limit_names_the_file(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json")
    doc_path = folder / "flow.ocp.json"
    text = doc_path.read_text(encoding="utf-8")
    held = '"ext": {\n      "dev.loopmath.hypothesis_edits"'  # the flow's ext, written as text
    assert text.count(held) == 1
    doc_path.write_text(text.replace(held, held.replace("{\n", '{\n      "x": 1' + "0" * 4999 + ",\n")),
                        encoding="utf-8")
    code, out, err = _flow(capsys, "export", str(folder))
    assert code == 1 and out == "" and err == (f"error: {doc_path}: a number with more than 4300 digits cannot be "
                                               "read; give flow.ocp.json with shorter numbers\n")
    assert sorted(p.name for p in folder.iterdir()) == ["flow.ocp.json"]


# ------------------------------------------------------------------ import

def test_import_writes_the_example_document(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.toml")
    code, out, err = _flow(capsys, "import", str(folder / "flow.toml"))
    assert code == 0 and err == "" and (folder / "flow.ocp.json").read_bytes() == EXAMPLE.read_bytes()
    assert out.splitlines() == [f"wrote {folder / 'flow.ocp.json'}", "flow lit-brief, digest flw_774ab7a14c50"]


def test_import_into_another_folder_json(tmp_path, capsys):
    target = tmp_path / "elsewhere"
    target.mkdir()
    code, out, _ = _flow(capsys, "import", str(TOML), str(target), "--json")
    assert code == 0 and json.loads(out) == {"schema": "loopmath.flow-import/1",
                                             "path": str(target / "flow.ocp.json"), "digest": "flw_774ab7a14c50",
                                             "replaced": False, "plan_stale": False}
    assert (target / "flow.ocp.json").read_bytes() == EXAMPLE.read_bytes()


def test_import_of_a_view_that_does_not_validate_writes_nothing(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.toml")
    toml = folder / "flow.toml"
    text = toml.read_text(encoding="utf-8")
    assert text.count('id = "person"') == 1
    toml.write_text(text.replace('id = "person"', 'id = "j-astra"'), encoding="utf-8")
    code, out, err = _flow(capsys, "import", str(toml))
    assert code == 1 and out == "" and f"error: {toml}: E214 at " in err
    assert err.endswith("nothing was written\n") and sorted(p.name for p in folder.iterdir()) == ["flow.toml"]


@pytest.mark.parametrize("content, named", [(None, "cannot be read"), ("not = [toml\n", "not a flow TOML view")])
def test_import_of_an_unreadable_file_is_one_error_object_under_json(content, named, tmp_path, capsys):
    source = tmp_path / "flow.toml"
    if content is not None:
        source.write_text(content, encoding="utf-8")
    code, out, err = _flow(capsys, "import", str(source))
    assert code == 1 and out == "" and err.startswith(f"error: {source}: {named}")
    code, out, _ = _flow(capsys, "import", str(source), "--json")
    error = json.loads(out)
    assert code == 1 and error["schema"] == "loopmath.error/1" and error["command"] == "flow import"
    assert sorted(p.name for p in tmp_path.iterdir()) == ([] if content is None else ["flow.toml"])


def test_an_import_refusal_naming_a_line_separator_is_one_whole_line(tmp_path, capsys):
    """The file's path and a kept path both hold U+2028: the refusal is one line, and the whole error under --json."""
    source = tmp_path / "flow\u2028view.toml"
    source.write_text(TOML.read_text(encoding="utf-8") + '\n[[document.keep]]\npath = ["flow", "ext", "x\\u2028y"]\n'
                      'set = "Infinity"\n', encoding="utf-8")
    problem = (f"{tmp_path}/flow\\u2028view.toml: not a flow TOML view (document.keep[0].set: 'Infinity' at "
               '["flow", "ext", "x\\u2028y"] is not a finite number; flow.ocp.json holds standard JSON only)')
    code, out, err = _flow(capsys, "import", str(source))
    assert code == 1 and out == "" and err.splitlines() == [f"error: {problem}", "nothing was written"]
    code, out, _ = _flow(capsys, "import", str(source), "--json")
    error = json.loads(out)
    assert code == 1 and error["exit"] == 1 and error["command"] == "flow import" and error["error"] == problem
    assert [p.name for p in tmp_path.iterdir()] == [source.name]


def test_import_replacing_a_flow_with_a_plan_says_the_plan_is_stale(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.ocp.json", "flow.toml")
    plan = make_plan(load(folder))
    save_plan(folder, plan)
    toml = folder / "flow.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace("rate = 0.2", "rate = 0.3"), encoding="utf-8")
    code, out, err = _flow(capsys, "import", str(toml))
    lines = out.splitlines()
    assert code == 0 and err == "" and lines[0] == f"wrote {folder / 'flow.ocp.json'}, replacing the flow there"
    assert lines[2] == (f"{folder / 'plan.json'} was made for flow digest flw_774ab7a14c50, so it is now stale; "
                        "loopmath flow plan rewrites it")
    assert load(folder).plan_stale is True
    code, out, err = _flow(capsys, "import", str(toml), "--json")
    result = json.loads(out)
    assert code == 0 and result["replaced"] is True and result["plan_stale"] is True
    assert err == f"{folder / 'plan.json'} is now stale; loopmath flow plan rewrites it\n"


@pytest.mark.parametrize("path, text", [
    ('["flow", "objective", "margin"]', "1e400"), ('["flow", "ext", "x"]', "NaN"),
    ('["flow", "ext", "x"]', "Infinity"), ('["flow", "ext", "x"]', "-Infinity"), ('["flow", "ext", "x"]', "-1e400"),
])
def test_import_of_a_view_keeping_a_non_finite_number_writes_nothing(path, text, tmp_path, capsys):
    folder = _copy(tmp_path, "flow.toml")
    toml = folder / "flow.toml"
    view = toml.read_text(encoding="utf-8")
    assert view.count("margin = 0.1\n") == 1
    if "margin" in path:  # the shipped view without its margin, the margin kept instead
        view = view.replace("margin = 0.1\n", "")
    toml.write_text(view + f'\n[[document.keep]]\npath = {path}\nset = "{text}"\n', encoding="utf-8")
    code, out, err = _flow(capsys, "import", str(toml))
    assert code == 1 and out == ""
    assert err == (f"error: {toml}: not a flow TOML view (document.keep[0].set: {text!r} at {path} {STANDARD})\n"
                   "nothing was written\n")
    code, out, _ = _flow(capsys, "import", str(toml), "--json")
    assert code == 1 and json.loads(out)["schema"] == "loopmath.error/1"
    assert sorted(p.name for p in folder.iterdir()) == ["flow.toml"]


def test_import_refuses_what_the_one_writer_refuses(tmp_path, capsys, monkeypatch):
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc["x-note"] = [float("inf")]
    monkeypatch.setattr(commands, "flow_from_toml", lambda text: copy.deepcopy(doc))
    folder = _copy(tmp_path, "flow.toml")
    code, out, err = _flow(capsys, "import", str(folder / "flow.toml"))
    assert code == 1 and out == ""
    assert err == f"error: {folder / 'flow.toml'}: $['x-note'][0]: inf {STANDARD}\nnothing was written\n"
    assert sorted(p.name for p in folder.iterdir()) == ["flow.toml"]


@pytest.mark.parametrize("kind", ["a folder", "a link to a folder"])
def test_import_onto_a_folder_named_flow_ocp_json_is_refused(kind, tmp_path, capsys):
    folder = tmp_path / "f"
    folder.mkdir()
    held = folder / "flow.ocp.json" if kind == "a folder" else tmp_path / "real"
    held.mkdir()
    (held / "kept.txt").write_text("kept\n", encoding="utf-8")
    if kind == "a link to a folder":
        (folder / "flow.ocp.json").symlink_to(held)
    argv = ["import", str(TOML), str(folder)]
    code, out, err = _flow(capsys, *argv)
    assert code == 1 and out == ""
    assert err == (f"error: {folder / 'flow.ocp.json'} is a folder; import writes flow.ocp.json there; move it or "
                   "give another DIR\nnothing was written\n")
    code, out, _ = _flow(capsys, *argv, "--json")
    assert code == 1 and json.loads(out)["schema"] == "loopmath.error/1"
    assert sorted(p.name for p in held.iterdir()) == ["kept.txt"] and sorted(p.name for p in folder.iterdir()) == [
        "flow.ocp.json"]
    assert (folder / "flow.ocp.json").is_symlink() is (kind == "a link to a folder")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("held", ["no file", "a file"])
def test_the_one_writer_refuses_a_non_finite_number_and_writes_nothing(value, held, tmp_path):
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc["flow"]["ext"]["x"] = {"a": [1, value]}
    target = tmp_path / "flow.ocp.json"
    if held == "a file":
        target.write_bytes(b"old bytes\n")
    where = f"$['flow']['ext']['x']['a'][1]: {value!r} {STANDARD}"
    with pytest.raises(ValueError, match="^" + re.escape(where) + "$"):
        commands.write_flow_document(target, doc)
    assert sorted(p.name for p in tmp_path.iterdir()) == ([] if held == "no file" else ["flow.ocp.json"])
    assert held == "no file" or target.read_bytes() == b"old bytes\n"


def test_import_refuses_an_inputs_ref_that_flow_init_refuses(tmp_path, capsys):
    folder = _copy(tmp_path, "flow.toml")
    toml = folder / "flow.toml"
    text = toml.read_text(encoding="utf-8")
    assert text.count('ref = "questions.jsonl"') == 1
    toml.write_text(text.replace('ref = "questions.jsonl"', 'ref = "questions\\u0007.jsonl"'), encoding="utf-8")
    problem = walk._ref_problem("questions\x07.jsonl")
    assert problem == "a path with a control character; give a plain path"
    code, out, err = _flow(capsys, "import", str(toml))
    assert code == 1 and out == ""
    assert err == f"error: {toml}: inputs.ref: 'questions\\x07.jsonl' is {problem}\nnothing was written\n"
    code, out, _ = _flow(capsys, "import", str(toml), "--json")
    assert code == 1 and json.loads(out)["schema"] == "loopmath.error/1"
    assert sorted(p.name for p in folder.iterdir()) == ["flow.toml"]


# ------------------------------------------------------------------ plan notices and the kept plan

def _recorded_plan(capsys, tmp_path: Path, monkeypatch, plan_id: str) -> tuple[Path, Path]:
    """The example folder with a plan whose id is `plan_id` and four runs of it in the store."""
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})
    lb, home = tmp_path / "lit-brief", tmp_path / "home"
    shutil.copytree(FOLDER, lb)
    assert _flow(capsys, "plan", str(lb), "--home", str(home))[0] == 0
    folder = load(lb)
    truth = json.loads((FOLDER / "truth.json").read_text(encoding="utf-8"))
    store = Store(home)
    for doc in simulate.simulate(folder, arm="base", n=0, seed=5, truth=truth, specs=next_runs(folder, n=4, seed=1)):
        doc["run"]["flow"]["plan"] = plan_id
        store.import_run(doc)
    plan = json.loads((lb / "plan.json").read_text(encoding="utf-8"))
    plan["id"] = plan_id
    (lb / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return lb, home


def test_a_plan_whose_id_cannot_be_kept_is_not_offered_replace(capsys, tmp_path, monkeypatch):
    lb, home = _recorded_plan(capsys, tmp_path, monkeypatch, "pln_a/b")
    before = (lb / "plan.json").read_bytes()
    code, out, err = _flow(capsys, "plan", str(lb), "--home", str(home), "--margin", "0.05")
    first, unchanged = err.splitlines()
    assert code == 1 and first.startswith("error: 4 runs of plan pln_a/b are recorded; ")
    assert "--replace writes" not in first
    assert first.endswith(f"; its id cannot be kept as plan.pln_a/b.json, not one file name, so --replace cannot write "
                          f"the new plan; move {lb / 'plan.json'} away, then run this again")
    assert unchanged == f"{lb / 'plan.json'} is unchanged" and (lb / "plan.json").read_bytes() == before


@pytest.mark.parametrize("as_json", [False, True])
def test_notices_of_a_written_plan_are_one_line_each(as_json, capsys, tmp_path, monkeypatch):
    lb, home = _recorded_plan(capsys, tmp_path, monkeypatch, "pln_a\u2028b")
    code, out, err = _flow(capsys, "plan", str(lb), "--home", str(home), "--margin", "0.05", "--replace",
                           *(["--json"] if as_json else []))
    new = json.loads((lb / "plan.json").read_text(encoding="utf-8"))["id"]
    notices = (err if as_json else out).split("\n")[:2]
    assert code == 0 and notices == [
        f"kept plan pln_a\\u2028b as {lb}/plan.pln_a\\u2028b.json",
        f"replaced plan pln_a\\u2028b (flow digest flw_774ab7a14c50) with plan {new} (flow digest flw_774ab7a14c50)"]
    assert (lb / "plan.pln_a\u2028b.json").is_file()


# ------------------------------------------------------------------ what saving keeps

def _description(verb: str) -> str:
    parser = argparse.ArgumentParser(prog="loopmath")
    cli_registry.register(parser.add_subparsers(dest="command", required=True))
    flow = parser._subparsers._group_actions[0].choices["flow"]
    return " ".join(flow._subparsers._group_actions[0].choices[verb].description.split())


def test_export_help_says_what_saving_keeps(capsys):
    assert cli_registry.FLOW_SAVING == (
        "Saving writes the TOML view in a standard form, so comments and shorthand spellings are not kept, and "
        "gates that need check features are saved as checks; the flow itself does not change.")
    assert _description("export").endswith(cli_registry.FLOW_SAVING)
    with pytest.raises(SystemExit):
        cli.main(["flow", "export", "--help"])
    assert cli_registry.FLOW_SAVING in " ".join(capsys.readouterr().out.split())


def test_import_help_points_to_export(capsys):
    assert "flow export writes one; see its help for what saving keeps" in _description("import")
