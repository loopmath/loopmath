"""`loopmath flow init`: the shipped answers write the example flow, byte for byte, with an inputs skeleton and
the first plan from the planner. Refused answers and a folder that already has a flow write nothing. With no
terminal and no answers it exits 2 and lists what is missing."""

from __future__ import annotations

import copy
import errno
import io
import json
import math
import os
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.flows import commands, load, tomlview, walk
from loopmath.flows.plan import make_plan, margin_problem

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "examples" / "flows" / "lit-brief"
EXAMPLE = FOLDER / "flow.ocp.json"
ANSWERS_FILE = FOLDER / "answers.json"
ANSWERS = json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


class _Pipe(io.StringIO):
    def isatty(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _no_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Pipe(""))


def _init(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["flow", "init", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _answers_file(tmp_path: Path, answers) -> Path:
    path = tmp_path / "answers-given.json"
    path.write_text(json.dumps(answers), encoding="utf-8")
    return path


def _files(path: Path) -> list[str]:
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*")) if path.exists() else []


# ------------------------------------------------------------------ the shipped answers

def test_init_on_the_shipped_answers_writes_the_example(tmp_path, capsys):
    out_dir = tmp_path / "lit-brief"
    code, out, err = _init(capsys, str(out_dir), "--answers", str(ANSWERS_FILE))
    assert code == 0 and err == ""
    assert (out_dir / "flow.ocp.json").read_bytes() == EXAMPLE.read_bytes()
    assert _files(out_dir) == ["flow.ocp.json", "plan.json", "questions.jsonl"]
    lines = [json.loads(line) for line in (out_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in lines] == ANSWERS["inputs"]["ids"]
    assert [r["split"] for r in lines] == ["train"] * 12 + ["holdout"] * 8
    assert lines[0]["text"] == "Stub question 1." and all(r["text"] == "" for r in lines[1:])
    plan = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan == make_plan(load(EXAMPLE)).to_dict()
    assert out.splitlines() == [f"wrote {out_dir}/flow.ocp.json", f"wrote {out_dir}/questions.jsonl",
                                f"wrote {out_dir}/plan.json",
                                f"flow lit-brief, digest flw_774ab7a14c50; plan {plan['id']}",
                                f"next: loopmath flow plan {out_dir}, then loopmath flow next {out_dir}"]


def test_the_producer_is_loopmath_as_it_runs(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(tomlview, "__version__", "9.9.9")
    code, _, _ = _init(capsys, str(tmp_path / "f"), "--answers", str(ANSWERS_FILE))
    assert code == 0
    written = json.loads((tmp_path / "f" / "flow.ocp.json").read_text(encoding="utf-8"))
    example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert [k for k in example if written[k] != example[k]] == ["producer"]
    assert written["producer"]["version"] == "9.9.9" and written["flow"]["digest"] == example["flow"]["digest"]
    written["producer"] = example["producer"]
    assert json.dumps(written, indent=2, ensure_ascii=False) + "\n" == EXAMPLE.read_text(encoding="utf-8")


def test_reversed_answer_keys_write_the_same_bytes(tmp_path, capsys):
    reversed_answers = {k: ANSWERS[k] for k in reversed(walk.KEYS)}
    reversed_answers["purpose"] = dict(reversed(list(ANSWERS["purpose"].items())))
    code, _, _ = _init(capsys, str(tmp_path / "f"), "--answers", str(_answers_file(tmp_path, reversed_answers)))
    assert code == 0 and (tmp_path / "f" / "flow.ocp.json").read_bytes() == EXAMPLE.read_bytes()


def test_toml_also_writes_the_example_view(tmp_path, capsys):
    code, out, _ = _init(capsys, str(tmp_path / "f"), "--answers", str(ANSWERS_FILE), "--toml")
    assert code == 0 and (tmp_path / "f" / "flow.toml").read_bytes() == (FOLDER / "flow.toml").read_bytes()
    assert f"wrote {tmp_path / 'f'}/flow.toml" in out.splitlines()


def test_json_output(tmp_path, capsys):
    out_dir = tmp_path / "f"
    code, out, err = _init(capsys, str(out_dir), "--answers", str(ANSWERS_FILE), "--json")
    result = json.loads(out)
    assert code == 0 and err == ""
    assert list(result) == ["schema", "dir", "written", "kept", "flow", "digest", "plan"]
    assert result["schema"] == "loopmath.flow-init/1" and result["dir"] == str(out_dir)
    assert result["written"] == [str(out_dir / n) for n in ("flow.ocp.json", "questions.jsonl", "plan.json")]
    assert result["kept"] == [] and result["flow"] == "lit-brief" and result["digest"] == "flw_774ab7a14c50"
    assert result["plan"] == json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))["id"]


def test_an_inputs_file_already_there_is_kept(tmp_path, capsys):
    out_dir = tmp_path / "f"
    out_dir.mkdir()
    (out_dir / "questions.jsonl").write_text("mine\n", encoding="utf-8")
    code, out, err = _init(capsys, str(out_dir), "--answers", str(ANSWERS_FILE), "--json")
    assert code == 0 and (out_dir / "questions.jsonl").read_text(encoding="utf-8") == "mine\n"
    assert json.loads(out)["kept"] == [str(out_dir / "questions.jsonl")]
    assert err == f"kept {out_dir}/questions.jsonl: it is already there\n"


def test_the_default_folder_is_here(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out, _ = _init(capsys, "--answers", str(ANSWERS_FILE))
    assert code == 0 and (tmp_path / "flow.ocp.json").read_bytes() == EXAMPLE.read_bytes()
    assert out.splitlines()[0] == "wrote flow.ocp.json"


# ------------------------------------------------------------------ refused answers write nothing

def _without(key: str) -> dict:
    answers = copy.deepcopy(ANSWERS)
    del answers[key]
    return answers


def _edited(edit) -> dict:
    answers = copy.deepcopy(ANSWERS)
    edit(answers)
    return answers


REFUSED = [(f"no {key}", _without(key), f"{key}: missing") for key in walk.KEYS] + [
    ("unknown top key", _edited(lambda a: a.update(extra=1)), "extra: unknown key"),
    ("unknown step key", _edited(lambda a: a["steps"][0].update(modle="x")), "steps[0].modle: unknown key"),
    ("n differs", _edited(lambda a: a["inputs"].update(n=7)), "inputs.n: 7 differs from the 20 ids"),
    ("edit not allowed", _edited(lambda a: a["hypothesis"]["edits"].append({"kind": "budget", "value": 9})),
     "hypothesis.edits[2]: "),
    ("absolute ref", _edited(lambda a: a["inputs"].update(ref="/questions.jsonl")), "inputs.ref: "),
    ("ref with ..", _edited(lambda a: a["inputs"].update(ref="../questions.jsonl")), "inputs.ref: "),
    ("second gate without an id", _edited(lambda a: a["checks"]["gates"].append({"after": "critic"})),
     "checks.gates[2]: a second gate after critic needs an id"),
]


@pytest.mark.parametrize("name, answers, named", REFUSED, ids=[r[0] for r in REFUSED])
def test_refused_answers_write_nothing(name, answers, named, tmp_path, capsys):
    out_dir = tmp_path / "f"
    code, out, err = _init(capsys, str(out_dir), "--answers", str(_answers_file(tmp_path, answers)))
    assert code == 1 and out == "" and f"error: {named}" in err
    assert err.endswith("nothing was written\n") and not out_dir.exists()


def test_refused_answers_under_json_are_one_error_object(tmp_path, capsys):
    code, out, err = _init(capsys, str(tmp_path / "f"), "--answers", str(_answers_file(tmp_path, _without("steps"))),
                           "--json")
    error = json.loads(out)
    assert code == 1 and error["schema"] == "loopmath.error/1" and error["exit"] == 1
    assert error["command"] == "flow init" and error["error"] == "steps: missing (question 2)"


@pytest.mark.parametrize("names, toml", [(["flow.ocp.json"], False), (["plan.json"], False),
                                         (["flow.toml"], False), (["flow.toml"], True),
                                         (["flow.ocp.json", "flow.toml", "plan.json"], False)])
def test_a_folder_with_a_flow_is_refused(names, toml, tmp_path, capsys):
    out_dir = tmp_path / "f"
    out_dir.mkdir()
    for name in names:
        (out_dir / name).write_text("kept\n", encoding="utf-8")
    code, out, err = _init(capsys, str(out_dir), "--answers", str(ANSWERS_FILE), *(["--toml"] if toml else []))
    assert code == 1 and out == ""
    assert err == (f"error: {out_dir} already holds {', '.join(names)}, so it already has a flow; init makes a new "
                   "one, and loopmath flow import replaces a flow\n")
    assert _files(out_dir) == sorted(names) and all((out_dir / n).read_text() == "kept\n" for n in names)


@pytest.mark.parametrize("ref, why", [
    ("flow.ocp.json", "is the flow folder's flow.ocp.json"),
    ("plan.json", "is the flow folder's plan.json"),
    ("flow.toml", "is the flow folder's flow.toml"),
    ("FLOW.OCP.JSON", "is the flow folder's flow.ocp.json"),
    ("./plan.json", "is the flow folder's plan.json"),
    ("alias.jsonl", "is the flow folder's plan.json"),
    ("outside/questions.jsonl", "resolves outside"),
    ("sub", "is a folder in"),
    (".", "resolves outside"),
])
def test_inputs_ref_must_be_its_own_file_inside_the_folder(ref, why, tmp_path, capsys):
    out_dir = tmp_path / "f"
    out_dir.mkdir()
    (tmp_path / "elsewhere").mkdir()
    os.symlink(out_dir / "plan.json", out_dir / "alias.jsonl")  # dangling: plan.json is not there yet
    os.symlink(tmp_path / "elsewhere", out_dir / "outside")
    (out_dir / "sub").mkdir()
    before = _files(out_dir)
    answers = _edited(lambda a: a["inputs"].update(ref=ref))
    code, out, err = _init(capsys, str(out_dir), "--answers", str(_answers_file(tmp_path, answers)))
    assert code == 1 and out == "" and err.startswith(f"error: inputs.ref: {ref!r} {why}")
    assert _files(out_dir) == before and _files(tmp_path / "elsewhere") == []


CONTROL_REFS = [("a\u0000b", '"a\\u0000b"'), ("a\tb", '"a\\tb"'), ("a\u007fb", '"a\\u007fb"'),
                ("a\u0085b", '"a\\u0085b"'), ("inputs/\u001b.jsonl", '"inputs/\\u001b.jsonl"'),
                ("q \u00e9\u0085.jsonl", '"q \u00e9\\u0085.jsonl"')]  # the ref, and the ref as the refusal shows it


@pytest.mark.parametrize("ref, shown", CONTROL_REFS)
def test_inputs_ref_with_a_control_character_writes_nothing(ref, shown, tmp_path, capsys):
    out_dir = tmp_path / "f"
    answers = _edited(lambda a: a["inputs"].update(ref=ref))
    code, out, err = _init(capsys, str(out_dir), "--answers", str(_answers_file(tmp_path, answers)))
    assert code == 1 and out == ""
    assert err == (f"error: inputs.ref: {shown} is a path with a control character; give a plain path\n"
                   "nothing was written\n")
    assert not out_dir.exists()


@pytest.mark.parametrize("ref, shown, why", [
    *((ref, shown, "a path with a control character; give a plain path") for ref, shown in CONTROL_REFS),
    ("a\u2028b/../q.jsonl", '"a\\u2028b/../q.jsonl"', "a path with '..' leaves the flow folder; give a path inside it"),
    ("/a\u2029b.jsonl", '"/a\\u2029b.jsonl"', "an absolute path; give a path inside the flow folder"),
])
def test_a_refused_inputs_ref_is_one_whole_line_under_json(ref, shown, why, tmp_path, capsys):
    out_dir = tmp_path / "f"
    path = _answers_file(tmp_path, _edited(lambda a: a["inputs"].update(ref=ref)))
    code, out, err = _init(capsys, str(out_dir), "--answers", str(path), "--toml", "--json")
    error = json.loads(out)
    assert code == 1 and error["exit"] == 1 and error["error"] == f"inputs.ref: {shown} is {why}"
    assert not out_dir.exists()
    code, out, err = _init(capsys, str(out_dir), "--answers", str(path), "--toml")
    assert code == 1 and out == ""
    assert err.splitlines() == [f"error: inputs.ref: {shown} is {why}", "nothing was written"]
    assert not out_dir.exists()


def test_inputs_ref_in_a_subfolder_is_written_there(tmp_path, capsys):
    out_dir = tmp_path / "f"
    answers = _edited(lambda a: a["inputs"].update(ref="inputs/questions.jsonl"))
    code, _, _ = _init(capsys, str(out_dir), "--answers", str(_answers_file(tmp_path, answers)))
    assert code == 0 and _files(out_dir) == ["flow.ocp.json", "inputs", "inputs/questions.jsonl", "plan.json"]


def test_an_inputs_ref_that_cannot_be_resolved_is_a_problem_not_a_raise(tmp_path):
    problem = commands._inputs_problem(tmp_path, "a\x00b")
    assert problem.startswith("inputs.ref: 'a\\x00b' cannot be resolved (")
    assert problem.endswith("); give a plain path inside the flow folder")
    assert commands._inputs_problem(tmp_path, "inputs/questions.jsonl") is None


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


@pytest.mark.parametrize("resolver", ["this python", "resolve before 3.13", "loop check missed"])
@pytest.mark.parametrize("ref", ["q.jsonl", "sub/q.jsonl"], ids=["a link that loops", "below one"])
def test_an_inputs_ref_through_a_link_that_loops_writes_nothing(ref, resolver, tmp_path, capsys, monkeypatch):
    out_dir = tmp_path / "f"
    out_dir.mkdir()
    (out_dir / "q.jsonl").symlink_to(out_dir / "q.jsonl")
    (out_dir / "sub").symlink_to(out_dir / "sub")
    before, links = _files(out_dir), {p.name: os.readlink(p) for p in out_dir.iterdir()}
    if resolver != "this python":
        _resolve_before_3_13(monkeypatch)
    if resolver == "loop check missed":  # only Path.resolve finds the loop, and the problem names its RuntimeError
        monkeypatch.setattr(commands, "_link_loop", lambda path: False)
    path = _answers_file(tmp_path, _edited(lambda a: a["inputs"].update(ref=ref)))
    code, out, err = _init(capsys, str(out_dir), "--answers", str(path), "--json")
    why = json.loads(out)["error"]
    assert code == 1 and err.splitlines() == [f"error: {why}", "nothing was written"]
    if resolver == "loop check missed":
        assert why.startswith(f"inputs.ref: {ref!r} cannot be resolved (Symlink loop from ")
        assert why.endswith("); give a plain path inside the flow folder")
    else:
        assert why == (f"inputs.ref: {ref!r} goes through a link that loops, or a chain of links too long to follow; "
                       "give a plain path inside the flow folder")
    code, out, err = _init(capsys, str(out_dir), "--answers", str(path))
    assert code == 1 and out == "" and err.splitlines() == [f"error: {why}", "nothing was written"]
    assert _files(out_dir) == before and {p.name: os.readlink(p) for p in out_dir.iterdir()} == links


def test_a_link_loop_at_the_ref_is_a_problem_on_every_python(tmp_path, monkeypatch):
    (tmp_path / "q.jsonl").symlink_to(tmp_path / "q.jsonl")
    _resolve_before_3_13(monkeypatch)
    assert commands._inputs_problem(tmp_path, "q.jsonl") == (
        "inputs.ref: 'q.jsonl' goes through a link that loops, or a chain of links too long to follow; give a plain "
        "path inside the flow folder")
    monkeypatch.setattr(commands, "_link_loop", lambda path: False)  # a loop made after the check
    assert commands._inputs_problem(tmp_path, "q.jsonl").startswith(
        "inputs.ref: 'q.jsonl' cannot be resolved (Symlink loop from ")


def test_init_refuses_a_document_the_one_writer_refuses(tmp_path, capsys, monkeypatch):
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc["x-note"] = float("nan")
    monkeypatch.setattr(walk, "build", lambda answers: copy.deepcopy(doc))
    out_dir = tmp_path / "f"
    for toml in ([], ["--toml"]):
        code, out, err = _init(capsys, str(out_dir), "--answers", str(ANSWERS_FILE), *toml)
        assert code == 1 and out == ""
        assert err == ("error: the flow cannot be written ($['x-note']: nan is not a finite number; flow.ocp.json "
                       "holds standard JSON only)\nnothing was written\n")
        assert not out_dir.exists()


def test_init_writes_nothing_when_the_toml_view_cannot_be_written(tmp_path, capsys, monkeypatch):
    def refuse(doc):
        raise RuntimeError("the TOML view does not read back")

    monkeypatch.setattr(commands, "flow_to_toml", refuse)
    out_dir = tmp_path / "f"
    code, out, err = _init(capsys, str(out_dir), "--answers", str(ANSWERS_FILE), "--toml")
    assert code == 1 and out == ""
    assert err == "error: the flow cannot be written (the TOML view does not read back)\nnothing was written\n"
    assert not out_dir.exists()


def test_a_plan_the_planner_refuses_keeps_the_flow(tmp_path, capsys, monkeypatch):
    def refuse(folder, **_):
        raise ValueError("no arms")

    monkeypatch.setattr("loopmath.flows.plan.make_plan", refuse)
    out_dir = tmp_path / "f"
    code, out, err = _init(capsys, str(out_dir), "--answers", str(ANSWERS_FILE))
    assert code == 1 and out == ""
    assert err == (f"wrote {out_dir}/flow.ocp.json\nwrote {out_dir}/questions.jsonl\nerror: the first plan: no arms\n"
                   "the files above are kept; loopmath flow plan writes the plan later\n")
    assert _files(out_dir) == ["flow.ocp.json", "questions.jsonl"]
    assert (out_dir / "flow.ocp.json").read_bytes() == EXAMPLE.read_bytes()


# ------------------------------------------------------------------ no terminal, the walk, the answers file

def test_no_terminal_and_no_answers_exits_2_listing_what_is_missing(tmp_path, capsys):
    code, out, err = _init(capsys, str(tmp_path / "f"))
    assert code == 2 and out == ""
    assert err == ("error: no answers: stdin is not a terminal and no --answers was given, so these are missing: "
                   "purpose, steps, checks, judging, objective, hypothesis, allow, inputs; give --answers FILE\n")
    assert not (tmp_path / "f").exists()


def test_no_terminal_under_json_is_one_error_object(tmp_path, capsys):
    code, out, _ = _init(capsys, str(tmp_path / "f"), "--json")
    error = json.loads(out)
    assert code == 2 and error["schema"] == "loopmath.error/1" and error["exit"] == 2
    assert error["error"].endswith("missing: " + ", ".join(walk.KEYS) + "; give --answers FILE")


def test_the_walk_on_a_terminal_writes_the_same_bytes(tmp_path, capsys, monkeypatch):
    fed = "".join(json.dumps(ANSWERS[k]) + "\n\n" for k in walk.KEYS) + "\n\n"
    monkeypatch.setattr("sys.stdin", _Terminal(fed))
    code, _, err = _init(capsys, str(tmp_path / "f"))
    assert code == 0 and (tmp_path / "f" / "flow.ocp.json").read_bytes() == EXAMPLE.read_bytes()
    assert "1. What is the flow for?" in err and "8. What are the inputs" in err


def test_the_walk_ending_early_exits_2_naming_the_rest(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", _Terminal("".join(json.dumps(ANSWERS[k]) + "\n\n" for k in walk.KEYS[:5])))
    code, out, err = _init(capsys, str(tmp_path / "f"))
    assert code == 2 and out == "" and not (tmp_path / "f").exists()
    assert err.splitlines()[-1] == ("error: the input ended before every answer was given, so these are missing: "
                                    "hypothesis, allow, inputs; give --answers FILE")


def test_a_folder_with_a_flow_is_refused_before_the_walk(tmp_path, capsys, monkeypatch):
    (tmp_path / "flow.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr("sys.stdin", _Terminal(""))
    code, _, err = _init(capsys, str(tmp_path))
    assert code == 1 and "1. What is the flow for?" not in err


@pytest.mark.parametrize("content", [None, "{not json", b"\xff\xfe", '{"purpose": 1' + "0" * 5000 + "}"],
                         ids=["absent", "not JSON", "not UTF-8", "an integer of 5001 digits"])
def test_an_unreadable_answers_file_exits_2(content, tmp_path, capsys):
    path = tmp_path / "answers.json"
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    elif content is not None:
        path.write_bytes(content)
    code, out, err = _init(capsys, str(tmp_path / "f"), "--answers", str(path))
    assert code == 2 and out == "" and err.startswith(f"error: --answers {path}: cannot be read as JSON (")
    assert not (tmp_path / "f").exists()


# --from-ocp and --from-rows (prefill): tests/flows/test_flow_prefill.py


def test_the_one_writer_is_the_example_format(tmp_path):
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    path = commands.write_flow_document(tmp_path / "flow.ocp.json", {k: doc[k] for k in reversed(list(doc))})
    assert path.read_bytes() == EXAMPLE.read_bytes() and _files(tmp_path) == ["flow.ocp.json"]


@pytest.mark.parametrize("answers, shown", [(None, "null"), ([], "[]"), ("lit-brief", '"lit-brief"')])
def test_answers_that_are_not_an_object_are_refused_without_the_walk(answers, shown, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", _Terminal("".join(json.dumps(ANSWERS[k]) + "\n\n" for k in walk.KEYS)))
    path = _answers_file(tmp_path, answers)
    code, out, err = _init(capsys, str(tmp_path / "f"), "--answers", str(path))
    assert code == 1 and out == "" and not (tmp_path / "f").exists()
    assert err == (f"error: --answers {path}: the answers: expected a JSON object with the keys "
                   f"{', '.join(walk.KEYS)}, got {shown}\nnothing was written\n")
    code, out, err = _init(capsys, str(tmp_path / "f"), "--answers", str(path), "--json")
    error = json.loads(out)
    assert code == 1 and error["exit"] == 1 and "JSON object" in error["error"] and "missing" not in error["error"]
    assert "Give the answer as JSON" not in err


@pytest.mark.parametrize("ref, why", [
    ("flow.ocp.json/rows.jsonl", "is below the flow folder's flow.ocp.json, a file init writes"),
    ("flow.toml/rows.jsonl", "is below the flow folder's flow.toml, a file init writes"),
    ("plan.json/rows.jsonl", "is below the flow folder's plan.json, a file init writes"),
    ("Plan.JSON/rows.jsonl", "is below the flow folder's plan.json, a file init writes"),
    ("to-plan/rows.jsonl", "is below the flow folder's plan.json, a file init writes"),
    ("notes.txt/rows.jsonl", "is below notes.txt, which is a file"),
    ("data/notes.txt/rows.jsonl", "is below data/notes.txt, which is a file"),
])
def test_inputs_ref_below_a_file_is_refused_before_any_write(ref, why, tmp_path, capsys):
    out_dir = tmp_path / "f"
    (out_dir / "data").mkdir(parents=True)
    (out_dir / "notes.txt").write_text("notes\n", encoding="utf-8")
    (out_dir / "data" / "notes.txt").write_text("notes\n", encoding="utf-8")
    os.symlink(out_dir / "plan.json", out_dir / "to-plan")  # dangling: plan.json is not there yet
    before = _files(out_dir)
    answers = _edited(lambda a: a["inputs"].update(ref=ref))
    code, out, err = _init(capsys, str(out_dir), "--answers", str(_answers_file(tmp_path, answers)), "--toml")
    assert code == 1 and out == "" and err.startswith(f"error: inputs.ref: {ref!r} {why}")
    assert err.endswith("nothing was written\n") and _files(out_dir) == before


def test_inputs_in_a_new_folder_are_written_there(tmp_path, capsys):
    out_dir = tmp_path / "f"
    answers = _edited(lambda a: a["inputs"].update(ref="data/questions.jsonl"))
    code, out, _ = _init(capsys, str(out_dir), "--answers", str(_answers_file(tmp_path, answers)), "--toml")
    assert code == 0 and f"wrote {out_dir}/data/questions.jsonl" in out.splitlines()
    assert _files(out_dir) == ["data", "data/questions.jsonl", "flow.ocp.json", "flow.toml", "plan.json"]


# ------------------------------------------------------------------ the margin a first plan takes; one-line refusals

ASSUMED = " (at a first plan's assumed spreads)"
BELOW = 0.0004897787465809092  # the float below the least margin a first plan takes at the example's two judges


def _one_model_judge(answers: dict) -> None:
    del answers["judging"]["judges"][1]


def _refused_on_one_line(tmp_path: Path, capsys, answers: dict, problem: str) -> None:
    """`--answers` refused with this one problem: the whole of it under --json, one line in text, and nothing is
    written, neither the folder nor any file in it."""
    path, out_dir = _answers_file(tmp_path, answers), tmp_path / "f"
    code, out, err = _init(capsys, str(out_dir), "--answers", str(path), "--toml", "--json")
    error = json.loads(out)
    assert code == 1 and error["exit"] == 1 and error["error"] == problem
    code, out, err = _init(capsys, str(out_dir), "--answers", str(path), "--toml")
    assert code == 1 and out == "" and err.splitlines() == [f"error: {problem}", "nothing was written"]
    assert _files(tmp_path) == ["answers-given.json"]


@pytest.mark.parametrize("edit, why", [
    pytest.param(lambda a: a["objective"].update(margin=1e-160),
                 "margin 1e-160: needs more than 1e308 check pairs at sd_pair 0.197, above the limit of 1000000",
                 id="1e-160"),
    pytest.param(lambda a: a["objective"].update(margin=BELOW),
                 f"margin {BELOW!r}: needs 1000001 check pairs at sd_pair 0.197, above the limit of 1000000",
                 id="below the edge"),
    pytest.param(lambda a: (_one_model_judge(a), a["objective"].update(margin=5.2e-4)),
                 "margin 0.00052: needs 1115787 check pairs at sd_pair 0.2209, above the limit of 1000000",
                 id="5.2e-4 at one model judge"),
])
def test_a_margin_a_first_plan_refuses_writes_nothing(edit, why, tmp_path, capsys):
    assert margin_problem(BELOW, k=2) is not None and margin_problem(math.nextafter(BELOW, 1.0), k=2) is None
    _refused_on_one_line(tmp_path, capsys, _edited(edit),
                         f"objective.margin: objective 'cheapest_within_margin': {why}{ASSUMED} (E215)")


def test_a_margin_a_first_plan_takes_writes_the_folder(tmp_path, capsys):
    out_dir = tmp_path / "f"
    answers = _edited(lambda a: a["objective"].update(margin=0.05))
    code, _, err = _init(capsys, str(out_dir), "--answers", str(_answers_file(tmp_path, answers)))
    assert code == 0 and err == "" and _files(out_dir) == ["flow.ocp.json", "plan.json", "questions.jsonl"]
    assert json.loads((out_dir / "flow.ocp.json").read_text(encoding="utf-8"))["flow"]["objective"]["margin"] == 0.05


@pytest.mark.parametrize("n, shown", [("a\u0085b", '"a\\u0085b"'), ("a\u2028b", '"a\\u2028b"'), ("q é", '"q é"')])
def test_a_refused_answer_value_is_one_whole_line_under_json(n, shown, tmp_path, capsys):
    _refused_on_one_line(tmp_path, capsys, _edited(lambda a: a["inputs"].update(n=n)),
                         f"inputs.n: expected an integer, 0 or more, got {shown}")


def test_an_answers_path_with_a_line_separator_is_one_whole_line(tmp_path, capsys):
    path = tmp_path / "answers\u2028given.json"
    path.write_text("{not json", encoding="utf-8")
    code, out, err = _init(capsys, str(tmp_path / "f"), "--answers", str(path))
    assert code == 2 and out == "" and len(err.splitlines()) == 1
    assert err.startswith(f"error: --answers {tmp_path}/answers\\u2028given.json: cannot be read as JSON (")
    code, out, _ = _init(capsys, str(tmp_path / "f"), "--answers", str(path), "--json")
    error = json.loads(out)
    assert code == 2 and error["exit"] == 2 and error["error"] == err[len("error: "):-1]
    assert not (tmp_path / "f").exists()
