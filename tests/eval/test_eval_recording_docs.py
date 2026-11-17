"""`docs/eval-recording.md` and the agent guide `eval-record`, held to the code.

The tutorial tells people how to record runs and judge outputs for `loopmath eval`. What it shows is checked
here against the real commands and the real validators:

- the example study under `docs/examples/eval-recording/` is what `record_study.py` writes, byte for byte, and
  every example passes `eval rubric check` and `eval scores check`, with the counts its design gives;
- a block marked `<!-- file: NAME -->` holds lines of that example file, word for word;
- a block marked `<!-- output: NAME -->` is the standard output of the command of that name, run here, and a
  block marked `<!-- stderr: NAME -->` is its standard error;
- a table marked `<!-- fields: SCHEMA -->` names members of that format: each one is known to the validator
  (a name it does not know is caught), and its "Required" cell is what the validator does when it is left out;
- a table marked `<!-- columns: TABLE -->` names the columns `eval scores import` reads from a CSV;
- a block marked `<!-- refusal: NAME -->` is the message `eval scores check` prints for that mistake, made here
  by breaking a copy of the example in one place;
- what the tutorial says the check lets through, it lets through;
- what the recording rules are for is checked on the model the fit builds, not only on the schema: the item
  lines of one call are one reading, the calls in the files are the requests the generator made, and a
  correction leaves the number of panel requests as it was;
- every example run is marked simulated, and the fit takes the examples only with `--allow-simulated`;
- a limit the tutorial states: the verdicts of one pair request share a call id in the record, and the fit
  builds the same model whether they share it or not. While that holds, the limit stays in the documents;
- every `loopmath eval` command shown parses, every link to a heading lands, and the agent guide's JSON
  examples validate.

A block marked `<!-- recorded: NAME -->` is output that depends on the sampler. The test marked `slow` runs the
fit of the walk (about 2 minutes on 2 cores) and compares those blocks with the numbers masked. It runs only
with LOOPMATH_EVAL_SLOW=1.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import types
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import engine_pymc as engine
from loopmath.eval import model as M
from loopmath.eval import scores as S

REPO = Path(__file__).parents[2]
DOC = REPO / "docs" / "eval-recording.md"
GUIDE = REPO / "src" / "loopmath" / "skill" / "guides" / "eval-record.md"
EX = REPO / "docs" / "examples" / "eval-recording"
needs_engine = pytest.mark.skipif(not engine.available(), reason="needs the [bayes] extra")
EM_DASH = chr(0x2014)


def lm(*argv: object) -> tuple[int, str, str]:
    """`loopmath ARGV` in this process: the exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main([str(word) for word in argv])
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


def check(study: object, rubric: object = None, judges: object = None, *more: str) -> tuple[int, str, str]:
    """`eval scores check` on a study folder, with the rubric and the judges file of that folder by default."""
    study = Path(str(study))
    return lm("eval", "scores", "check", study, "--rubric", rubric or study / "rubric.json",
              "--judges", judges or study / "judges.json", *more)


def report(study: object, rubric: object = None, judges: object = None) -> dict:
    """The JSON report of `eval scores check`, whatever the exit code."""
    return json.loads(check(study, rubric, judges, "--json")[1])


# ---------------------------------------------------------------------------------------------------
# Reading the tutorial

TAG = re.compile(r"<!-- (file|output|stderr|recorded|refusal|fields|columns): (.+?) -->")


def tagged(text: str, kind: str) -> list[tuple[str, list[str]]]:
    """`(argument, lines)` of each fenced block or table that follows a `<!-- kind: argument -->` line."""
    lines, found = text.splitlines(), []
    for n, line in enumerate(lines):
        match = TAG.fullmatch(line.strip())
        if match is None or match.group(1) != kind:
            continue
        rest, body = lines[n + 1:], []
        if rest and rest[0].startswith("```"):
            for row in rest[1:]:
                if row.startswith("```"):
                    break
                body.append(row)
        else:
            for row in rest:
                if not row.startswith("|"):
                    break
                body.append(row)
        assert body, f"nothing follows the mark on line {n + 1}"
        found.append((match.group(2), body))
    return found


def table(rows: list[str]) -> list[list[str]]:
    """The cells of a Markdown table, without its two heading rows."""
    return [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows[2:]]


def named(cell: str) -> str:
    """The name a table cell gives in backticks."""
    match = re.fullmatch(r"`([^`]+)`", cell)
    assert match, f"the cell {cell!r} is not one name in backticks"
    return match.group(1)


def shown_in(block: list[str], real: list[str]) -> bool:
    """Whether the lines of a block are lines of the real output, in order. A `...` line stands for lines left
    out; a block without one is the whole output. Blank lines do not count."""
    want = [line.rstrip() for line in block if line.strip()]
    have = [line.rstrip() for line in real if line.strip()]
    if "..." not in want:
        return want == have
    rest = iter(have)
    return all(any(line == other for other in rest) for line in want if line != "...")


def masked(lines: list[str]) -> list[str]:
    """Lines with every number replaced, for output that depends on the sampler."""
    return [re.sub(r"[-+]?\d+(?:\.\d+)?", "N", line) for line in lines]


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# The walk of the tutorial, up to the fit

STUDY_ARGS = ("--rubric", "my-study/rubric.json", "--judges", "my-study/judges.json")
OCP_IMPORT = ("eval", "scores", "import", "--ocp", EX / "ocp" / "r201.ocp.json", EX / "ocp" / "r202.ocp.json")
REVIEWER_IMPORT = ("eval", "scores", "import", "--csv", "my-study-reviewer.csv", "--pairs-csv",
                   "my-study-reviewer-pairs.csv", "--rubric", "my-study/rubric.json", "--out", "my-study")


@pytest.fixture(scope="module")
def walk(tmp_path_factory) -> tuple[Path, dict[str, str]]:
    """The commands of the tutorial, run as written in a new folder: the folder, and each standard output under
    the name the tutorial marks it with. `generated` is a copy of the study before the reviewer's sheets."""
    root, before, out = tmp_path_factory.mktemp("recording"), Path.cwd(), {}
    os.chdir(root)
    try:
        done = subprocess.run([sys.executable, str(EX / "record_study.py"), "--out", "my-study"],
                              capture_output=True, text=True, check=True)
        out["record"] = done.stdout
        shutil.copytree("my-study", "generated")

        def step(name: str, *argv: object) -> None:
            code, text, err = lm(*argv)
            assert code == 0, (name, err.strip()[-600:])
            out[name] = text

        step("rubric-check", "eval", "rubric", "check", "my-study/rubric.json", "--judges", "my-study/judges.json",
             "--value", "my-study/value.json")
        step("import-reviewer", *REVIEWER_IMPORT)
        step("import-reviewer-again", *REVIEWER_IMPORT)
        step("check", "eval", "scores", "check", "my-study", *STUDY_ARGS)
        step("import-ocp", *OCP_IMPORT, "--rubric", "my-study/rubric.json", "--out", "ocp-study",
             "--allow-simulated")
        step("check-ocp", "eval", "scores", "check", "ocp-study", *STUDY_ARGS)
    finally:
        os.chdir(before)
    return root, out


def test_the_committed_study_is_what_the_generator_writes(walk):
    """`record_study.py` writes the files under `docs/examples/eval-recording/`, byte for byte."""
    root, _ = walk
    names = sorted(path.name for path in (EX / "study").iterdir())
    assert names == ["judges.json", "pairs.jsonl", "rubric.json", "runs.jsonl", "scores.jsonl", "value.json"]
    assert sorted(path.name for path in (root / "generated").iterdir()) == names
    for name in names:
        assert (root / "generated" / name).read_bytes() == (EX / "study" / name).read_bytes(), name
    for name in ("reviewer.csv", "reviewer-pairs.csv"):
        assert (root / f"my-study-{name}").read_bytes() == (EX / f"study-{name}").read_bytes(), name


def test_the_generated_study_has_the_readings_its_design_gives(walk):
    """Counts worked out from the design, not from the files: 16 inputs on 2 arms; `judge-main` scores 4
    components per output and 2 again on every other output; `judge-cheap` makes one panel call per output and
    a second on every fourth, and one of its lines is corrected; the reviewer scores 12 outputs, 4 of them
    twice on 2 components, and corrects one slip; `judge-main` compares 8 pairs in both orders and the reviewer
    6 pairs."""
    root, _ = walk
    got = report(root / "my-study")
    assert got["ok"] is True and got["problems"] == [] and got["warnings"] == [] and got["mixed_arm"] == []
    main, cheap, reviewer = 32 * 4 + 16 * 2, (32 + 8) * 4 + 1, 12 * 4 + 4 * 2 + 1
    assert (main, cheap, reviewer) == (160, 161, 57)
    assert got["counts"] == {"runs": 32, "readings": main + cheap + reviewer, "readings_read": main + cheap + reviewer - 2,
                             "pairs": 8 * 2 + 6, "pairs_read": 22, "items": 0}
    assert sorted((row["id"], row["by"]) for row in got["not_read"]) == [
        ("c0161.depth", "c0161.depth.fix1"), ("rv.r026.depth", "rv.r026.depth.fixed")]
    assert {(row["table"], row["reason"]) for row in got["not_read"]} == {("readings", "replaced")}
    design = {(row["term"], row["judge"], row["dimension"]): row for row in got["design"]}
    for dim, repeated in (("final/correct", 0), ("final/clarity", 0), ("final/depth", 16), ("final/overall", 16)):
        row = design[("sigma", "judge-main", dim)]
        assert (row["count"], row["met"]) == (repeated, repeated > 0), dim
        assert design[("sigma", "judge-cheap", dim)]["count"] == 8, dim
        assert (design[("anchor", None, dim)]["count"], design[("anchor", None, dim)]["state"]) == (12, "weak"), dim
    assert design[("halo", "judge-cheap", None)]["count"] == 8
    assert design[("position", "judge-main", "final/overall")]["count"] == 8


@pytest.mark.parametrize("folder, counts", [
    ("study", {"runs": 32, "readings": 321, "readings_read": 320, "pairs": 16, "pairs_read": 16, "items": 0}),
    ("scales", {"runs": 2, "readings": 25, "readings_read": 22, "pairs": 5, "pairs_read": 4, "items": 3}),
])
def test_each_committed_example_passes_the_checks(folder, counts):
    """`eval rubric check` and `eval scores check` pass on the example folders as they are in the repository.
    The counts of `scales` are counted by hand from its files."""
    study = EX / folder
    extra = ["--items", study / "items.jsonl"] if (study / "items.jsonl").is_file() else []
    extra += ["--value", study / "value.json"] if (study / "value.json").is_file() else []
    code, _, err = lm("eval", "rubric", "check", study / "rubric.json", "--judges", study / "judges.json", *extra)
    assert code == 0, err
    code, out, err = check(study)
    assert code == 0 and out.rstrip().endswith("\nok"), err
    got = report(study)
    assert got["ok"] is True and got["problems"] == [] and got["counts"] == counts


def test_the_ocp_examples_import_as_the_tutorial_says(walk):
    """Two runs come in. The signals that carry a call are two repeats; the two that carry none are read as a
    reading and its correction; the signal that names no component is left out."""
    root, _ = walk
    code, _, err = lm("ocp", "validate", EX / "ocp" / "r201.ocp.json", EX / "ocp" / "r202.ocp.json")
    assert code == 0, err
    got = report(root / "ocp-study", root / "my-study" / "rubric.json", root / "my-study" / "judges.json")
    assert got["ok"] is True
    assert got["counts"] == {"runs": 2, "readings": 6, "readings_read": 5, "pairs": 0, "pairs_read": 0, "items": 0}
    rows = {row["id"]: row for row in C.read_lines(root / "ocp-study" / "scores.jsonl", C.SCHEMA_SCORES)}
    assert (rows["c0901.depth"]["call"], rows["c0902.depth"]["call"]) == ("c0901", "c0902")
    assert "replaces" not in rows["c0901.depth"] and "replaces" not in rows["c0902.depth"]
    assert rows["c0903.clarity"]["value"] == "ok" and rows["h0901.correct"]["judge"] == "reviewer-1"
    assert rows["r202.s2"]["replaces"] == "r202.s1" and len(rows) == 6
    runs = {row["run"]: row for row in C.read_lines(root / "ocp-study" / "runs.jsonl", C.SCHEMA_RUNS)}
    assert (runs["r201"]["input"], runs["r201"]["arm"], runs["r202"]["arm"]) == ("q17", "baseline", "candidate")
    assert runs["r202"]["steps"] == [{"step": "final", "model": {"id": "model-large", "family": "alpha"},
                                      "effort": "high"}]
    assert runs["r201"]["simulated"] is True and runs["r202"]["simulated"] is True


# ---------------------------------------------------------------------------------------------------
# Blocks of the tutorial against the files and the commands

def test_every_file_block_holds_lines_of_its_example_file(doc):
    blocks = tagged(doc, "file")
    assert len(blocks) >= 20
    for name, block in blocks:
        real = (EX / name).read_text(encoding="utf-8").splitlines()
        rest = iter(real)
        for line in block:
            assert any(line == other for other in rest), f"{name}: not a line of the file, or out of order: {line[:120]}"
    assert {name for name, _ in blocks} >= {
        "record_study.py", "study/runs.jsonl", "study/scores.jsonl", "study/pairs.jsonl", "study/value.json",
        "study-reviewer.csv", "study-reviewer-pairs.csv", "scales/rubric.json", "scales/judges.json",
        "scales/runs.jsonl", "scales/scores.jsonl", "scales/pairs.jsonl", "scales/items.jsonl", "ocp/r201.ocp.json"}


def test_every_json_line_shown_is_a_valid_record(doc):
    """Each line of a `.jsonl` block validates on its own, so a reader can copy any of them."""
    validators = {C.SCHEMA_RUNS: C.validate_run, C.SCHEMA_SCORES: C.validate_score, C.SCHEMA_PAIRS: C.validate_pair,
                  C.SCHEMA_ITEMS: C.validate_item}
    seen = 0
    for name, block in tagged(doc, "file"):
        if name.endswith(".jsonl"):
            for line in block:
                row = json.loads(line)
                validators[row["schema"]](row, name)
                seen += 1
    assert seen >= 30


def test_every_output_block_is_what_the_command_prints(doc, walk):
    _, out = walk
    blocks = tagged(doc, "output")
    assert {name for name, _ in blocks} == set(out), "an output block has no command here, or a command no block"
    for name, block in blocks:
        assert shown_in(block, out[name].splitlines()), f"{name}:\n{out[name]}"


@needs_engine
def test_the_fit_report_shown_is_the_one_check_only_prints(doc, walk, monkeypatch):
    """The part of the fit output above its last two lines does not depend on the sampler: it is what
    `eval fit --check-only` prints, and that command writes nothing."""
    root, _ = walk
    monkeypatch.chdir(root)
    code, out, err = lm("eval", "fit", "my-study", *STUDY_ARGS, "--allow-simulated", "--check-only")
    assert code == 0, err
    assert not (root / "my-study" / "fit.json").exists() or os.environ.get("LOOPMATH_EVAL_SLOW") == "1"
    (block,) = [block for name, block in tagged(doc, "recorded") if name == "fit"]
    assert block[-2].startswith("fit: my-study/fit.json") and block[-1].startswith("converged: yes")
    assert shown_in(block[:-2] + ["..."], out.splitlines()), out
    assert "readings: 376 of 378 used; pairs: 22 of 22 used; 2 rows not used" in out


# ---------------------------------------------------------------------------------------------------
# Field tables against the validators

FULL = {
    C.SCHEMA_RUBRIC: (C.validate_rubric, {
        "schema": C.SCHEMA_RUBRIC, "id": "answer", "version": "1", "steps": ["draft", "final"],
        "components": [{"id": "depth", "title": "Depth", "role": "criterion",
                        "scale": {"kind": "linear", "range": [0, 10]}, "better": "higher", "applies_to": ["final"],
                        "anchors": "anchors/depth.md", "review_minutes": 2, "compare_minutes": 1,
                        "human_only": False}]}),
    C.SCHEMA_JUDGES: (C.validate_judges, {
        "schema": C.SCHEMA_JUDGES, "anchor": ["reviewer-1"], "min_anchor_targets": 6,
        "judges": [{"id": "judge-main", "kind": "model",
                    "model": {"id": "model-large", "family": "alpha", "provider": "example-cloud"},
                    "effort": "high", "prompt": "rubric-prompt-v2", "mode": "panel",
                    "reference": {"shown": True, "version": "ref-1",
                                  "author": {"kind": "verified", "model": {"id": "model-small", "family": "beta"}}},
                    "blinded": True, "cost_usd_per_call": 0.03, "components": ["depth"]},
                   {"id": "reviewer-1", "kind": "person"}]}),
    C.SCHEMA_RUNS: (C.validate_run, {
        "schema": C.SCHEMA_RUNS, "run": "r001", "input": "q01", "arm": "baseline", "split": "dev",
        "steps": [{"step": "final", "model": {"id": "model-small", "family": "beta"}, "effort": "low",
                   "variant": "prompt-a"}],
        "cost_usd": 0.01, "simulated": False, "refs": {"output": "outputs/r001.md"}}),
    C.SCHEMA_SCORES: (C.validate_score, {
        "schema": C.SCHEMA_SCORES, "id": "c0001.depth", "run": "r001", "step": "final", "component": "depth",
        "judge": "judge-main", "call": "c0001", "value": 6.5, "rubric": "answer@1",
        "observed_at": "2026-03-02T09:00:00+00:00", "replaces": "c0000.depth", "assignment": "lb-001",
        "cost_usd": 0.03}),
    C.SCHEMA_PAIRS: (C.validate_pair, {
        "schema": C.SCHEMA_PAIRS, "id": "c0002", "a": "r001", "b": "r002", "step": "final", "component": "overall",
        "judge": "judge-main", "call": "c0002", "shown_first": "a", "verdict": "tie", "rubric": "answer@1",
        "observed_at": "2026-03-02T09:00:20+00:00", "replaces": "c0001", "assignment": "lb-002", "cost_usd": 0.05}),
    C.SCHEMA_ITEMS: (C.validate_item, {
        "schema": C.SCHEMA_ITEMS, "id": "q01.deadline", "input": "q01", "component": "covers",
        "label": "States the deadline", "source": {"kind": "model_verified", "model": "model-large"}}),
    C.SCHEMA_VALUE: (C.validate_value, {
        "schema": C.SCHEMA_VALUE, "weights": {"depth": 2}, "gated_value": 0,
        "gates": [{"component": "correct", "at_least": 1, "step": "final"}],
        "objective": {"kind": "best_value", "margin": 0.1, "reference": "baseline", "cost_cap_usd": 1.0}}),
}
FREE = ("ext", "refs", "weights")       # objects whose member names are the user's own


def holder(obj: dict, path: str) -> tuple[dict, str]:
    """The object that holds the last name of a dotted path, and that name. `name[]` is the first entry of a list."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[part[:-2]][0] if part.endswith("[]") else obj[part]
        assert isinstance(obj, dict), path
    return obj, parts[-1]


def code_of(validate, obj: dict) -> str | None:
    """The code of the refusal a validator gives an object, or None when it takes it."""
    try:
        validate(obj)
    except C.ContractError as exc:
        return exc.code
    return None


def paths(obj: object, base: str = "") -> set[str]:
    """Every member path of a JSON value, in the tutorial's notation."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for name, value in obj.items():
            here = f"{base}.{name}" if base else name
            found.add(here)
            if name not in FREE:
                found |= paths(value, here)
    elif isinstance(obj, list) and any(isinstance(one, dict) for one in obj):
        for one in obj:
            found |= {base + "[]" + path[len(base):] for path in paths(one, base)}
    return found


def field_tables(doc: str) -> dict[str, list[list[str]]]:
    tables = {schema: table(rows) for schema, rows in tagged(doc, "fields")}
    assert set(tables) == set(FULL), "one field table per format"
    return tables


def test_the_full_examples_behind_the_field_tables_are_valid():
    for schema, (validate, full) in FULL.items():
        assert code_of(validate, copy.deepcopy(full)) is None, schema


@pytest.mark.parametrize("schema", sorted(FULL))
def test_every_member_a_field_table_names_is_known_to_the_validator(doc, schema):
    """A name the format does not have is refused as an unknown member. So: each documented name, set to null in
    a full valid record, must not be refused that way, and a made-up name in the same place must be."""
    validate, full = FULL[schema]
    rows = field_tables(doc)[schema]
    assert rows[0][0] == "`schema`" and schema in rows[0][3]
    for row in rows:
        path = named(row[0])
        obj = copy.deepcopy(full)
        where, name = holder(obj, path)
        where[name.removesuffix("[]")] = None
        assert code_of(validate, obj) != "unknown_member", f"{schema}: {path} is not a member"
        obj = copy.deepcopy(full)
        where, name = holder(obj, path)
        where[name.removesuffix("[]") + "_x"] = None
        assert code_of(validate, obj) == "unknown_member", f"{schema}: the check does not reach {path}"


@pytest.mark.parametrize("schema", sorted(FULL))
def test_every_required_cell_is_what_the_validator_does(doc, schema):
    """Left out of a full valid record, a member marked `yes` is refused as missing and a member marked `no` is
    not missed. Cells that state a condition are not checked here."""
    validate, full = FULL[schema]
    checked = 0
    for row in field_tables(doc)[schema]:
        path, required = named(row[0]), row[2]
        obj = copy.deepcopy(full)
        where, name = holder(obj, path)
        where.pop(name.removesuffix("[]"), None)
        if required == "yes":
            code = code_of(validate, obj)
            assert code is not None if path == "schema" else code == "missing_member", f"{schema}: {path}: {code}"
        elif required.startswith("no"):
            assert code_of(validate, obj) is None, f"{schema}: {path} is marked optional and is missed"
        else:
            continue
        checked += 1
    assert checked >= 5


def test_the_field_tables_leave_no_member_of_the_examples_out(doc):
    """Every member used by the full records above and by the committed example files has a row."""
    tables = {schema: {named(row[0]).replace("[]", "") for row in rows} for schema, rows in field_tables(doc).items()}
    used = {schema: paths(full) for schema, (_, full) in FULL.items()}
    for folder in ("study", "scales"):
        for name, schema in (("rubric.json", C.SCHEMA_RUBRIC), ("judges.json", C.SCHEMA_JUDGES),
                             ("value.json", C.SCHEMA_VALUE)):
            if (EX / folder / name).is_file():
                used[schema] |= paths(json.loads((EX / folder / name).read_text(encoding="utf-8")))
        for name, schema in ((C.RUNS_FILE, C.SCHEMA_RUNS), (C.SCORES_FILE, C.SCHEMA_SCORES),
                             (C.PAIRS_FILE, C.SCHEMA_PAIRS), (C.ITEMS_FILE, C.SCHEMA_ITEMS)):
            if (EX / folder / name).is_file():
                for line in (EX / folder / name).read_text(encoding="utf-8").splitlines():
                    used[schema] |= paths(json.loads(line))
    for schema, names in used.items():
        missing = {path.replace("[]", "") for path in names} - tables[schema]
        assert not missing, f"{schema}: no row for {sorted(missing)}"


# ---------------------------------------------------------------------------------------------------
# CSV columns against the import

CSV_ROW = {
    "scores": {"run": "r102", "step": "final", "component": "depth", "judge": "reviewer-1", "call": "h0900",
               "value": "6", "item": "", "of": "", "observed_at": "2026-03-02T15:00:00+00:00", "id": "h0900",
               "replaces": "", "assignment": "", "rubric": "report@2", "cost_usd": "0"},
    "pairs": {"a": "r101", "b": "r102", "step": "final", "component": "overall", "judge": "reviewer-1",
              "call": "h0901", "shown_first": "a", "verdict": "", "choice": "right",
              "observed_at": "2026-03-02T15:01:00+00:00", "id": "h0901", "replaces": "", "assignment": "",
              "rubric": "report@2", "cost_usd": "0"},
}


def import_csv(tmp_path: Path, kind: str, row: dict) -> tuple[int, dict]:
    """Import one CSV row into a new copy of `scales`: the exit code and the JSON the command prints."""
    study = tmp_path / f"study-{len(list(tmp_path.iterdir()))}"
    shutil.copytree(EX / "scales", study)
    sheet = study.with_suffix(".csv")
    with open(sheet, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    code, out, _ = lm("eval", "scores", "import", "--csv" if kind == "scores" else "--pairs-csv", sheet,
                      "--rubric", study / "rubric.json", "--out", study, "--json")
    got = json.loads(out)
    if code == 0:
        assert check(study)[0] == 0
    return code, got


@pytest.mark.parametrize("kind", ["scores", "pairs"])
def test_every_csv_column_the_tutorial_names_is_read(doc, kind, tmp_path):
    """A sheet with every documented column comes in whole, with no column ignored. A column of another name is
    reported as ignored. A row without a column marked `yes` does not come in."""
    (rows,) = [table(rows) for name, rows in tagged(doc, "columns") if name == kind]
    columns = [named(row[0]) for row in rows]
    known = (S.SCORE_COLUMNS if kind == "scores" else S.PAIR_COLUMNS + ("choice",)) + S.MORE_COLUMNS
    assert sorted(columns) == sorted(known)
    counted = "readings" if kind == "scores" else "pairs"
    row = {column: CSV_ROW[kind][column] for column in columns}
    code, got = import_csv(tmp_path, kind, row)
    assert code == 0 and got[counted]["new"] == 1 and "ignored_columns" not in got, got
    code, got = import_csv(tmp_path, kind, {**row, "confidence": "high"})
    assert code == 0 and list(got["ignored_columns"].values()) == [["confidence"]], got
    for cells in rows:
        if cells[1] == "yes":
            code, got = import_csv(tmp_path, kind, {k: v for k, v in row.items() if k != named(cells[0])})
            assert code != 0 or got[counted]["new"] == 0, f"{cells[0]} is marked required and the row came in"


def test_a_choice_is_read_by_the_output_shown_first(tmp_path):
    """`left` is the output shown first and `right` the other one, as the tutorial says."""
    for first, choice, verdict in (("a", "left", "a"), ("a", "right", "b"), ("b", "left", "b"), ("b", "tie", "tie")):
        folder = tmp_path / f"{first}-{choice}"
        folder.mkdir()
        code, got = import_csv(folder, "pairs", {**CSV_ROW["pairs"], "shown_first": first, "choice": choice})
        assert code == 0, got
        pairs = C.read_lines(folder / "study-0" / "pairs.jsonl", C.SCHEMA_PAIRS)
        assert [row["verdict"] for row in pairs if row["id"] == "h0901"] == [verdict]


def test_a_value_changed_in_a_sheet_without_ids_comes_in_as_a_repeat(tmp_path):
    """The trap the CSV reference names: with no `id` column the id of a row is made from its content, so a
    changed value is a new row beside the old one, under its own call, and nothing is replaced."""
    study, sheet = tmp_path / "study", tmp_path / "sheet.csv"
    shutil.copytree(EX / "scales", study)
    for value, new in (("6", 1), ("6", 0), ("7", 1)):
        sheet.write_text("run,component,judge,value,observed_at\n"
                         f"r102,depth,reviewer-1,{value},2026-03-02T16:00:00+00:00\n", encoding="utf-8")
        code, out, err = lm("eval", "scores", "import", "--csv", sheet, "--rubric", study / "rubric.json", "--out",
                            study, "--json")
        assert code == 0 and json.loads(out)["readings"]["new"] == new, (value, out, err)
    assert check(study)[0] == 0
    rows = [row for row in C.read_lines(study / C.SCORES_FILE, C.SCHEMA_SCORES)
            if (row["judge"], row["run"], row["component"]) == ("reviewer-1", "r102", "depth")]
    assert sorted(row["value"] for row in rows) == [6, 7] and len({row["call"] for row in rows}) == 2
    assert not any("replaces" in row for row in rows)


# ---------------------------------------------------------------------------------------------------
# The mistakes the tutorial names, made on a copy of `scales`

def broken(tmp_path: Path, change, source: str = "scales") -> Path:
    """A copy of an example study with one change made to its tables."""
    study = tmp_path / "study"
    shutil.copytree(EX / source, study)
    names = [name for name in ("runs", "scores", "pairs", "items") if (study / f"{name}.jsonl").is_file()]
    tables = {name: [json.loads(line) for line in (study / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
              for name in names}
    change(tables)
    for name in names:
        (study / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tables[name]),
                                             encoding="utf-8")
    return study


def score(n: int):
    """A change to line `n` of `scales/scores.jsonl` (1 is the first line)."""
    def with_change(**members):
        def change(tables):
            row = tables["scores"][n - 1]
            for name, value in members.items():
                if value is None:
                    del row[name]
                else:
                    row[name] = value
        return change
    return with_change


def _another(tables, **members):
    tables["scores"].append({**tables["scores"][3], **members})


def _other_input(tables):
    tables["runs"].append({**tables["runs"][1], "run": "r103", "input": "doc-02"})
    tables["pairs"][0]["b"] = "r103"


# name in the tutorial: (the change, the code of the refusal)
MISTAKES = {
    "call-twice": (lambda t: _another(t, id="c0004.depth.again", value=8), "call_twice"),
    "id-twice": (lambda t: _another(t, call="c0099"), "duplicate"),
    "no-call": (score(4)(call=None), "missing_member"),
    "no-id": (score(4)(id=None), "missing_member"),
    "no-rubric": (score(4)(rubric=None), "missing_member"),
    "no-time": (score(4)(observed_at=None), "missing_member"),
    "bad-time": (score(4)(observed_at="2026-03-02T09:01:00"), "bad_time"),
    "binary-word": (score(1)(value="pass"), "bad_value"),
    "level-case": (score(2)(value="Good"), "bad_value"),
    "level-text": (score(3)(value="4"), "bad_value"),
    "fraction-percent": (score(6)(value=80), "bad_value"),
    "linear-range": (score(4)(value=75), "bad_value"),
    "unknown-judge": (score(4)(judge="judge-new"), "unknown_judge"),
    "unknown-component": (score(4)(component="Depth"), "unknown_component"),
    "unknown-run": (score(4)(run="r999"), "no_run"),
    "rubric-version": (score(4)(rubric="report@1"), "unknown_rubric"),
    "extra-member": (score(4)(model="model-large"), "unknown_member"),
    "items-both": (lambda t: t["scores"].append({k: v for k, v in {**t["scores"][7], "id": "c0008.covers", "value": 2,
                                                                  "of": 3}.items() if k != "item"}),
                   "items_and_count"),
    "unknown-item": (score(8)(item="doc-01.none"), "unknown_item"),
    "pair-inputs": (_other_input, "different_inputs"),
    "replaces-unknown": (score(12)(replaces="c0000.depth"), "replaces_unknown"),
    "replaces-self": (score(12)(replaces="c0010.depth"), "replaces_self"),
}


def refusal(study: Path) -> tuple[list[str], str]:
    """What `eval scores check` prints on stderr for a broken study, and the code of the refusal. A line that is
    not a valid record is refused when the file is read; anything else is a problem of the check's report."""
    code, _, err = check(study)
    assert code == 1, err
    try:
        C.read_study(study)
    except C.ContractError as exc:
        return err.splitlines(), exc.code
    (problem,) = report(study)["problems"]
    return err.splitlines(), problem["code"]


def test_every_refusal_shown_is_the_message_the_check_prints(doc, tmp_path):
    blocks = tagged(doc, "refusal")
    used = [name.strip() for names, _ in blocks for name in names.split(",")]
    assert sorted(used) == sorted(MISTAKES), "each mistake is shown once"
    for names, block in blocks:
        real = []
        for name in (name.strip() for name in names.split(",")):
            change, code = MISTAKES[name]
            folder = tmp_path / name
            folder.mkdir()
            lines, got = refusal(broken(folder, change))
            assert got == code, (name, got, lines)
            assert len(lines) == 1 and lines[0].startswith("error: "), (name, lines)
            real += lines
        assert block == real, names


def test_whole_number_levels_are_refused_as_text_and_as_a_decimal(tmp_path):
    """The tutorial says to write a number level as `4`, not `"4"` and not `4.0`."""
    for n, value in enumerate(("4", 4.0)):
        folder = tmp_path / str(n)
        folder.mkdir()
        assert refusal(broken(folder, score(3)(value=value)))[1] == "bad_value"
    assert check(broken(tmp_path, score(3)(value=4)))[0] == 0


def _own_calls(tables):
    for n, row in enumerate(tables["scores"]):
        if row["judge"] == "judge-cheap":
            row["call"] = f"own-{n}"


def _tenth(tables):
    for row in tables["scores"]:
        if row["judge"] == "judge-cheap" and row["component"] == "depth":
            row["value"] = round(row["value"] / 10, 2)


def _arm_in_run_id(tables):
    new = {row["run"]: f'{row["arm"]}-{row["run"]}' for row in tables["runs"]}
    for row in tables["runs"]:
        row["run"] = new[row["run"]]
    for row in tables["scores"]:
        row["run"] = new[row["run"]]
    for row in tables["pairs"]:
        row["a"], row["b"] = new[row["a"]], new[row["b"]]


def _own_item_calls(tables):
    for row in tables["scores"]:
        if "item" in row:
            row["call"] = row["id"]


def _fix_under_a_new_call(tables):
    for row in tables["scores"]:
        if "replaces" in row:
            row["call"] = row["id"]


def _unmarked(tables):
    for row in tables["runs"]:
        row.pop("simulated", None)


@pytest.mark.parametrize("change, source", [
    (_own_calls, "study"), (_tenth, "study"), (_arm_in_run_id, "study"), (_own_item_calls, "scales"),
    (_fix_under_a_new_call, "study"), (_fix_under_a_new_call, "scales"), (_unmarked, "study")],
    ids=["panel-calls-split", "wrong-scale-in-range", "arm-in-run-id", "item-calls-split", "fix-under-a-new-call",
         "fix-under-a-new-call-scales", "made-up-runs-unmarked"])
def test_what_the_tutorial_says_the_check_lets_through_passes(change, source, tmp_path):
    """The tutorial warns that these pass. If the check learns to refuse one, the warning goes."""
    code, out, err = check(broken(tmp_path, change, source))
    assert code == 0 and out.rstrip().endswith("\nok"), err


def test_a_misspelt_judge_passes_only_without_the_judges_file(tmp_path):
    study = broken(tmp_path, score(4)(judge="judge-new"))
    assert lm("eval", "scores", "check", study, "--rubric", study / "rubric.json")[0] == 0
    assert check(study)[0] == 1


def test_a_shared_call_and_a_mixed_arm_are_named_and_the_check_still_passes(tmp_path):
    """A call id used by two judges is a warning. An arm whose runs differ in a setting is listed."""
    def change(tables):
        tables["scores"][6]["call"] = "c0011"           # a `judge-main` reading under the call of a `judge-cheap` panel
        steps = copy.deepcopy(tables["runs"][0]["steps"])
        steps[1]["effort"] = "low"
        tables["runs"].append({**tables["runs"][0], "run": "r103", "input": "doc-02", "steps": steps})
    study = broken(tmp_path, change)
    code, out, _ = check(study)
    assert code == 0
    assert 'warning: scores.jsonl: call "c0011": shared by 2 pairs of judge and target' in out
    assert "mixed_arm: short-prompt: 2 settings over 2 runs" in out


@needs_engine
def test_the_fit_leaves_out_a_row_the_check_refuses(tmp_path):
    """`eval fit` counts a reading by an unknown judge as not used, where `eval scores check` refuses it."""
    def change(tables):
        tables["scores"][0]["judge"] = "judge-new"
    study = broken(tmp_path, change, "study")
    assert check(study)[0] == 1
    code, out, err = lm("eval", "fit", study, "--rubric", study / "rubric.json", "--judges", study / "judges.json",
                        "--allow-simulated", "--check-only")
    assert code == 0, err
    assert "readings: 319 of 321 used; pairs: 16 of 16 used; 2 rows not used" in out      # 1 replaced, 1 unknown judge


# ---------------------------------------------------------------------------------------------------
# What the recording rules are for: the model the fit builds

def design(study: Path, **options) -> M.Design:
    """The model `eval fit` builds from a study folder (the examples are simulated, so that is allowed)."""
    rubric = C.read_document(study / "rubric.json", C.SCHEMA_RUBRIC)
    judges = C.read_document(study / "judges.json", C.SCHEMA_JUDGES)
    return M.build(C.read_study(study), rubric, judges, allow_simulated=True, **options)


def item_readings(built: M.Design) -> list[tuple[str, int, int]]:
    """`(judge, passed, of)` of every `items` reading of a model."""
    data, judges = built.data, [row["id"] for row in built.judges["judges"]]
    return sorted((judges[j], int(k), int(n)) for kind, j, k, n in
                  zip(data["g_kind"], data["g_j"], data["g_k"], data["g_n"]) if kind == M.ITEMS)


def _drop(reading_id: str):
    def change(tables):
        tables["scores"] = [row for row in tables["scores"] if row["id"] != reading_id]
    return change


def test_the_item_lines_of_one_call_are_one_reading_of_the_model(tmp_path):
    """Call `c0008` of `scales` has three item lines, passed 1, 0, 1: the model reads one reading, 2 of 3. With
    the correction of its `owner` line, which keeps the call, it reads 3 of 3. Under a call id per line, which
    the check lets through, the same lines are three readings of one item each."""
    lines = [json.loads(line) for line in (EX / "scales" / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    first = [(row["item"], row["value"]) for row in lines if row["call"] == "c0008" and "replaces" not in row]
    assert first == [("doc-01.deadline", 1), ("doc-01.owner", 0), ("doc-01.cost", 1)]
    count = ("judge-cheap", 2, 3)                                   # the count line of the panel call
    for name in ("as-written", "split"):
        (tmp_path / name).mkdir()
    before = broken(tmp_path / "as-written", _drop("c0008.covers.owner.fix1"))
    assert item_readings(design(before)) == [count, ("judge-main", 2, 3)]
    assert item_readings(design(EX / "scales")) == [count, ("judge-main", 3, 3)]

    def split(tables):
        _drop("c0008.covers.owner.fix1")(tables)
        _own_item_calls(tables)
    study = broken(tmp_path / "split", split)
    assert check(study)[0] == 0
    assert item_readings(design(study)) == [count, ("judge-main", 0, 1), ("judge-main", 1, 1), ("judge-main", 1, 1)]


def generator():
    """`record_study.py`, loaded as a module. Compiled here, so that no bytecode file is left beside it."""
    path = EX / "record_study.py"
    module = types.ModuleType("record_study")
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def test_the_recorded_calls_are_the_requests_the_generator_made(tmp_path):
    """The generator keeps a list of every request it made of a judge. The files show the same requests: each
    `call` is one request, by that judge, about that output, for those components, and each judge's `mode` in
    the judges file is how it was asked. Counts by hand: `judge-main` 32 x 4 and 16 x 2 repeats, `judge-cheap`
    32 and 8 panel requests, the reviewer 12 x 4 and 4 x 2 second looks; 16 and 6 comparisons."""
    study = tmp_path / "my-study"
    made = generator().main(study)
    for name in (C.RUNS_FILE, C.SCORES_FILE, C.PAIRS_FILE):
        assert (study / name).read_bytes() == (EX / "study" / name).read_bytes(), name
    code, _, err = lm("eval", "scores", "import", "--csv", tmp_path / "my-study-reviewer.csv", "--pairs-csv",
                      tmp_path / "my-study-reviewer-pairs.csv", "--rubric", study / "rubric.json", "--out", study)
    assert code == 0, err

    asked = Counter((judge, run, frozenset(parts)) for kind, judge, run, parts in
                    (entry for entry in made.asked if entry[0] == "score"))
    by_call: dict[str, list[dict]] = {}
    for row in C.read_lines(study / C.SCORES_FILE, C.SCHEMA_SCORES):
        by_call.setdefault(row["call"], []).append(row)
    recorded = Counter()
    for call, rows in by_call.items():
        assert len({(row["judge"], row["run"]) for row in rows}) == 1, call
        recorded[(rows[0]["judge"], rows[0]["run"], frozenset(row["component"] for row in rows))] += 1
    assert recorded == asked
    assert sum(asked.values()) == (32 * 4 + 16 * 2) + (32 + 8) + (12 * 4 + 4 * 2) == 256

    sizes: dict[str, set[int]] = {}
    for judge, _, parts in asked:
        sizes.setdefault(judge, set()).add(len(parts))
    modes = {row["id"]: row["mode"] for row in C.read_document(study / "judges.json", C.SCHEMA_JUDGES)["judges"]}
    assert modes == {"reviewer-1": "per_component", "judge-main": "per_component", "judge-cheap": "panel"}
    assert sizes == {"reviewer-1": {1}, "judge-main": {1}, "judge-cheap": {4}}

    asked_pairs = Counter(entry[1:] for entry in made.asked if entry[0] == "pair")
    pairs = C.read_lines(study / C.PAIRS_FILE, C.SCHEMA_PAIRS)
    assert len({row["call"] for row in pairs}) == len(pairs) == 16 + 6
    assert Counter((row["judge"], row["a"], row["b"], row["shown_first"]) for row in pairs) == asked_pairs
    assert len(made.asked) == 256 + 22


def test_a_correction_keeps_its_call_so_a_panel_request_stays_one_request(tmp_path):
    """The model has one call variable per panel request. The generated study has 32 + 8 panel requests and
    one corrected line; `scales` has one panel request of `judge-cheap` and one sitting of `reviewer-2` with a
    corrected line. The corrections keep their calls, so the counts are 40 and 2. Under a new call id, which
    the check lets through, each correction adds a request that was never made."""
    for name in ("study", "scales"):
        (tmp_path / name).mkdir()
    assert design(EX / "study").data["n_pc"] == 32 + 8
    assert design(broken(tmp_path / "study", _fix_under_a_new_call, "study")).data["n_pc"] == 32 + 8 + 1
    assert design(EX / "scales").data["n_pc"] == 2
    assert design(broken(tmp_path / "scales", _fix_under_a_new_call)).data["n_pc"] == 2 + 1
    got = report(EX / "scales")
    assert sorted((row["table"], row["id"], row["by"]) for row in got["not_read"]) == [
        ("pairs", "h0003.overall", "h0003.overall.fix1"), ("readings", "c0008.covers.owner", "c0008.covers.owner.fix1"),
        ("readings", "h0001", "h0001.fix1"), ("readings", "h0002.depth", "h0002.depth.fix1")]
    for table in ("scores", "pairs"):
        rows = {row["id"]: row for row in C.read_lines(EX / "scales" / f"{table}.jsonl",
                                                       C.SCHEMA_SCORES if table == "scores" else C.SCHEMA_PAIRS)}
        fixes = [row for row in rows.values() if "replaces" in row]
        assert fixes and all(row["call"] == rows[row["replaces"]]["call"] and row["id"] != row["replaces"]
                             for row in fixes), table


def test_the_helpers_write_a_correction_under_the_old_call_and_a_new_answer_under_a_new_one(tmp_path):
    """`record_reading` and `record_pair` take the id of the line apart from the call. A correction with its
    own id keeps the call, and the panel request count stays. Without its own id it would be the old line
    again, which is refused. A new answer of the judge is a new call: one more panel request."""
    record, study = generator(), tmp_path / "study"
    shutil.copytree(EX / "study", study)
    at = "2026-03-02T12:00:00+00:00"
    line = dict(run="r001", component="overall", judge="judge-cheap", observed_at=at)
    assert record.record_reading(study, call="c0161", value=5.0, **line) == "c0161.overall"
    code, _, err = check(study)
    assert code == 1 and '"c0161.overall"' in err and "is already on line" in err, err

    shutil.copyfile(EX / "study" / C.SCORES_FILE, study / C.SCORES_FILE)
    fixed = record.record_reading(study, call="c0161", value=5.0, reading_id="c0161.overall.fix1",
                                  replaces="c0161.overall", **line)
    assert fixed == "c0161.overall.fix1"
    got = report(study)
    assert got["ok"] is True and got["warnings"] == []
    assert (got["counts"]["readings"], got["counts"]["readings_read"]) == (322, 320)
    assert design(study).data["n_pc"] == 32 + 8

    for component, value in (("correct", 1), ("clarity", "ok"), ("depth", 5.5), ("overall", 6.0)):
        new = record.record_reading(study, run="r001", component=component, judge="judge-cheap", call="c0999",
                                    value=value, observed_at=at)
        assert new == f"c0999.{component}"
    assert report(study)["ok"] is True and design(study).data["n_pc"] == 32 + 8 + 1

    first = C.read_lines(study / C.PAIRS_FILE, C.SCHEMA_PAIRS)[0]
    pair = dict(a=first["a"], b=first["b"], judge="judge-main", call="c0998", shown_first="a", observed_at=at)
    assert record.record_pair(study, verdict="a", **{**pair, "call": "c0997"}) == "c0997"
    assert record.record_pair(study, component="depth", verdict="a", **pair) == "c0998.depth"
    assert record.record_pair(study, component="overall", verdict="tie", **pair) == "c0998.overall"
    assert record.record_pair(study, component="overall", verdict="b", pair_id="c0998.overall.fix1",
                              replaces="c0998.overall", **pair) == "c0998.overall.fix1"
    got = report(study)
    assert got["ok"] is True and got["warnings"] == []
    assert (got["counts"]["pairs"], got["counts"]["pairs_read"]) == (16 + 4, 16 + 3)
    live = {row["id"]: row for row in C.live(C.read_lines(study / C.PAIRS_FILE, C.SCHEMA_PAIRS), C.PAIRS_FILE)[0]}
    assert {row["id"] for row in live.values() if row["call"] == "c0998"} == {"c0998.depth", "c0998.overall.fix1"}
    assert live["c0998.overall.fix1"]["verdict"] == "b"
    record.record_pair(study, component="depth", verdict="b", pair_id="c0998.depth.again", **pair)
    assert [row["code"] for row in report(study)["problems"]] == ["call_twice"]


def same(one: object, other: object) -> bool:
    """Whether two values of a model's engine input are equal: arrays by shape, type and content."""
    if isinstance(one, np.ndarray) or isinstance(other, np.ndarray):
        return (isinstance(one, np.ndarray) and isinstance(other, np.ndarray) and one.shape == other.shape
                and one.dtype == other.dtype and bool(np.array_equal(one, other)))
    if isinstance(one, (list, tuple)):
        return (type(one) is type(other) and len(one) == len(other)
                and all(same(a, b) for a, b in zip(one, other)))
    return type(one) is type(other) and one == other


def _split_pair_calls(tables):
    """Give every verdict of a pair call with several verdicts a call of its own. A correction keeps the
    call of the line it corrects."""
    by_id = {row["id"]: row for row in tables["pairs"]}
    for row in tables["pairs"]:
        if "replaces" not in row and row["call"] == "h0003":
            row["call"] = row["id"]
    for row in tables["pairs"]:
        if "replaces" in row:
            row["call"] = by_id[row["replaces"]]["call"]


def _split_score_panel(tables):
    for row in tables["scores"]:
        if row["call"] == "c0011":
            row["call"] = row["id"]


PAIR_LIMIT = "the current fit does not model noise shared by the verdicts of one pair request"


def test_the_fit_builds_the_same_model_for_shared_and_split_pair_calls_and_the_documents_say_so(doc, guide, tmp_path):
    """A limit of the current fit, which the tutorial states. Call `h0003` of `scales` holds two verdicts of
    one sitting. The pair part of the model has no term for a pair call: every value of the engine input is
    the same when each verdict gets a call of its own. (The comparison can see a difference: a score panel
    split the same way changes the input.) So the page, the guide and the pair helper must say that a shared
    pair call id is kept for the record and not fitted, recommend one verdict per pair request, and say that
    true call ids stay.

    When the engine learns to fit noise shared by the verdicts of one pair call, the first part of this test
    fails. Change the documents then, and this test with them."""
    for name in ("pairs", "scores"):
        (tmp_path / name).mkdir()
    pairs = C.read_lines(EX / "scales" / C.PAIRS_FILE, C.SCHEMA_PAIRS)
    shared = [row for row in pairs if row["call"] == "h0003"]
    assert sorted((row["id"], row["component"], row.get("replaces")) for row in shared) == [
        ("h0003.depth", "depth", None), ("h0003.overall", "overall", None),
        ("h0003.overall.fix1", "overall", "h0003.overall")]

    split = broken(tmp_path / "pairs", _split_pair_calls)
    rows = {row["id"]: row for row in C.read_lines(split / C.PAIRS_FILE, C.SCHEMA_PAIRS)}
    assert (rows["h0003.depth"]["call"], rows["h0003.overall"]["call"], rows["h0003.overall.fix1"]["call"]) == (
        "h0003.depth", "h0003.overall", "h0003.overall")
    assert check(split)[0] == 0 and report(split)["counts"] == report(EX / "scales")["counts"]
    as_recorded, as_split = design(EX / "scales"), design(split)
    assert as_recorded.counts["pairs_used"] == as_split.counts["pairs_used"] == 4
    assert set(as_recorded.data) == set(as_split.data) and len(as_recorded.data) > 40
    differing = sorted(name for name in as_recorded.data if not same(as_recorded.data[name], as_split.data[name]))
    assert differing == [], ("the model now tells a shared pair call from split ones: update the limit in "
                             "docs/eval-recording.md, the guide and record_pair", differing)
    assert not any("call" in name for name in as_recorded.data if name.startswith("p_"))

    control = design(broken(tmp_path / "scores", _split_score_panel))
    assert not same(as_recorded.data["n_pc"], control.data["n_pc"])           # the comparison is not blind
    assert not same(as_recorded.data["g_pc"], control.data["g_pc"])

    for name, text in (("the page", doc), ("the guide", guide), ("record_pair", generator().record_pair.__doc__)):
        flat = " ".join(text.split())
        assert PAIR_LIMIT in flat, f"{name} does not state the limit"
        assert "one overall verdict per pair request" in flat, name
        assert "one request per compared component" in flat, name
        assert "0.4.4 does not" not in flat and "0.4.4 fits" not in flat, name
    flat_doc, flat_guide = " ".join(doc.split()), " ".join(guide.split())
    for flat in (flat_doc, flat_guide):
        assert "record-format example" in flat and "loopmath 0.4.x" in flat and "true call id" in flat
    head = "### One pair request, several verdicts: a limit of the current fit"
    section = " ".join(doc[doc.index(head):doc.index("## Recording from your pipeline")].split())
    for words in (PAIR_LIMIT, "one request per compared component", "one overall verdict per pair request",
                  "keep their true call id", "does not fix the model", "loopmath 0.4.x"):
        assert words in section, words
    before = " ".join(doc[doc.index("## Recording comparisons"):doc.index(head)].split())
    assert "The last three lines are a record-format example" in before


def test_the_helpers_leave_out_what_has_no_value(tmp_path):
    """No member is written as null, and `simulated` is written only when asked for."""
    record = generator()
    record.record_run(tmp_path, run="r900", input="q90", arm="baseline", steps=[record.ARMS["baseline"]])
    record.record_run(tmp_path, run="r901", input="q90", arm="candidate", steps=[record.ARMS["candidate"]],
                      simulated=True)
    record.record_reading(tmp_path, run="r900", component="depth", judge="judge-main", call="c9000", value=5.0,
                          observed_at="2026-03-02T12:00:00+00:00")
    text = (tmp_path / C.RUNS_FILE).read_text(encoding="utf-8") + (tmp_path / C.SCORES_FILE).read_text(encoding="utf-8")
    assert "null" not in text
    real, made_up = C.read_lines(tmp_path / C.RUNS_FILE, C.SCHEMA_RUNS)
    raw = [json.loads(line) for line in (tmp_path / C.RUNS_FILE).read_text(encoding="utf-8").splitlines()]
    assert "simulated" not in raw[0] and "cost_usd" not in raw[0] and raw[1]["simulated"] is True
    assert real["simulated"] is False and made_up["simulated"] is True


# ---------------------------------------------------------------------------------------------------
# Simulated runs, and null

def test_every_example_run_is_marked_simulated(guide):
    """All example data is made up, and every run of it says so: the study folders, the OCP documents and the
    run lines of the agent guide."""
    seen = 0
    for folder in ("study", "scales"):
        for line in (EX / folder / C.RUNS_FILE).read_text(encoding="utf-8").splitlines():
            assert json.loads(line)["simulated"] is True, (folder, line[:60])
            seen += 1
    for name in ("r201.ocp.json", "r202.ocp.json"):
        document = json.loads((EX / "ocp" / name).read_text(encoding="utf-8"))
        assert isinstance(document["run"]["ext"]["dev.loopmath.simulate"], dict), name
        seen += 1
    for line in re.findall(r'^\{"schema": "loopmath\.eval-runs/1".*$', guide, flags=re.M):
        assert json.loads(line)["simulated"] is True, line[:60]
        seen += 1
    assert seen == 32 + 2 + 2 + 2


@pytest.mark.parametrize("folder", ["study", "scales"])
def test_the_model_takes_an_example_only_when_simulated_runs_are_allowed(folder):
    study = EX / folder
    rubric = C.read_document(study / "rubric.json", C.SCHEMA_RUBRIC)
    judges = C.read_document(study / "judges.json", C.SCHEMA_JUDGES)
    with pytest.raises(C.ContractError) as refused:
        M.build(C.read_study(study), rubric, judges)
    assert refused.value.code == "simulated"
    assert M.build(C.read_study(study), rubric, judges, allow_simulated=True).counts["readings_used"] > 0


@needs_engine
def test_the_fit_refuses_the_example_without_the_flag_as_the_tutorial_shows(doc, walk, monkeypatch, tmp_path):
    """`eval fit` on the walk's study: refused without `--allow-simulated`, with the message the tutorial
    shows, and taken with it. The same study with the marks removed, as real runs would be, needs no flag."""
    root, _ = walk
    monkeypatch.chdir(root)
    code, out, err = lm("eval", "fit", "my-study", *STUDY_ARGS, "--check-only")
    assert code == 1 and out == "", (out, err)
    (block,) = [block for name, block in tagged(doc, "stderr") if name == "fit-refused"]
    assert block == err.splitlines() and "--allow-simulated" in err
    assert lm("eval", "fit", "my-study", *STUDY_ARGS, "--allow-simulated", "--check-only")[0] == 0
    real = broken(tmp_path, _unmarked, "study")
    code, _, err = lm("eval", "fit", real, "--rubric", real / "rubric.json", "--judges", real / "judges.json",
                      "--check-only")
    assert code == 0, err


def test_the_ocp_import_refuses_the_simulated_runs_without_the_flag(tmp_path):
    """Both OCP examples carry the simulate mark: without `--allow-simulated` nothing is written."""
    rubric = EX / "study" / "rubric.json"
    code, out, err = lm(*OCP_IMPORT, "--rubric", rubric, "--out", tmp_path / "refused")
    assert code == 1 and "nothing written" in out, (out, err)
    assert err.count("--allow-simulated takes simulated runs") == 2
    assert not (tmp_path / "refused" / C.RUNS_FILE).exists()
    code, out, err = lm(*OCP_IMPORT, "--rubric", rubric, "--out", tmp_path / "taken", "--allow-simulated")
    assert code == 0 and "runs: 2 new" in out, err


def test_null_is_refused_for_mode_and_blinded_and_leaving_them_out_is_fine(doc, guide):
    """The tutorial says to leave out an optional member with no value, not to write null. `mode` and `blinded`
    of a judge show why: left out they get their defaults, set to null they are refused."""
    base = {"schema": C.SCHEMA_JUDGES, "judges": [{"id": "reviewer-1", "kind": "person"}]}
    (judge,) = C.validate_judges(copy.deepcopy(base))["judges"]
    assert (judge["mode"], judge["blinded"]) == ("per_component", False)
    for name, code in (("mode", "bad_choice"), ("blinded", "type")):
        document = copy.deepcopy(base)
        document["judges"][0][name] = None
        assert code_of(C.validate_judges, document) == code, name
    assert "counts as absent" not in doc
    assert "Do not write `null`" in doc and '`"mode": null`' in doc and '`"blinded": null`' in doc
    assert "null" in guide


# ---------------------------------------------------------------------------------------------------
# The order of the page

def test_the_page_walks_first_and_keeps_the_reference_for_the_end(doc):
    """A contents list, the short version, then the checked walk. The field tables, the CSV columns and the
    refusals are in reference sections, which are the last sections of the page."""
    lines = doc.splitlines()
    heads = [line[3:] for line in lines if line.startswith("## ")]
    assert heads[:3] == ["Contents", "The short version", "A first study, start to end"]
    assert heads[-3:] == ["Reference: the files, member by member", "Reference: CSV columns",
                          "Reference: what the check refuses, and the fix"]
    assert all(not head.startswith("Reference") for head in heads[:-3])
    at = {head: n for n, line in enumerate(lines) for head in heads if line == "## " + head}
    assert at["A first study, start to end"] < 70
    contents = "\n".join(lines[at["Contents"]:at["The short version"]])
    for head in heads[1:]:
        assert f"](#{slug(head)})" in contents, head
    walked = "\n".join(lines[at["A first study, start to end"]:at["The words"]])
    for command in TUTORIAL_COMMANDS:
        assert "$ loopmath " + command in walked, command
    for name in ("record", "rubric-check", "check"):
        assert f"<!-- output: {name} -->" in walked, name
    reference = at["Reference: the files, member by member"]
    marks = [n for n, line in enumerate(lines) if re.match(r"<!-- (fields|columns|refusal): ", line)]
    assert len(marks) >= 7 + 2 + 9 and min(marks) > reference
    assert reference > len(lines) * 0.55


# ---------------------------------------------------------------------------------------------------
# The text itself

def real_parser() -> argparse.ArgumentParser:
    """The top-level parser, as `cli.main` builds it."""
    class Built(Exception):
        pass

    def grab(self, *args, **kwargs):
        raise Built(self)

    real = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = grab
    try:
        cli.main(["--help"])
    except Built as got:
        return got.args[0]
    finally:
        argparse.ArgumentParser.parse_args = real
    raise AssertionError("cli.main did not build a parser")


def written_commands(text: str) -> list[str]:
    """Every `loopmath ...` line of the code blocks of a document."""
    found, inside = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
        elif inside and line.lstrip("$ ").startswith("loopmath "):
            found.append(line.lstrip("$ ").strip())
    return found


TUTORIAL_COMMANDS = ("eval rubric check", "eval scores import", "eval scores check", "eval fit", "eval judges",
                     "eval value", "eval labels next")
GUIDE_COMMANDS = ("eval rubric check", "eval scores import", "eval scores check")     # the guide stops at the check


@pytest.mark.parametrize("path, at_least, named", [(DOC, 10, TUTORIAL_COMMANDS), (GUIDE, 5, GUIDE_COMMANDS)],
                         ids=["tutorial", "guide"])
def test_every_command_shown_parses(path, at_least, named, capsys):
    top = real_parser()
    lines = written_commands(path.read_text(encoding="utf-8"))
    assert len(lines) >= at_least, lines
    for line in lines:
        try:
            top.parse_args(shlex.split(line)[1:])
        except SystemExit:
            raise AssertionError(f"{path.name}: does not parse: {line}: {capsys.readouterr().err.strip()[-300:]}")
    shown = " ".join(lines)
    for command in named:
        assert command in shown, f"{path.name} shows no {command}"


def slug(heading: str) -> str:
    """The anchor a Markdown heading gets: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def test_every_link_in_the_tutorial_lands(doc):
    """Links to a heading of this page or of `eval.md` name a real heading, and linked files are there."""
    anchors = {"": {slug(line.lstrip("#")) for line in doc.splitlines() if line.startswith("#")}}
    links = re.findall(r"\]\(([^)\s]+)\)", doc)
    assert len(links) >= 10
    for link in links:
        file, _, anchor = link.partition("#")
        if file:
            assert (DOC.parent / file).exists(), link
        if anchor:
            if file not in anchors:
                other = (DOC.parent / file).read_text(encoding="utf-8")
                anchors[file] = {slug(line.lstrip("#")) for line in other.splitlines() if line.startswith("#")}
            assert anchor in anchors[file], link


def test_the_tutorial_says_eval_is_experimental_and_links_the_limits(doc):
    assert "experimental in 0.4.4" in doc and "(eval.md#limits)" in doc
    assert "## Limits" in (DOC.parent / "eval.md").read_text(encoding="utf-8")


def test_no_em_dash_in_the_tutorial_the_guide_or_the_examples():
    files = [DOC, GUIDE, Path(__file__)] + [path for path in EX.rglob("*")
                                             if path.is_file() and "__pycache__" not in path.parts]
    assert len(files) == 3 + 17
    for path in files:
        assert EM_DASH not in path.read_text(encoding="utf-8"), path


# ---------------------------------------------------------------------------------------------------
# The agent guide

def guide_study(guide: str, folder: Path) -> tuple[Path, set[str]]:
    """The JSON blocks of the guide, written out as the study folder they describe, and the formats seen."""
    validators = {C.SCHEMA_RUBRIC: C.validate_rubric, C.SCHEMA_JUDGES: C.validate_judges,
                  C.SCHEMA_RUNS: C.validate_run, C.SCHEMA_SCORES: C.validate_score, C.SCHEMA_PAIRS: C.validate_pair,
                  C.SCHEMA_ITEMS: C.validate_item, C.SCHEMA_VALUE: C.validate_value}
    files = {C.SCHEMA_RUBRIC: "rubric.json", C.SCHEMA_JUDGES: "judges.json", C.SCHEMA_RUNS: C.RUNS_FILE,
             C.SCHEMA_SCORES: C.SCORES_FILE, C.SCHEMA_PAIRS: C.PAIRS_FILE, C.SCHEMA_ITEMS: C.ITEMS_FILE}
    blocks = re.findall(r"```json\n(.*?)```", guide, flags=re.S)
    assert len(blocks) >= 5
    study, seen = folder / "study", set()
    study.mkdir()
    for block in blocks:
        try:
            rows = [json.loads(block)]
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in block.splitlines() if line.strip()]
        for row in rows:
            validators[row["schema"]](row)
            seen.add(row["schema"])
            with open(study / files[row["schema"]], "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
    for name in ("rubric.json", "judges.json"):
        (study / name).write_text((study / name).read_text(encoding="utf-8").strip() + "\n", encoding="utf-8")
    return study, seen


def test_the_guide_examples_are_valid_records(guide, tmp_path):
    """Each JSON block of the guide validates under the format its `schema` names, and its blocks together make
    a study that passes `eval rubric check` and `eval scores check`."""
    study, seen = guide_study(guide, tmp_path)
    assert seen == {C.SCHEMA_RUBRIC, C.SCHEMA_JUDGES, C.SCHEMA_RUNS, C.SCHEMA_SCORES, C.SCHEMA_PAIRS, C.SCHEMA_ITEMS}
    code, _, err = lm("eval", "rubric", "check", study / "rubric.json", "--judges", study / "judges.json", "--items",
                      study / C.ITEMS_FILE)
    assert code == 0, err
    code, out, err = check(study)
    assert code == 0 and out.rstrip().endswith("\nok"), err
    got = report(study)
    assert got["warnings"] == [] and got["mixed_arm"] == []


def test_the_guide_item_example_is_one_reading_of_two_of_three(guide, tmp_path):
    """The guide shows a list of checks answered in one request: three item lines under one call, passed 1, 0
    and 1. The model reads them as one reading, 2 of 3. Its corrections keep the call of the line they
    correct, and its panel answer and its sitting stay one request each."""
    study, _ = guide_study(guide, tmp_path)
    rows = C.read_lines(study / C.SCORES_FILE, C.SCHEMA_SCORES)
    items = [row for row in rows if "item" in row]
    assert [row["value"] for row in items] == [1, 0, 1] and len({row["call"] for row in items}) == 1
    assert len({row["id"] for row in items}) == 3 and all(row["id"].startswith(row["call"] + ".") for row in items)
    built = design(study)
    assert item_readings(built) == [("judge-main", 2, 3)]
    by_id = {row["id"]: row for row in rows}
    fixes = [row for row in rows if "replaces" in row]
    assert fixes and all(row["call"] == by_id[row["replaces"]]["call"] for row in fixes)
    modes = {row["id"]: row["mode"] for row in built.judges["judges"]}
    panel_calls = {row["call"] for row in rows if modes[row["judge"]] == "panel"}
    assert built.data["n_pc"] == len(panel_calls) >= 2
    assert {modes[row["judge"]] for row in fixes} >= {"panel"}


def test_the_guide_names_the_rules_the_tutorial_gives(guide):
    """The agent version carries the same rules: experimental, one call id per call, repeats and corrections,
    neutral run ids, the check before the fit."""
    assert guide.startswith("# ") and "experimental" in guide and "loopmath guide eval" in guide
    for words in ("run `--help`", "`call`", "`replaces`", "repeat", "neutral", "`ext`", "eval scores check",
                  "`cost_usd`", "`cost_usd_per_call`", "## Never", "`item`", "`simulated`", "--allow-simulated",
                  "sitting", "null"):
        assert words in guide, words


def test_the_guide_gives_the_refusals_their_real_codes_and_words(guide, tmp_path):
    """Each fix the guide lists starts from words the check really prints for that mistake."""
    quoted = re.findall(r"^\| `([^`]+)` \|", guide, flags=re.M)
    assert len(quoted) >= 8, quoted
    printed = []
    for name, (change, _) in MISTAKES.items():
        folder = tmp_path / name
        folder.mkdir()
        printed += refusal(broken(folder, change))[0]
    for words in quoted:
        assert any(words in line for line in printed), f"the check prints no message with: {words}"


# ---------------------------------------------------------------------------------------------------
# The fit of the walk (slow: it samples)

@pytest.mark.slow
@needs_engine
def test_the_walk_fits_and_the_recorded_outputs_have_their_shape(doc, walk, monkeypatch, tmp_path):
    """The fit, the judges table, the value and the reviewer plan of the walk, run as written. The blocks marked
    `recorded` are compared with every number masked: the sampler does not give the same draws on every machine."""
    root, _ = walk
    recorded = dict(tagged(doc, "recorded"))
    assert set(recorded) == {"fit", "judges", "value", "labels", "not-blind"}
    monkeypatch.chdir(root)
    code, out, err = lm("eval", "fit", "my-study", *STUDY_ARGS, "--chains", "2")
    assert code == 1 and "--allow-simulated" in err and not (root / "my-study" / "fit.json").exists(), err
    code, out, err = lm("eval", "fit", "my-study", *STUDY_ARGS, "--allow-simulated", "--chains", "2")
    assert code == 0 and "converged: yes" in out, (out[-400:], err[-400:])
    assert shown_in(masked(recorded["fit"]), masked(out.splitlines())), out
    code, out, err = lm("eval", "judges", "my-study/fit.json")
    assert code == 0, err
    assert shown_in(masked(recorded["judges"]), masked(out.splitlines())), out
    code, out, err = lm("eval", "value", "my-study/fit.json", "--value", "my-study/value.json")
    assert code == 0, err
    assert masked(recorded["value"])[:-1] == masked(out.splitlines())[:-1], out   # the last line holds the decision
    assert out.splitlines()[-1].startswith("  candidate against baseline: value ")
    code, out, err = lm("eval", "labels", "next", "my-study/fit.json", "--mode", "pilot", "--n", "6", "--out",
                        "my-study/labels-001.json", "--form", "labels-001.csv")
    assert code == 0, err
    assert masked(recorded["labels"])[1:] == masked(out.splitlines())[1:], out
    assert "not blind" not in out                # neutral run ids: no note, although the study is simulated
    with open("labels-001.csv", encoding="utf-8", newline="") as handle:
        form = list(csv.DictReader(handle))
    plan = (root / "my-study" / "labels-001.json").read_text(encoding="utf-8")
    assert form and set(form[0]) == {"assignment", "run", "step", "component", "judge", "item", "of", "value",
                                     "observed_at"}
    assert all(re.fullmatch(r"r\d{3}", row["run"]) for row in form)
    for word in ("baseline", "candidate", "model-small", "model-large", "alpha", "beta"):
        assert word not in plan, f"the plan names {word}"

    # the filled form: without a `call` column every row is its own call; with one, a sitting is one call
    answer = {"correct": "1", "clarity": "ok", "depth": "6", "overall": "6"}
    filled = [{**row, "value": answer[row["component"]], "observed_at": "2026-03-05T10:00:00+00:00"} for row in form]
    assert len({row["assignment"] for row in filled}) < len(filled)       # the form groups rows by assignment
    for name, with_call in (("own", False), ("sittings", True)):
        study = tmp_path / name
        shutil.copytree(root / "my-study", study)
        rows = [{**row, "call": "sit-" + row["assignment"]} for row in filled] if with_call else filled
        with open(tmp_path / f"{name}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        code, out, err = lm("eval", "scores", "import", "--csv", tmp_path / f"{name}.csv", "--labels",
                            study / "labels-001.json", "--rubric", study / "rubric.json", "--out", study)
        assert code == 0, err
        new = [row for row in C.read_lines(study / C.SCORES_FILE, C.SCHEMA_SCORES) if "assignment" in row]
        assert len(new) == len(filled)
        calls = {row["call"] for row in new}
        assert len(calls) == (len({row["assignment"] for row in filled}) if with_call else len(filled)), name

    # the same study under run ids that name the arm. As real runs: the check passes, the plan is refused.
    # Marked simulated, as the examples are: the plan is written with a note that it is not blind.
    def named_real(tables):
        _arm_in_run_id(tables)
        _unmarked(tables)
    for name, change, flags in (("real", named_real, ()), ("simulated", _arm_in_run_id, ("--allow-simulated",))):
        (tmp_path / name).mkdir()
        named_study = broken(tmp_path / name, change, "study")
        assert check(named_study)[0] == 0
        code, out, err = lm("eval", "fit", named_study, "--rubric", named_study / "rubric.json", "--judges",
                            named_study / "judges.json", *flags, "--components", "depth", "--chains", "2",
                            "--draws", "300")
        assert code == 0, err
        code, out, err = lm("eval", "labels", "next", named_study / "fit.json", "--mode", "pilot", "--n", "6",
                            "--out", tmp_path / f"labels-{name}.json", "--allow-unconverged")
        if name == "real":
            assert code == 1 and masked(err.splitlines()) == masked(recorded["not-blind"]), err
        else:
            assert code == 0 and "note: the plan is not blind: the study is simulated" in out, (out, err)
